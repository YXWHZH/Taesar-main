#!/usr/bin/env python3
"""Stage 2: train and evaluate a leakage-safe utility router.

The script consumes the frozen embeddings and fusion probe produced by
``audit_multisource_utility.py``.  Utility labels are computed with the frozen
probe on train/valid/test, but test labels are *never* passed to the router or
the mask-selection code.  They are used only after decisions have been frozen
to report utility-prediction and recommendation metrics.

Example
-------
python stage2_train_utility_router.py --target-dom dom1 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from audit_multisource_utility import (
    DOMAINS,
    AlignedData,
    MaskFusionProbe,
    ProbeDataset,
    align_for_target,
    all_binary_masks,
    load_encoded,
    mask_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a marginal-utility + synergy sparse router.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target-dom", choices=DOMAINS, default="dom1")
    p.add_argument("--stage1-dir", default=None,
                   help="Default: save/multisource_utility/<target-dom>")
    p.add_argument("--output-dir", default=None,
                   help="Default: save/stage2_utility_router/<target-dom>")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--rank-weight", type=float, default=0.5)
    p.add_argument("--sign-weight", type=float, default=0.5)
    p.add_argument("--synergy-weight", type=float, default=0.5)
    p.add_argument("--k-max", type=int, choices=(1, 2), default=2)
    p.add_argument("--threshold-grid-size", type=int, default=101)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-train-users", type=int, default=0)
    p.add_argument("--max-valid-users", type=int, default=0)
    p.add_argument("--max-test-users", type=int, default=0)
    p.add_argument("--no-synergy", action="store_true",
                   help="Use additive single-source utilities for pairs.")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: Any) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(value), indent=2, ensure_ascii=False), encoding="utf-8")


def torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def subset(data: AlignedData, n: int, seed: int) -> AlignedData:
    if n <= 0 or len(data.users) <= n:
        return data
    idx = np.sort(np.random.default_rng(seed).choice(len(data.users), n, replace=False))
    return AlignedData(
        users=data.users[idx], h_target=data.h_target[idx],
        h_sources=data.h_sources[idx], available=data.available[idx],
        y_class=data.y_class[idx], target_seq_len=data.target_seq_len[idx],
        source_seq_lens=data.source_seq_lens[idx],
    )


def load_stage1(stage1_dir: Path, target: str, device: torch.device
                ) -> Tuple[MaskFusionProbe, Dict[str, AlignedData], List[str]]:
    cfg = json.loads((stage1_dir / "config.json").read_text(encoding="utf-8"))
    sources = list(cfg.get("source_domains") or [d for d in DOMAINS if d != target])
    cache = stage1_dir / "cache"
    source_encoded = {d: load_encoded(cache / f"{d}_train_embeddings.npz") for d in sources}
    splits: Dict[str, AlignedData] = {}
    for split in ("train", "valid", "test"):
        target_encoded = load_encoded(cache / f"{target}_{split}_embeddings.npz")
        # Stage 1 local IDs are padding=0 and real target items=1..item_num-1.
        max_id = int(target_encoded.target_inner_ids.max())
        data, _ = align_for_target(target_encoded, source_encoded, sources, 1, max_id, True)
        splits[split] = data

    checkpoint = torch_load(stage1_dir / "probe_best.pt", device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    item_w = state["target_item_embeddings"]
    hidden = int(item_w.shape[1])
    # Dropout/temperature do not affect eval; read them from Stage 1 config when present.
    args1 = cfg.get("args", {})
    probe = MaskFusionProbe(hidden, len(sources), item_w,
                            dropout=float(args1.get("dropout", 0.1)),
                            temperature=float(args1.get("temperature", 1.0))).to(device)
    probe.load_state_dict(state, strict=True)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe, splits, sources


@dataclass
class ProbeLabels:
    losses: np.ndarray                 # [N, all masks]
    ranks: np.ndarray                  # [N, all masks]
    utilities: np.ndarray              # [N,S]
    synergies: np.ndarray              # [N,P]


@torch.no_grad()
def compute_probe_labels(probe: MaskFusionProbe, data: AlignedData, device: torch.device,
                         batch_size: int, workers: int) -> ProbeLabels:
    masks = all_binary_masks(data.h_sources.shape[1])
    mask_np = masks.numpy().astype(int)
    zero = int(np.flatnonzero(mask_np.sum(1) == 0)[0])
    singles = [int(np.flatnonzero((mask_np.sum(1) == 1) & (mask_np[:, d] == 1))[0])
               for d in range(mask_np.shape[1])]
    pairs = list(combinations(range(mask_np.shape[1]), 2))
    pair_indices = [int(np.flatnonzero((mask_np[:, i] == 1) & (mask_np[:, j] == 1)
                                      & (mask_np.sum(1) == 2))[0]) for i, j in pairs]
    out_l, out_r = [], []
    loader = DataLoader(ProbeDataset(data), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    for ht, hs, avail, y, _ in loader:
        ht, hs, avail, y = ht.to(device), hs.to(device), avail.to(device), y.to(device)
        ls, rs = [], []
        for m0 in masks:
            m = m0.to(device).expand(len(y), -1)
            logits = probe(ht, hs, m, avail)
            ls.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
            score = logits.gather(1, y[:, None]).squeeze(1)
            rs.append((1 + (logits > score[:, None]).sum(1)).cpu().numpy())
        out_l.append(np.stack(ls, 1)); out_r.append(np.stack(rs, 1))
    losses, ranks = np.concatenate(out_l), np.concatenate(out_r)
    utilities = losses[:, zero, None] - losses[:, singles]
    pair_u = losses[:, zero, None] - losses[:, pair_indices]
    synergies = np.stack([pair_u[:, k] - utilities[:, i] - utilities[:, j]
                          for k, (i, j) in enumerate(pairs)], axis=1)
    return ProbeLabels(losses, ranks, utilities.astype(np.float32), synergies.astype(np.float32))


def raw_features(data: AlignedData) -> np.ndarray:
    """z=[hT,hS,hT*hS,|hT-hS|,cos,log lengths,domain one-hot]."""
    n, s, _ = data.h_sources.shape
    ht = np.repeat(data.h_target[:, None, :], s, axis=1)
    hs = data.h_sources
    cos = np.sum(ht * hs, axis=2, keepdims=True) / (
        np.linalg.norm(ht, axis=2, keepdims=True) * np.linalg.norm(hs, axis=2, keepdims=True) + 1e-8)
    tl = np.log1p(data.target_seq_len).astype(np.float32)[:, None, None]
    tl = np.repeat(tl, s, axis=1)
    sl = np.log1p(data.source_seq_lens).astype(np.float32)[:, :, None]
    domain = np.repeat(np.eye(s, dtype=np.float32)[None, :, :], n, axis=0)
    return np.concatenate([ht, hs, ht * hs, np.abs(ht - hs), cos, tl, sl, domain], 2).astype(np.float32)


class RouterData(Dataset):
    def __init__(self, x: np.ndarray, labels: ProbeLabels):
        self.x = torch.from_numpy(x)
        self.u = torch.from_numpy(labels.utilities)
        self.i = torch.from_numpy(labels.synergies)
    def __len__(self) -> int: return len(self.x)
    def __getitem__(self, k: int): return self.x[k], self.u[k], self.i[k]


class UtilityRouter(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, dropout: float, n_sources: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout))
        self.utility = nn.Linear(h2, 1)
        self.sign = nn.Linear(h2, 1)
        self.pairs = list(combinations(range(n_sources), 2))
        self.synergy = nn.Sequential(nn.Linear(4 * h2, h2), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(h2, 1))
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        u, sign = self.utility(h).squeeze(-1), self.sign(h).squeeze(-1)
        pair_h = [torch.cat([h[:, i], h[:, j], h[:, i] * h[:, j],
                            torch.abs(h[:, i] - h[:, j])], -1) for i, j in self.pairs]
        synergy = torch.cat([self.synergy(v) for v in pair_h], 1)
        return u, sign, synergy


def router_loss(pred: Tuple[torch.Tensor, ...], u: torch.Tensor, synergy: torch.Tensor,
                args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    pu, sign, pi = pred
    reg = F.huber_loss(pu, u)
    diffs_p, diffs_y = [], []
    for i, j in combinations(range(u.shape[1]), 2):
        diffs_p.append(pu[:, i] - pu[:, j]); diffs_y.append(u[:, i] - u[:, j])
    dp, dy = torch.stack(diffs_p, 1), torch.stack(diffs_y, 1)
    rank = F.softplus(-dp * torch.sign(dy)).mean()
    sign_loss = F.binary_cross_entropy_with_logits(sign, (u > 0).float())
    syn = F.huber_loss(pi, synergy) if not args.no_synergy else pu.new_zeros(())
    total = reg + args.rank_weight * rank + args.sign_weight * sign_loss + args.synergy_weight * syn
    return total, {"reg": reg.item(), "rank": rank.item(), "sign": sign_loss.item(), "synergy": syn.item()}


@torch.no_grad()
def predict(model: UtilityRouter, x: np.ndarray, device: torch.device, batch: int
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); us, ss, ii = [], [], []
    tx = torch.from_numpy(x)
    for start in range(0, len(tx), batch):
        u, s, i = model(tx[start:start + batch].to(device))
        us.append(u.cpu().numpy()); ss.append(torch.sigmoid(s).cpu().numpy()); ii.append(i.cpu().numpy())
    return np.concatenate(us), np.concatenate(ss), np.concatenate(ii)


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort"); ranks = np.empty(len(x), float)
    sorted_x = x[order]; start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a.ravel()), rankdata(b.ravel())
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool).ravel(); scores = scores.ravel()
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 - 1) / 2.0) / (n1 * n0))


def mask_helpers(s: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    return all_binary_masks(s).numpy().astype(int), list(combinations(range(s), 2))


def select_masks(pu: np.ndarray, pi: np.ndarray, threshold: float, kmax: int,
                 use_synergy: bool) -> np.ndarray:
    n, s = pu.shape; masks, pairs = mask_helpers(s)
    allowed = np.flatnonzero(masks.sum(1) <= kmax)
    scores = np.zeros((n, len(allowed)), dtype=np.float32)
    pair_to_col = {p: k for k, p in enumerate(pairs)}
    for c, mi in enumerate(allowed):
        active = np.flatnonzero(masks[mi])
        scores[:, c] = pu[:, active].sum(1) - threshold * len(active)
        if use_synergy and len(active) == 2:
            scores[:, c] += pi[:, pair_to_col[tuple(active)]]
    return allowed[np.argmax(scores, axis=1)]


def tune_threshold(pu: np.ndarray, pi: np.ndarray, labels: ProbeLabels, args: argparse.Namespace
                   ) -> Tuple[float, float]:
    scale = max(float(np.std(pu)), 1e-4)
    grid = np.linspace(-2 * scale, 2 * scale, args.threshold_grid_size)
    row = np.arange(len(pu)); best = (float("inf"), 0.0)
    for t in grid:
        chosen = select_masks(pu, pi, float(t), args.k_max, not args.no_synergy)
        loss = float(labels.losses[row, chosen].mean())
        if loss < best[0]: best = (loss, float(t))
    return best[1], best[0]


def recommendation_metrics(ranks: np.ndarray) -> Dict[str, float]:
    ranks = ranks.astype(float)
    return {"Recall@10": float((ranks <= 10).mean()),
            "NDCG@10": float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean()),
            "MRR": float((1 / ranks).mean())}


def evaluate(split: str, data: AlignedData, labels: ProbeLabels, pu: np.ndarray,
             psign: np.ndarray, pi: np.ndarray, threshold: float, args: argparse.Namespace,
             sources: Sequence[str], output: Path) -> Dict[str, Any]:
    masks, _ = mask_helpers(len(sources)); row = np.arange(len(data.users))
    chosen = select_masks(pu, pi, threshold, args.k_max, not args.no_synergy)
    zero = int(np.flatnonzero(masks.sum(1) == 0)[0]); dense = int(np.flatnonzero(masks.sum(1) == len(sources))[0])
    allowed = np.flatnonzero(masks.sum(1) <= args.k_max)
    oracle = allowed[np.argmin(labels.losses[:, allowed], axis=1)]
    chosen_loss = labels.losses[row, chosen]; dense_loss = labels.losses[:, dense]
    oracle_loss = labels.losses[row, oracle]
    denom = float(np.mean(dense_loss - oracle_loss))
    ocr = float(np.mean(dense_loss - chosen_loss) / denom) if abs(denom) > 1e-12 else None
    true_best = np.argmax(labels.utilities, 1); pred_best = np.argmax(pu, 1)
    regret = labels.losses[row, chosen] - labels.losses[row, oracle]
    result = {
        "split": split, "n_users": len(data.users), "threshold": threshold,
        "utility_spearman": spearman(pu, labels.utilities),
        "sign_auroc": auroc(psign, labels.utilities > 0),
        "best_source_accuracy": float(np.mean(true_best == pred_best)),
        "mean_regret_vs_safe_oracle_k": float(regret.mean()),
        "median_regret_vs_safe_oracle_k": float(np.median(regret)),
        "oracle_capture_ratio": ocr,
        "ce": {"target_only": float(labels.losses[:, zero].mean()),
               "dense": float(dense_loss.mean()), "predicted_router": float(chosen_loss.mean()),
               "safe_oracle_k": float(oracle_loss.mean())},
        "ranking": {"target_only": recommendation_metrics(labels.ranks[:, zero]),
                    "dense": recommendation_metrics(labels.ranks[:, dense]),
                    "predicted_router": recommendation_metrics(labels.ranks[row, chosen]),
                    "safe_oracle_k": recommendation_metrics(labels.ranks[row, oracle])},
        "selection": {mask_name(m, sources): int(np.sum(chosen == i)) for i, m in enumerate(masks)},
    }
    with (output / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["user_id", *[f"true_utility_{d}" for d in sources],
                                      *[f"pred_utility_{d}" for d in sources], "selected_mask",
                                      "selected_ce", "oracle_k_ce", "regret"])
        for k, user in enumerate(data.users):
            w.writerow([str(user), *labels.utilities[k].tolist(), *pu[k].tolist(),
                        mask_name(masks[chosen[k]], sources), float(chosen_loss[k]),
                        float(oracle_loss[k]), float(regret[k])])
    return result


def main() -> int:
    args = parse_args(); seed_everything(args.seed)
    root = Path(__file__).resolve().parent
    stage1 = Path(args.stage1_dir) if args.stage1_dir else root / "save/multisource_utility" / args.target_dom
    output = Path(args.output_dir) if args.output_dir else root / "save/stage2_utility_router" / args.target_dom
    stage1, output = stage1.resolve(), output.resolve(); output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    print(f"[LOAD] Stage 1: {stage1}")
    probe, splits, sources = load_stage1(stage1, args.target_dom, device)
    splits["train"] = subset(splits["train"], args.max_train_users, args.seed)
    splits["valid"] = subset(splits["valid"], args.max_valid_users, args.seed + 1)
    splits["test"] = subset(splits["test"], args.max_test_users, args.seed + 2)
    labels: Dict[str, ProbeLabels] = {}
    features: Dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        print(f"[LABEL] {split}: {len(splits[split].users)} users")
        labels[split] = compute_probe_labels(probe, splits[split], device, args.eval_batch_size, args.num_workers)
        features[split] = raw_features(splits[split])
    # Normalize from train only. Domain one-hot is intentionally normalized too; zero-variance is guarded.
    mean = features["train"].reshape(-1, features["train"].shape[-1]).mean(0)
    std = features["train"].reshape(-1, features["train"].shape[-1]).std(0)
    std[std < 1e-6] = 1.0
    for split in features: features[split] = ((features[split] - mean) / std).astype(np.float32)

    model = UtilityRouter(features["train"].shape[-1], args.hidden1, args.hidden2,
                          args.dropout, len(sources)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(RouterData(features["train"], labels["train"]), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    best, bad, history = float("inf"), 0, []
    best_path = output / "router_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train(); total, count = 0.0, 0
        for x, u, syn in loader:
            x, u, syn = x.to(device), u.to(device), syn.to(device)
            opt.zero_grad(set_to_none=True); loss, _ = router_loss(model(x), u, syn, args)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += loss.item() * len(x); count += len(x)
        vu, _, vi = predict(model, features["valid"], device, args.eval_batch_size)
        val = F.huber_loss(torch.from_numpy(vu), torch.from_numpy(labels["valid"].utilities)).item()
        rec = {"epoch": epoch, "train_loss": total / count, "valid_utility_huber": val}; history.append(rec)
        print(f"[EPOCH {epoch:03d}] train={rec['train_loss']:.6f} valid_huber={val:.6f}")
        if val < best - 1e-7:
            best, bad = val, 0
            torch.save({"state_dict": model.state_dict(), "feature_mean": mean,
                        "feature_std": std, "args": vars(args), "sources": sources}, best_path)
        else:
            bad += 1
            if bad >= args.patience: print("[EARLY STOP]"); break
    model.load_state_dict(torch_load(best_path, device)["state_dict"])
    predictions = {sp: predict(model, features[sp], device, args.eval_batch_size)
                   for sp in ("valid", "test")}
    threshold, valid_ce = tune_threshold(predictions["valid"][0], predictions["valid"][2],
                                         labels["valid"], args)
    print(f"[VALID] selected threshold={threshold:+.6f}, CE={valid_ce:.6f}")
    results = {sp: evaluate(sp, splits[sp], labels[sp], *predictions[sp], threshold,
                           args, sources, output) for sp in ("valid", "test")}
    report = {"protocol": {"train": "fit router and feature normalization",
                           "valid": "early stopping and transfer threshold selection",
                           "test": "predicted masks frozen before labels are used for evaluation"},
              "args": vars(args), "sources": sources, "history": history, "results": results}
    save_json(output / "stage2_report.json", report)
    save_json(output / "config.json", {"args": vars(args), "stage1_dir": str(stage1), "sources": sources})
    print(json.dumps(results["test"], indent=2, ensure_ascii=False))
    print(f"[DONE] {output / 'stage2_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
