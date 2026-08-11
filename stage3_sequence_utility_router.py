#!/usr/bin/env python3
"""Stage 3: controlled sequence-level utility predictability experiment.

The four routers share the same frozen SASRec states, teacher, split, optimizer,
and early-stopping rule.  This is a diagnostic experiment, not a final model.
By default the existing Stage-1 probe supplies CE and log-rank utility labels.
Those labels are not OOF; the report records this limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from audit_multisource_utility import (
    DOMAINS, build_recbole_config, create_domain_dataset, create_domain_model,
    get_token_mapping, index_unique_by_user, load_checkpoint, read_inter,
    resolve_checkpoint, resolve_inter_path, token_to_inner_id,
)
from audit_ranking_utility import utility_arrays
from stage2_train_utility_router import (
    ProbeLabels, auroc, compute_probe_labels, load_stage1, save_json, spearman,
    subset,
)


ROUTERS = ("pooled_mlp", "pooled_larger_mlp", "sequence_pooling", "cross_attention")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target-dom", choices=("dom1", "dom2"), default="dom1")
    p.add_argument("--routers", nargs="+", choices=ROUTERS, default=list(ROUTERS))
    p.add_argument("--stage1-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--token-cache-dir", default="save/stage3_token_cache")
    p.add_argument("--checkpoint-template", default="checkpoint/SASRec-BEST-2025.tune.ckpt-{domain}-sim")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--encode-batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--max-train-users", type=int, default=20000)
    p.add_argument("--max-valid-users", type=int, default=5000)
    p.add_argument("--max-test-users", type=int, default=0)
    p.add_argument("--max-seq-len", type=int, default=50)
    p.add_argument("--shuffle-token-order", action="store_true")
    p.add_argument("--shuffle-source-pairs", action="store_true")
    p.add_argument("--reuse-token-cache", action="store_true")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@dataclass
class TokenData:
    users: np.ndarray
    states: np.ndarray
    lengths: np.ndarray


def save_tokens(path: Path, x: TokenData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, users=x.users, states=x.states.astype(np.float16), lengths=x.lengths)


def load_tokens(path: Path) -> TokenData:
    z = np.load(path, allow_pickle=False)
    return TokenData(z["users"], z["states"].astype(np.float32), z["lengths"].astype(np.int64))


@torch.no_grad()
def encode_token_rows(model: nn.Module, dataset: Any, rows: Sequence[Any], device: torch.device,
                      batch_size: int, cap: int) -> TokenData:
    model.eval(); mapping = get_token_mapping(dataset, model.ITEM_ID)
    width = min(int(model.max_seq_length), cap)
    users: List[str] = []; seqs: List[List[int]] = []
    for row in rows:
        try: ids = [token_to_inner_id(mapping, t) for t in row.seq[-width:]]
        except KeyError: continue
        if ids: users.append(row.user); seqs.append(ids)
    chunks: List[np.ndarray] = []
    for start in range(0, len(seqs), batch_size):
        cur = seqs[start:start + batch_size]
        padded = torch.zeros(len(cur), width, dtype=torch.long, device=device)
        for i, ids in enumerate(cur): padded[i, :len(ids)] = torch.as_tensor(ids, device=device)
        chunks.append(model.forward(padded).float().cpu().numpy())
    states = np.concatenate(chunks) if chunks else np.zeros((0, width, model.hidden_size), np.float32)
    return TokenData(np.asarray(users, dtype=str), states, np.asarray([len(x) for x in seqs], np.int64))


def ensure_token_caches(args: argparse.Namespace, root: Path, stage1_cfg: Dict[str, Any],
                        sources: Sequence[str], device: torch.device) -> Dict[Tuple[str, str], TokenData]:
    cache_root = (root / args.token_cache_dir).resolve()
    wanted = [(args.target_dom, s) for s in ("train", "valid", "test")]
    wanted += [(d, "train") for d in sources]
    out: Dict[Tuple[str, str], TokenData] = {}
    for domain, split in wanted:
        path = cache_root / f"{domain}_{split}_L{args.max_seq_len}.npz"
        if args.reuse_token_cache and path.exists():
            out[(domain, split)] = load_tokens(path); continue
        cfg = build_recbole_config(root, domain, args.seed, args.gpu_id)
        cfg["device"] = device; cfg["use_gpu"] = device.type == "cuda"
        ds = create_domain_dataset(cfg); model = create_domain_model(cfg, ds, device)
        ckpt = resolve_checkpoint(root, cfg, domain, args.checkpoint_template)
        load_checkpoint(model, ckpt, device, True)
        inter = resolve_inter_path(Path(stage1_cfg["dataset_dir"]), "BEST", domain, split)
        rows, _ = index_unique_by_user(read_inter(inter))
        print(f"[TOKENS] {domain}/{split}: {len(rows)} rows")
        data = encode_token_rows(model, ds, list(rows.values()), device, args.encode_batch_size, args.max_seq_len)
        save_tokens(path, data); out[(domain, split)] = data
        del model, ds
        if device.type == "cuda": torch.cuda.empty_cache()
    return out


def align_tokens(users: np.ndarray, target: TokenData, source: TokenData) -> Tuple[np.ndarray, ...]:
    ti = {str(u): i for i, u in enumerate(target.users)}; si = {str(u): i for i, u in enumerate(source.users)}
    missing = [str(u) for u in users if str(u) not in ti or str(u) not in si]
    if missing: raise RuntimeError(f"Token cache missing {len(missing)} aligned users; first={missing[:3]}")
    a = np.asarray([ti[str(u)] for u in users]); b = np.asarray([si[str(u)] for u in users])
    return target.states[a], target.lengths[a], source.states[b], source.lengths[b]


class PairDataset(Dataset):
    def __init__(self, ht, lt, hs, ls, ce, rank, domain, shuffle_tokens=False, shuffle_pairs=False, seed=0):
        n, s = ce.shape; rng = np.random.default_rng(seed)
        self.ht = np.repeat(ht[:, None], s, 1).reshape(n*s, *ht.shape[1:])
        self.lt = np.repeat(lt[:, None], s, 1).reshape(-1)
        self.hs = hs.reshape(n*s, *hs.shape[2:]); self.ls = ls.reshape(-1)
        self.ce = ce.reshape(-1); self.rank = rank.reshape(-1); self.domain = np.tile(np.arange(s), n)
        self.user_row = np.repeat(np.arange(n), s)
        if shuffle_pairs:
            for d in range(s):
                idx = np.flatnonzero(self.domain == d); perm = rng.permutation(idx)
                self.hs[idx], self.ls[idx] = self.hs[perm].copy(), self.ls[perm].copy()
        if shuffle_tokens:
            for arr, lens in ((self.ht, self.lt), (self.hs, self.ls)):
                for i, length in enumerate(lens): arr[i, :length] = arr[i, rng.permutation(length)]
    def __len__(self): return len(self.ce)
    def __getitem__(self, i):
        return (torch.from_numpy(self.ht[i]), self.lt[i], torch.from_numpy(self.hs[i]), self.ls[i],
                self.domain[i], self.ce[i], self.rank[i], self.user_row[i])


def last_at(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(len(x), device=x.device), lengths.long() - 1]


class Router(nn.Module):
    def __init__(self, kind: str, input_hidden: int, hidden: int, heads: int, domains: int, dropout: float):
        super().__init__(); self.kind = kind; self.domain = nn.Embedding(domains, 8)
        pooled_dim = 4 * input_hidden + 1 + 8
        seqpool_dim = 8 * input_hidden + 1 + 8
        if kind == "cross_attention":
            self.tproj = nn.Linear(input_hidden, hidden); self.sproj = nn.Linear(input_hidden, hidden)
            self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            dim = hidden + 2 * input_hidden + 1 + 8
        else: dim = seqpool_dim if kind == "sequence_pooling" else pooled_dim
        wide = hidden * 2 if kind == "pooled_larger_mlp" else hidden
        self.encoder = nn.Sequential(nn.Linear(dim, wide), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(wide, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.ce = nn.Linear(hidden, 1); self.rank = nn.Linear(hidden, 1)
    def forward(self, ht, lt, hs, ls, dom):
        a, b = last_at(ht, lt), last_at(hs, ls)
        cos = F.cosine_similarity(a, b).unsqueeze(1); de = self.domain(dom.long())
        if self.kind == "cross_attention":
            q, kv = self.tproj(ht), self.sproj(hs)
            pos = torch.arange(hs.shape[1], device=hs.device)[None]
            key_pad = pos >= ls[:, None]
            c, _ = self.attn(q, kv, kv, key_padding_mask=key_pad, need_weights=False)
            qmask = (pos < lt[:, None]).unsqueeze(-1); cp = (c*qmask).sum(1) / lt[:, None].clamp_min(1)
            z = torch.cat([cp, a, b, cos, de], 1)
        elif self.kind == "sequence_pooling":
            pos = torch.arange(ht.shape[1], device=ht.device)[None]
            mt, ms = pos < lt[:, None], pos < ls[:, None]
            mean_t = (ht*mt[..., None]).sum(1)/lt[:, None]; mean_s = (hs*ms[..., None]).sum(1)/ls[:, None]
            max_t = ht.masked_fill(~mt[..., None], -1e4).max(1).values
            max_s = hs.masked_fill(~ms[..., None], -1e4).max(1).values
            z = torch.cat([mean_t, max_t, mean_s, max_s, a, b, a*b, (a-b).abs(), cos, de], 1)
        else: z = torch.cat([a, b, a*b, (a-b).abs(), cos, de], 1)
        h = self.encoder(z); return self.ce(h).squeeze(1), self.rank(h).squeeze(1)


@torch.no_grad()
def predict(model, loader, device, n, s):
    model.eval(); ce = np.empty(n*s, np.float32); rank = np.empty(n*s, np.float32)
    offset = 0
    for ht, lt, hs, ls, dom, _, _, _ in loader:
        b = len(ht); pc, pr = model(ht.to(device), lt.to(device), hs.to(device), ls.to(device), dom.to(device))
        ce[offset:offset+b] = pc.cpu(); rank[offset:offset+b] = pr.cpu(); offset += b
    return ce.reshape(n, s), rank.reshape(n, s)


def metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    return {"spearman": spearman(pred, true), "sign_auroc": auroc(pred, true > 0),
            "best_source_accuracy": float((pred.argmax(1) == true.argmax(1)).mean())}


def downstream(pred: np.ndarray, labels: ProbeLabels, seed: int) -> Dict[str, float]:
    n, s = pred.shape; row = np.arange(n)
    # all_binary_masks uses bit order: zero=0 and single-source masks=1,2,4,...
    singles = np.asarray([1 << d for d in range(s)], dtype=np.int64)
    chosen = singles[pred.argmax(1)]
    rng = np.random.default_rng(seed); perm = singles[pred[rng.permutation(n)].argmax(1)]
    base = labels.losses[:, 0]; cp, pp = labels.losses[row, chosen], labels.losses[row, perm]
    cr, pr = labels.ranks[row, chosen], labels.ranks[row, perm]
    return {"ce_target_only": float(base.mean()), "ce_predicted": float(cp.mean()),
            "ce_permuted": float(pp.mean()), "recall10_predicted": float((cr <= 10).mean()),
            "recall10_permuted": float((pr <= 10).mean()),
            "ndcg10_predicted": float(np.where(cr <= 10, 1/np.log2(cr+1), 0).mean()),
            "ndcg10_permuted": float(np.where(pr <= 10, 1/np.log2(pr+1), 0).mean())}


def main() -> int:
    args = parse_args(); seed_all(args.seed); root = Path(__file__).resolve().parent
    device = torch.device(args.device)
    stage1 = Path(args.stage1_dir).resolve() if args.stage1_dir else root / "save/multisource_utility" / args.target_dom
    output = Path(args.output_dir).resolve() if args.output_dir else root / "save/stage3_sequence_router" / args.target_dom
    output.mkdir(parents=True, exist_ok=True)
    probe, splits, sources = load_stage1(stage1, args.target_dom, device)
    splits["train"] = subset(splits["train"], args.max_train_users, args.seed)
    splits["valid"] = subset(splits["valid"], args.max_valid_users, args.seed+1)
    splits["test"] = subset(splits["test"], args.max_test_users, args.seed+2)
    labels: Dict[str, ProbeLabels] = {}; utilities: Dict[str, Dict[str, np.ndarray]] = {}
    for sp in ("train", "valid", "test"):
        labels[sp] = compute_probe_labels(probe, splits[sp], device, args.eval_batch_size, 0)
        utilities[sp], _, _ = utility_arrays(labels[sp], len(sources))
    cfg = json.loads((stage1 / "config.json").read_text())
    cache = ensure_token_caches(args, root, cfg, sources, device)
    datasets = {}
    for sp in ("train", "valid", "test"):
        target_tokens = cache[(args.target_dom, sp)]; hss=[]; lss=[]; ht=lt=None
        for d in sources:
            ht, lt, hs, ls = align_tokens(splits[sp].users, target_tokens, cache[(d, "train")])
            hss.append(hs); lss.append(ls)
        datasets[sp] = PairDataset(ht, lt, np.stack(hss,1), np.stack(lss,1),
            utilities[sp]["ce"].astype(np.float32), utilities[sp]["rank"].astype(np.float32), sources,
            args.shuffle_token_order, args.shuffle_source_pairs, args.seed + {"train":0,"valid":1,"test":2}[sp])
    loaders = {sp: DataLoader(ds, batch_size=args.batch_size if sp=="train" else args.eval_batch_size,
                              shuffle=sp=="train") for sp, ds in datasets.items()}
    report = {"protocol": {"teacher": "existing frozen Stage-1 probe (NOT OOF)",
              "warning": "Run an OOF-teacher replication before treating positive results as final evidence.",
              "scope": "dom1/dom2 only; single-source routing"}, "args": vars(args), "sources": sources, "routers": {}}
    hdim = datasets["train"].ht.shape[-1]
    for kind in args.routers:
        seed_all(args.seed); model = Router(kind, hdim, args.hidden, args.heads, len(sources), args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best=float("inf"); bad=0; path=output/f"{kind}_best.pt"; history=[]
        for epoch in range(1, args.epochs+1):
            model.train(); total=count=0
            for ht,lt,hs,ls,dom,ce,rank,_ in loaders["train"]:
                ht,lt,hs,ls,dom,ce,rank = [x.to(device) for x in (ht,lt,hs,ls,dom,ce,rank)]
                opt.zero_grad(set_to_none=True); pc,pr=model(ht,lt,hs,ls,dom)
                loss=F.huber_loss(pc,ce)+F.huber_loss(pr,rank); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); total += loss.item()*len(ht); count += len(ht)
            pvc,pvr=predict(model,loaders["valid"],device,len(splits["valid"].users),len(sources))
            val=float(F.huber_loss(torch.from_numpy(pvc),torch.from_numpy(utilities["valid"]["ce"].astype(np.float32))) +
                      F.huber_loss(torch.from_numpy(pvr),torch.from_numpy(utilities["valid"]["rank"].astype(np.float32))))
            history.append({"epoch":epoch,"train_loss":total/count,"valid_joint_huber":val})
            print(f"[{kind} {epoch:02d}] train={total/count:.6f} valid={val:.6f}")
            if val < best-1e-7: best=val; bad=0; torch.save(model.state_dict(),path)
            else:
                bad+=1
                if bad>=args.patience: break
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        pc,pr=predict(model,loaders["test"],device,len(splits["test"].users),len(sources))
        report["routers"][kind] = {"parameters":sum(p.numel() for p in model.parameters()), "history":history,
            "ce_utility":metrics(pc,utilities["test"]["ce"]), "rank_utility":metrics(pr,utilities["test"]["rank"]),
            "routing_by_rank_head":downstream(pr,labels["test"],args.seed+99)}
        print(f"[DONE] {kind}: rank rho={report['routers'][kind]['rank_utility']['spearman']:.4f}")
    save_json(output/"stage3_report.json",report); print(f"[REPORT] {output/'stage3_report.json'}"); return 0


if __name__ == "__main__": raise SystemExit(main())
