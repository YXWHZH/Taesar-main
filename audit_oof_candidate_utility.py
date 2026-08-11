#!/usr/bin/env python3
"""Five-fold OOF teacher replication for P1 versus P5 candidate utility."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_candidate_conditioned_utility import features, seed_all, train_variant
from audit_multisource_utility import AlignedData, MaskFusionProbe, train_probe
from audit_ranking_utility import utility_arrays
from stage2_train_utility_router import compute_probe_labels, load_stage1, save_json, subset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-dom", choices=("dom1", "dom2"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--probe-patience", type=int, default=3)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-train-users", type=int, default=20000)
    p.add_argument("--max-valid-users", type=int, default=5000)
    p.add_argument("--max-test-users", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def take(data: AlignedData, idx: np.ndarray) -> AlignedData:
    return AlignedData(
        users=data.users[idx], h_target=data.h_target[idx], h_sources=data.h_sources[idx],
        available=data.available[idx], y_class=data.y_class[idx],
        target_seq_len=data.target_seq_len[idx], source_seq_lens=data.source_seq_lens[idx],
    )


def new_probe(base, n_sources, dropout, device):
    item = base.target_item_embeddings.detach()
    model = MaskFusionProbe(item.shape[1], n_sources, item, dropout=dropout,
                            temperature=base.temperature).to(device)
    return model


def average(parts):
    return {kind: np.mean(np.stack([x[kind] for x in parts], axis=0), axis=0)
            for kind in ("ce", "rank", "margin")}


def main():
    args = parse_args(); seed_all(args.seed)
    root = Path(__file__).resolve().parent
    device = torch.device(args.device)
    output = Path(args.output_dir).resolve() if args.output_dir else (
        root / "save" / "oof_candidate_utility" / args.target_dom)
    output.mkdir(parents=True, exist_ok=True)
    stage1 = root / "save" / "multisource_utility" / args.target_dom
    base, splits, sources = load_stage1(stage1, args.target_dom, device)
    splits["train"] = subset(splits["train"], args.max_train_users, args.seed)
    splits["valid"] = subset(splits["valid"], args.max_valid_users, args.seed + 1)
    splits["test"] = subset(splits["test"], args.max_test_users, args.seed + 2)

    n = len(splits["train"].users); rng = np.random.default_rng(args.seed + 700)
    order = rng.permutation(n); fold_ids = np.empty(n, np.int64)
    for f, idx in enumerate(np.array_split(order, args.folds)): fold_ids[idx] = f
    oof = {k: np.empty((n, len(sources)), np.float64) for k in ("ce", "rank", "margin")}
    ensemble = {"valid": [], "test": []}; fold_records = []

    probe_args = SimpleNamespace(
        batch_size=args.probe_batch_size, num_workers=0, lr=args.lr,
        weight_decay=args.weight_decay, masks_per_sample=2,
        epochs=args.probe_epochs, patience=args.probe_patience,
    )
    for fold in range(args.folds):
        seed = args.seed + 1000 + fold; seed_all(seed)
        held = np.flatnonzero(fold_ids == fold); remain = np.flatnonzero(fold_ids != fold)
        inner = np.random.default_rng(seed).permutation(remain)
        nv = max(1, min(2000, int(round(0.1 * len(inner)))))
        teacher_valid, teacher_train = inner[:nv], inner[nv:]
        fold_dir = output / "teachers" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        probe = new_probe(base, len(sources), args.dropout, device)
        history = train_probe(probe, take(splits["train"], teacher_train),
                              take(splits["train"], teacher_valid),
                              probe_args, device, fold_dir)
        held_u, _, _ = utility_arrays(
            compute_probe_labels(probe, take(splits["train"], held), device,
                                 args.eval_batch_size, 0), len(sources))
        for kind in oof: oof[kind][held] = held_u[kind]
        for sp in ("valid", "test"):
            u, _, _ = utility_arrays(
                compute_probe_labels(probe, splits[sp], device,
                                     args.eval_batch_size, 0), len(sources))
            ensemble[sp].append(u)
        fold_records.append({"fold": fold, "seed": seed, "heldout_users": len(held),
                             "teacher_train_users": len(teacher_train),
                             "teacher_valid_users": len(teacher_valid),
                             "best_valid_loss": min(x["valid_mask_avg_loss"] for x in history)})
        print(f"[OOF] {args.target_dom} fold={fold} heldout={len(held)}")
        del probe
        if device.type == "cuda": torch.cuda.empty_cache()

    utilities = {"train": oof, "valid": average(ensemble["valid"]),
                 "test": average(ensemble["test"])}
    item = base.target_item_embeddings.detach().cpu().numpy()
    report = {"protocol": {
        "teacher": f"{args.folds}-fold OOF mask-fusion probes",
        "router_train": "each user labeled only by a probe that excluded its fold",
        "router_valid_test": "mean utility from all fold probes; neither split used for probe fitting/early stopping",
        "candidate_embedding": "shared frozen target-expert item table",
        "comparison": "P1 candidate-source prior versus P5 personalized interactions",
    }, "args": vars(args), "sources": sources, "folds": fold_records, "variants": {}}
    for variant in ("P1_candidate_prior", "P5_interactions"):
        x = {sp: features(splits[sp], item, variant) for sp in ("train", "valid", "test")}
        report["variants"][variant] = train_variant(
            args, variant, x, utilities, device, output)
        print(f"[DONE] {args.target_dom}/{variant} "
              f"rank_rho={report['variants'][variant]['rank_utility']['spearman']:.4f}")
    save_json(output / "oof_candidate_report.json", report)
    print(json.dumps({v: report["variants"][v]["rank_utility"] for v in report["variants"]},
                     indent=2))
    print(f"[REPORT] {output / 'oof_candidate_report.json'}")


if __name__ == "__main__":
    main()
