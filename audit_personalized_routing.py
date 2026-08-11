#!/usr/bin/env python3
"""Audit whether Stage-2 routing gains depend on user-mask correspondence."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from audit_multisource_utility import all_binary_masks, mask_name
from stage2_train_utility_router import (
    compute_probe_labels,
    load_stage1,
    recommendation_metrics,
    subset,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-dom", required=True, choices=["dom1", "dom2", "dom3", "dom4"])
    p.add_argument("--stage1-dir")
    p.add_argument("--stage2-dir")
    p.add_argument("--output-dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--permutation-seed", type=int, default=3407)
    return p.parse_args()


def read_selected(path, expected_users, masks, sources):
    lookup = {mask_name(m, sources): i for i, m in enumerate(masks)}
    users, selected = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            users.append(str(row["user_id"]))
            selected.append(lookup[row["selected_mask"]])
    if users != [str(x) for x in expected_users]:
        raise RuntimeError(f"User/order mismatch in {path}")
    return np.asarray(selected, dtype=int)


def metrics(labels, chosen):
    row = np.arange(len(chosen))
    return {
        "ce": float(labels.losses[row, chosen].mean()),
        "ranking": recommendation_metrics(labels.ranks[row, chosen]),
    }


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    stage1 = Path(args.stage1_dir) if args.stage1_dir else root / "save/multisource_utility" / args.target_dom
    stage2 = Path(args.stage2_dir) if args.stage2_dir else root / "save/stage2_utility_router" / args.target_dom
    output = Path(args.output_dir) if args.output_dir else root / "save/personalized_routing_audit" / args.target_dom
    output.mkdir(parents=True, exist_ok=True)

    report2 = json.loads((stage2 / "stage2_report.json").read_text(encoding="utf-8"))
    cfg = report2["args"]
    device = torch.device(args.device)
    probe, splits, sources = load_stage1(stage1, args.target_dom, device)
    seed = int(cfg["seed"])
    splits["valid"] = subset(splits["valid"], int(cfg["max_valid_users"]), seed + 1)
    splits["test"] = subset(splits["test"], int(cfg["max_test_users"]), seed + 2)
    labels = {
        sp: compute_probe_labels(probe, splits[sp], device, int(cfg["eval_batch_size"]), int(cfg["num_workers"]))
        for sp in ("valid", "test")
    }

    masks = all_binary_masks(len(sources)).numpy().astype(int)
    allowed = np.flatnonzero(masks.sum(1) <= int(cfg["k_max"]))
    singles = np.flatnonzero(masks.sum(1) == 1)
    valid_mean = labels["valid"].losses.mean(0)
    fixed_mask = int(allowed[np.argmin(valid_mean[allowed])])
    fixed_single = int(singles[np.argmin(valid_mean[singles])])
    test_selected = read_selected(stage2 / "predictions_test.csv", splits["test"].users, masks, sources)
    predicted = metrics(labels["test"], test_selected)

    rng = np.random.default_rng(args.permutation_seed)
    row = np.arange(len(test_selected))
    perm_ce = np.empty(args.permutations, dtype=float)
    perm_r10 = np.empty(args.permutations, dtype=float)
    for i in range(args.permutations):
        shuffled = rng.permutation(test_selected)
        perm_ce[i] = labels["test"].losses[row, shuffled].mean()
        perm_r10[i] = (labels["test"].ranks[row, shuffled] <= 10).mean()

    dense_idx = int(np.flatnonzero(masks.sum(1) == len(sources))[0])
    zero_idx = int(np.flatnonzero(masks.sum(1) == 0)[0])
    result = {
        "target_domain": args.target_dom,
        "n_test_users": len(test_selected),
        "protocol": {
            "fixed_masks_selected_on": "validation only",
            "permutation": "test selected masks shuffled across users; exact mask counts preserved",
            "permutations": args.permutations,
            "permutation_seed": args.permutation_seed,
        },
        "predicted_router": predicted,
        "fixed_best_mask": {"mask": mask_name(masks[fixed_mask], sources), **metrics(labels["test"], np.full(len(test_selected), fixed_mask))},
        "fixed_best_single_source": {"mask": mask_name(masks[fixed_single], sources), **metrics(labels["test"], np.full(len(test_selected), fixed_single))},
        "target_only": metrics(labels["test"], np.full(len(test_selected), zero_idx)),
        "dense": metrics(labels["test"], np.full(len(test_selected), dense_idx)),
        "permuted_router": {
            "ce_mean": float(perm_ce.mean()), "ce_std": float(perm_ce.std(ddof=1)),
            "ce_p05": float(np.quantile(perm_ce, .05)), "ce_p95": float(np.quantile(perm_ce, .95)),
            "recall10_mean": float(perm_r10.mean()), "recall10_std": float(perm_r10.std(ddof=1)),
            "personalized_ce_advantage": float(perm_ce.mean() - predicted["ce"]),
            "one_sided_p_permuted_ce_le_predicted": float((1 + np.sum(perm_ce <= predicted["ce"])) / (args.permutations + 1)),
        },
        "selection": {mask_name(m, sources): int(np.sum(test_selected == i)) for i, m in enumerate(masks)},
    }
    out = output / "personalized_routing_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
