#!/usr/bin/env python3
"""Compare CE, target-rank, and full-negative margin utility across probe seeds."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from stage2_train_utility_router import compute_probe_labels, load_stage1


SEEDS = (2025, 2026, 2027)
DOMAINS = ("dom1", "dom2", "dom3", "dom4")


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--output-dir", default="save/ranking_utility_audit")
    return p.parse_args()


def seed_dir(root, seed, target):
    if seed == 2025:
        return root / "save/multisource_utility" / target
    return root / "save/multisource_utility_stability" / f"seed_{seed}" / target


def margin_from_ce(x):
    # CE = log(1 + sum_neg exp(s_j-s_y)); margin = -log(sum_neg exp(s_j-s_y)).
    # Stable equivalent of -log(expm1(CE)).
    x = np.asarray(x, dtype=np.float64)
    return -x - np.log1p(-np.exp(-x))


def utility_arrays(labels, n_sources):
    masks = np.asarray([[int((i >> d) & 1) for d in range(n_sources)] for i in range(2 ** n_sources)])
    zero = int(np.flatnonzero(masks.sum(1) == 0)[0])
    singles = [int(np.flatnonzero((masks.sum(1) == 1) & (masks[:, d] == 1))[0]) for d in range(n_sources)]
    l0, r0 = labels.losses[:, zero, None], labels.ranks[:, zero, None]
    ls, rs = labels.losses[:, singles], labels.ranks[:, singles]
    return {
        "ce": l0 - ls,
        "rank": np.log1p(r0.astype(np.float64)) - np.log1p(rs.astype(np.float64)),
        "margin": margin_from_ce(ls) - margin_from_ce(l0),
    }, rs, r0


def pair_stability(a, b):
    return {
        "spearman": float(spearmanr(a.ravel(), b.ravel()).statistic),
        "sign_agreement": float(np.mean((a > 0) == (b > 0))),
        "best_source_agreement": float(np.mean(np.argmax(a, 1) == np.argmax(b, 1))),
    }


def auroc(scores, labels):
    scores, labels = np.asarray(scores).ravel(), np.asarray(labels, dtype=bool).ravel()
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_router_predictions(path, users, sources):
    got_users, values = [], []
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for row in rows:
            got_users.append(str(row["user_id"]))
            values.append([float(row[f"pred_utility_{s}"]) for s in sources])
    if got_users != [str(u) for u in users]:
        raise RuntimeError(f"Router/test user mismatch: {path}")
    return np.asarray(values)


def main():
    args = args_parser()
    root = Path(__file__).resolve().parent
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    reports = {}

    for target in args.targets:
        by_seed, users_by_seed, sources = {}, {}, None
        for seed in SEEDS:
            probe, splits, current_sources = load_stage1(seed_dir(root, seed, target), target, device)
            probe_labels = compute_probe_labels(probe, splits["test"], device, args.eval_batch_size, 0)
            by_seed[seed], _, _ = utility_arrays(probe_labels, len(current_sources))
            users_by_seed[seed] = [str(x) for x in splits["test"].users]
            sources = current_sources
        if not all(users_by_seed[s] == users_by_seed[2025] for s in SEEDS):
            raise RuntimeError(f"Cross-seed test users differ for {target}")

        stability = {}
        for kind in ("ce", "rank", "margin"):
            stability[kind] = {
                f"{a}-{b}": pair_stability(by_seed[a][kind], by_seed[b][kind])
                for a, b in combinations(SEEDS, 2)
            }

        u = by_seed[2025]
        rank_positive = u["rank"] > 0
        top10_improve = None
        # A rank crosses into Top-10 iff rank utility is positive enough; reconstruct from labels again.
        probe, splits, _ = load_stage1(seed_dir(root, 2025, target), target, device)
        labels = compute_probe_labels(probe, splits["test"], device, args.eval_batch_size, 0)
        _, single_ranks, base_ranks = utility_arrays(labels, len(sources))
        top10_improve = (base_ranks > 10) & (single_ranks <= 10)
        top10_harm = (base_ranks <= 10) & (single_ranks > 10)

        alignment = {}
        for kind in ("ce", "margin"):
            alignment[kind] = {
                "spearman_with_rank_utility": float(spearmanr(u[kind].ravel(), u["rank"].ravel()).statistic),
                "sign_agreement_with_rank_utility": float(np.mean((u[kind] > 0) == rank_positive)),
                "auroc_rank_improves": auroc(u[kind], rank_positive),
                "mean_on_top10_entry": float(u[kind][top10_improve].mean()) if top10_improve.any() else None,
                "mean_on_top10_exit": float(u[kind][top10_harm].mean()) if top10_harm.any() else None,
            }

        pred = load_router_predictions(root / "save/stage2_utility_router" / target / "predictions_test.csv", splits["test"].users, sources)
        predictability = {
            kind: {
                "spearman": float(spearmanr(pred.ravel(), u[kind].ravel()).statistic),
                "sign_auroc": auroc(pred, u[kind] > 0),
                "best_source_accuracy": float(np.mean(np.argmax(pred, 1) == np.argmax(u[kind], 1))),
            }
            for kind in ("ce", "rank", "margin")
        }
        report = {
            "target": target, "n_users": len(users_by_seed[2025]), "sources": sources,
            "definitions": {
                "ce": "CE(target_only)-CE(single_source)",
                "rank": "log(1+rank_target_only)-log(1+rank_single_source)",
                "margin": "full-negative logit margin(single_source)-margin(target_only), derived exactly from full-softmax CE",
            },
            "seed_stability": stability,
            "alignment_with_ranking_2025": alignment,
            "existing_router_predictability_2025": predictability,
            "top10_events": {"entries": int(top10_improve.sum()), "exits": int(top10_harm.sum())},
        }
        reports[target] = report
        target_out = output / target
        target_out.mkdir(parents=True, exist_ok=True)
        (target_out / "ranking_utility_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[DONE] {target}")

    (output / "ranking_utility_audit_all.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
