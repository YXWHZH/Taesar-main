#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_multisource_cdsr.py

Multi-source CDSR feasibility audit for Taesar/RecBole-style *.inter files.

What it answers
---------------
1) How much user overlap exists across dom1..dom4?
2) For each target domain, how many target users have 0/1/2/3 usable source domains?
3) How much history does each user have in each domain?
4) Are timestamps available and internally consistent enough for a temporal router?
5) For each target-domain sample, which source domains have history available BEFORE
   the target timestamp (to avoid future-information leakage)?

The script is READ-ONLY:
- It does not modify any *.inter file.
- It does not build/retrain a RecBole model.
- It does not require scipy.
- Dependencies: Python stdlib + numpy + pandas.

Expected default layout
-----------------------
<project_root>/
  dataset/BEST/
    BEST.dom1.train.inter
    BEST.dom1.valid.inter
    BEST.dom1.test.inter
    ...
    BEST.dom4.train.inter
    BEST.dom4.valid.inter
    BEST.dom4.test.inter

It also tolerates "val" instead of "valid".

Example
-------
python audit_multisource_cdsr.py

python audit_multisource_cdsr.py \
  --dataset-dir dataset/BEST \
  --prefix BEST \
  --domains dom1 dom2 dom3 dom4 \
  --domain-names Books Electronics Sports Tools \
  --splits train valid test \
  --output-dir save/multisource_audit

Important interpretation note
-----------------------------
Cross-domain overlap is meaningful only if user_id is globally consistent across
domains. The script cannot prove semantic identity of IDs; it reports this assumption
explicitly in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

TYPE_SUFFIX_RE = re.compile(r":(?:token|token_seq|float|float_seq)$", re.IGNORECASE)


def strip_type(field: str) -> str:
    return TYPE_SUFFIX_RE.sub("", field.strip())


def parse_token_seq(value: Any) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return s.split()


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_float_seq(value: Any) -> List[float]:
    out = []
    for tok in parse_token_seq(value):
        v = safe_float(tok)
        if v is not None:
            out.append(v)
    return out


def q(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, p))


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def ratio(num: int, den: int) -> Optional[float]:
    return float(num / den) if den else None


def fmt_pct(x: Optional[float]) -> str:
    return "NA" if x is None else f"{100*x:.2f}%"


def fmt_num(x: Optional[float], digits: int = 3) -> str:
    return "NA" if x is None else f"{x:.{digits}f}"


def ensure_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): ensure_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ensure_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ---------------------------------------------------------------------------
# Schema / record definitions
# ---------------------------------------------------------------------------

USER_CANDIDATES = [
    "user_id", "uid", "user", "user_token"
]

ITEM_SEQ_CANDIDATES = [
    "item_id_list", "item_list", "item_seq", "item_id_seq",
    "sequence", "seq", "items"
]

TARGET_ITEM_CANDIDATES = [
    "item_id", "target_item", "target_item_id", "next_item"
]

TIME_SEQ_CANDIDATES = [
    "timestamp_list", "time_list", "timestamp_seq", "time_seq",
    "timestamps"
]

TARGET_TIME_CANDIDATES = [
    "timestamp", "target_timestamp", "target_time", "time"
]

LENGTH_CANDIDATES = [
    "item_length", "item_seq_len", "seq_len", "length"
]


def first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


@dataclass
class Schema:
    user: str
    item_seq: Optional[str]
    target_item: Optional[str]
    time_seq: Optional[str]
    target_time: Optional[str]
    seq_len_field: Optional[str]
    columns: List[str]


@dataclass
class RowRecord:
    user_id: str
    domain: str
    split: str
    seq_len: int
    target_item: Optional[str]
    target_time: Optional[float]
    seq_times: List[float]
    seq_times_raw_len: int
    seq_items_raw_len: int
    has_time_seq: bool
    time_item_len_match: Optional[bool]
    time_monotonic: Optional[bool]
    first_time: Optional[float]
    last_time: Optional[float]


# ---------------------------------------------------------------------------
# File discovery / parsing
# ---------------------------------------------------------------------------

def resolve_inter_file(dataset_dir: Path, prefix: str, domain: str, split: str) -> Optional[Path]:
    aliases = [split]
    if split == "valid":
        aliases += ["val", "validation"]
    elif split == "val":
        aliases += ["valid", "validation"]

    candidates = []
    for sp in aliases:
        candidates.extend([
            dataset_dir / f"{prefix}.{domain}.{sp}.inter",
            dataset_dir / f"{domain}.{sp}.inter",
        ])

    for p in candidates:
        if p.exists():
            return p

    # Conservative fallback: exact domain + split in basename, but avoid regenerated
    # filenames such as BEST.2025.dom1.train.inter unless user explicitly sets prefix
    globs = []
    for sp in aliases:
        globs.extend(dataset_dir.glob(f"*{domain}*{sp}.inter"))

    globs = sorted(set(globs))
    if len(globs) == 1:
        return globs[0]
    return None


def detect_schema(path: Path) -> Schema:
    with path.open("r", encoding="utf-8", newline="") as f:
        header_line = f.readline().rstrip("\n\r")
    if not header_line:
        raise ValueError(f"Empty file: {path}")

    raw_cols = header_line.split("\t")
    cols = [strip_type(c) for c in raw_cols]

    user = first_present(cols, USER_CANDIDATES)
    if user is None:
        raise ValueError(
            f"Cannot infer user field in {path}. Columns={cols}"
        )

    return Schema(
        user=user,
        item_seq=first_present(cols, ITEM_SEQ_CANDIDATES),
        target_item=first_present(cols, TARGET_ITEM_CANDIDATES),
        time_seq=first_present(cols, TIME_SEQ_CANDIDATES),
        target_time=first_present(cols, TARGET_TIME_CANDIDATES),
        seq_len_field=first_present(cols, LENGTH_CANDIDATES),
        columns=cols,
    )


def iter_inter_rows(path: Path, schema: Schema, domain: str, split: str) -> Iterable[RowRecord]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        raw_header = next(reader)
        header = [strip_type(c) for c in raw_header]

        for line_no, values in enumerate(reader, start=2):
            if not values:
                continue
            if len(values) != len(header):
                # tolerate only rows that can be padded safely
                if len(values) < len(header):
                    values = values + [""] * (len(header) - len(values))
                else:
                    values = values[:len(header)]
            row = dict(zip(header, values))

            uid = str(row.get(schema.user, "")).strip()
            if not uid:
                continue

            item_seq = parse_token_seq(row.get(schema.item_seq)) if schema.item_seq else []

            # If explicit length exists and is parseable, keep the actual parsed sequence
            # as primary because it reflects the data fed to sequential models.
            seq_len = len(item_seq)
            if seq_len == 0 and schema.seq_len_field:
                maybe_len = safe_float(row.get(schema.seq_len_field))
                if maybe_len is not None and maybe_len >= 0:
                    seq_len = int(maybe_len)

            target_item = None
            if schema.target_item:
                s = str(row.get(schema.target_item, "")).strip()
                target_item = s if s else None

            seq_times = parse_float_seq(row.get(schema.time_seq)) if schema.time_seq else []
            target_time = safe_float(row.get(schema.target_time)) if schema.target_time else None

            has_time_seq = bool(schema.time_seq and str(row.get(schema.time_seq, "")).strip())
            match = None
            monotonic = None
            first_time = None
            last_time = None

            if has_time_seq:
                match = (len(seq_times) == len(item_seq))
                if seq_times:
                    arr = np.asarray(seq_times, dtype=float)
                    monotonic = bool(np.all(arr[1:] >= arr[:-1])) if arr.size >= 2 else True
                    first_time = float(arr[0])
                    last_time = float(arr[-1])

            yield RowRecord(
                user_id=uid,
                domain=domain,
                split=split,
                seq_len=seq_len,
                target_item=target_item,
                target_time=target_time,
                seq_times=seq_times,
                seq_times_raw_len=len(seq_times),
                seq_items_raw_len=len(item_seq),
                has_time_seq=has_time_seq,
                time_item_len_match=match,
                time_monotonic=monotonic,
                first_time=first_time,
                last_time=last_time,
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_all(
    dataset_dir: Path,
    prefix: str,
    domains: Sequence[str],
    splits: Sequence[str],
) -> Tuple[
    Dict[Tuple[str, str], List[RowRecord]],
    Dict[Tuple[str, str], Schema],
    Dict[Tuple[str, str], str],
]:
    records = {}
    schemas = {}
    paths = {}

    for domain in domains:
        for split in splits:
            path = resolve_inter_file(dataset_dir, prefix, domain, split)
            if path is None:
                print(f"[WARN] Missing file for domain={domain} split={split}")
                continue

            schema = detect_schema(path)
            rows = list(iter_inter_rows(path, schema, domain, split))
            records[(domain, split)] = rows
            schemas[(domain, split)] = schema
            paths[(domain, split)] = str(path)

            print(
                f"[LOAD] {domain:>5s} {split:>5s}: "
                f"{len(rows):>7d} rows | {path.name}"
            )

    return records, schemas, paths


def domain_stats(
    records: Dict[Tuple[str, str], List[RowRecord]],
    domains: Sequence[str],
    splits: Sequence[str],
    display_names: Dict[str, str],
) -> pd.DataFrame:
    rows = []

    for domain in domains:
        for split in splits:
            recs = records.get((domain, split), [])
            if not recs:
                continue

            users = {r.user_id for r in recs}
            seq_lens = [r.seq_len for r in recs]

            time_rows = [r for r in recs if r.has_time_seq]
            time_match = [r for r in time_rows if r.time_item_len_match is True]
            mono = [r for r in time_rows if r.time_monotonic is True]
            tgt_time_rows = [r for r in recs if r.target_time is not None]

            rows.append({
                "domain": domain,
                "domain_name": display_names[domain],
                "split": split,
                "rows": len(recs),
                "users": len(users),
                "duplicate_rows_per_user": len(recs) - len(users),
                "seq_len_mean": float(np.mean(seq_lens)) if seq_lens else None,
                "seq_len_median": float(np.median(seq_lens)) if seq_lens else None,
                "seq_len_p25": q(seq_lens, 0.25),
                "seq_len_p75": q(seq_lens, 0.75),
                "seq_len_p90": q(seq_lens, 0.90),
                "time_seq_coverage": ratio(len(time_rows), len(recs)),
                "target_time_coverage": ratio(len(tgt_time_rows), len(recs)),
                "time_item_len_match_rate": ratio(len(time_match), len(time_rows)),
                "time_monotonic_rate": ratio(len(mono), len(time_rows)),
            })

    return pd.DataFrame(rows)


def user_sets(
    records: Dict[Tuple[str, str], List[RowRecord]],
    domains: Sequence[str],
    split: str,
) -> Dict[str, set]:
    out = {}
    for d in domains:
        out[d] = {r.user_id for r in records.get((d, split), [])}
    return out


def pairwise_overlap_df(
    domain_users: Dict[str, set],
    domains: Sequence[str],
    display_names: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for a, b in combinations(domains, 2):
        ua, ub = domain_users[a], domain_users[b]
        inter = ua & ub
        union = ua | ub
        rows.append({
            "domain_a": a,
            "domain_a_name": display_names[a],
            "domain_b": b,
            "domain_b_name": display_names[b],
            "users_a": len(ua),
            "users_b": len(ub),
            "intersection": len(inter),
            "union": len(union),
            "jaccard": ratio(len(inter), len(union)),
            "coverage_a_by_b": ratio(len(inter), len(ua)),
            "coverage_b_by_a": ratio(len(inter), len(ub)),
        })
    return pd.DataFrame(rows)


def user_domain_matrix_df(
    records: Dict[Tuple[str, str], List[RowRecord]],
    domains: Sequence[str],
    split: str,
    display_names: Dict[str, str],
) -> pd.DataFrame:
    by_domain_user: Dict[str, Dict[str, RowRecord]] = {}

    for d in domains:
        dmap = {}
        for r in records.get((d, split), []):
            # If duplicates exist, retain the longest sequence for audit purposes.
            prev = dmap.get(r.user_id)
            if prev is None or r.seq_len > prev.seq_len:
                dmap[r.user_id] = r
        by_domain_user[d] = dmap

    all_users = sorted(set().union(*(set(x.keys()) for x in by_domain_user.values())))
    rows = []

    for uid in all_users:
        row = {"user_id": uid}
        n_domains = 0

        for d in domains:
            name = display_names[d]
            rec = by_domain_user[d].get(uid)
            present = rec is not None
            n_domains += int(present)

            row[f"{d}_present"] = int(present)
            row[f"{d}_seq_len"] = rec.seq_len if rec else 0
            row[f"{d}_target_time"] = rec.target_time if rec else None
            row[f"{d}_last_history_time"] = rec.last_time if rec else None

        row["n_domains_present"] = n_domains
        rows.append(row)

    return pd.DataFrame(rows)


def exact_signature_df(
    user_matrix: pd.DataFrame,
    domains: Sequence[str],
    display_names: Dict[str, str],
) -> pd.DataFrame:
    if user_matrix.empty:
        return pd.DataFrame()

    counts = Counter()
    for _, r in user_matrix.iterrows():
        present = tuple(d for d in domains if int(r[f"{d}_present"]) == 1)
        counts[present] += 1

    rows = []
    total = len(user_matrix)
    for signature, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        rows.append({
            "signature": "+".join(signature) if signature else "NONE",
            "signature_names": "+".join(display_names[d] for d in signature) if signature else "NONE",
            "n_domains": len(signature),
            "users": n,
            "fraction_all_users": n / total if total else None,
        })
    return pd.DataFrame(rows)


def target_source_availability_df(
    user_matrix: pd.DataFrame,
    domains: Sequence[str],
    display_names: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    combo_rows = []

    if user_matrix.empty:
        return pd.DataFrame(), pd.DataFrame()

    for target in domains:
        target_users = user_matrix[user_matrix[f"{target}_present"] == 1].copy()
        sources = [d for d in domains if d != target]
        n_target = len(target_users)

        if n_target == 0:
            continue

        source_counts = []
        combo_counter = Counter()

        for _, row in target_users.iterrows():
            available = tuple(d for d in sources if int(row[f"{d}_present"]) == 1)
            source_counts.append(len(available))
            combo_counter[available] += 1

        source_counts = np.asarray(source_counts, dtype=int)

        summary = {
            "target_domain": target,
            "target_name": display_names[target],
            "target_users": n_target,
            "users_with_0_sources": int(np.sum(source_counts == 0)),
            "users_with_1_source": int(np.sum(source_counts == 1)),
            "users_with_2_sources": int(np.sum(source_counts == 2)),
            "users_with_3_sources": int(np.sum(source_counts == 3)),
            "frac_with_ge1_source": float(np.mean(source_counts >= 1)),
            "frac_with_ge2_sources": float(np.mean(source_counts >= 2)),
            "frac_with_3_sources": float(np.mean(source_counts == 3)),
            "mean_sources_available": float(source_counts.mean()),
        }
        summary_rows.append(summary)

        for combo, n in sorted(combo_counter.items(), key=lambda x: (-x[1], x[0])):
            combo_rows.append({
                "target_domain": target,
                "target_name": display_names[target],
                "source_combo": "+".join(combo) if combo else "NONE",
                "source_combo_names": "+".join(display_names[d] for d in combo) if combo else "NONE",
                "n_sources": len(combo),
                "users": n,
                "fraction_of_target_users": n / n_target,
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(combo_rows)


def build_domain_user_map(
    records: Dict[Tuple[str, str], List[RowRecord]],
    domains: Sequence[str],
    split: str,
) -> Dict[str, Dict[str, RowRecord]]:
    out = {}
    for d in domains:
        dmap = {}
        for r in records.get((d, split), []):
            prev = dmap.get(r.user_id)
            if prev is None or r.seq_len > prev.seq_len:
                dmap[r.user_id] = r
        out[d] = dmap
    return out


def temporal_alignment_df(
    records: Dict[Tuple[str, str], List[RowRecord]],
    domains: Sequence[str],
    split: str,
    display_names: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pair-level source->target temporal audit.

    A source domain is "usable before target" for a target sample if:
    - target sample has target_time
    - source sample has a non-empty timestamp sequence
    - at least one source history timestamp <= target target_time

    We also report future-event contamination:
    fraction of source history timestamps > target_time.
    """
    dmap = build_domain_user_map(records, domains, split)
    summary_rows = []
    detail_rows = []

    for target in domains:
        sources = [d for d in domains if d != target]
        target_map = dmap[target]

        for source in sources:
            source_map = dmap[source]
            common = sorted(set(target_map) & set(source_map))

            n_common = len(common)
            n_target_time = 0
            n_source_time_seq = 0
            n_both_time = 0
            n_has_past = 0
            n_only_future = 0
            n_has_future_contam = 0

            usable_past_counts = []
            future_fracs = []
            recencies = []

            for uid in common:
                t = target_map[uid]
                s = source_map[uid]

                if t.target_time is not None:
                    n_target_time += 1
                if s.seq_times:
                    n_source_time_seq += 1

                if t.target_time is None or not s.seq_times:
                    continue

                n_both_time += 1
                tgt = float(t.target_time)
                times = np.asarray(s.seq_times, dtype=float)
                past = times[times <= tgt]
                future = times[times > tgt]

                if past.size > 0:
                    n_has_past += 1
                    usable_past_counts.append(int(past.size))
                    recencies.append(float(tgt - past.max()))
                else:
                    n_only_future += 1

                if future.size > 0:
                    n_has_future_contam += 1

                future_fracs.append(float(future.size / times.size))

                detail_rows.append({
                    "user_id": uid,
                    "target_domain": target,
                    "target_name": display_names[target],
                    "source_domain": source,
                    "source_name": display_names[source],
                    "target_time": tgt,
                    "source_history_len": len(times),
                    "source_past_events": int(past.size),
                    "source_future_events": int(future.size),
                    "source_future_fraction": float(future.size / times.size),
                    "has_usable_past": int(past.size > 0),
                    "only_future": int(past.size == 0 and future.size > 0),
                    "source_recency_to_target": float(tgt - past.max()) if past.size > 0 else None,
                })

            summary_rows.append({
                "target_domain": target,
                "target_name": display_names[target],
                "source_domain": source,
                "source_name": display_names[source],
                "common_users": n_common,
                "target_time_coverage_on_common": ratio(n_target_time, n_common),
                "source_time_seq_coverage_on_common": ratio(n_source_time_seq, n_common),
                "both_time_available": n_both_time,
                "both_time_coverage": ratio(n_both_time, n_common),
                "has_usable_past": n_has_past,
                "usable_past_rate_given_both_time": ratio(n_has_past, n_both_time),
                "only_future_rate_given_both_time": ratio(n_only_future, n_both_time),
                "future_contamination_rate_given_both_time": ratio(n_has_future_contam, n_both_time),
                "future_event_fraction_mean": mean_or_none(future_fracs),
                "usable_past_events_median": q(usable_past_counts, 0.50),
                "source_recency_median": q(recencies, 0.50),
                "source_recency_p90": q(recencies, 0.90),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def temporal_router_target_readiness_df(
    temporal_detail: pd.DataFrame,
    user_matrix: pd.DataFrame,
    domains: Sequence[str],
    display_names: Dict[str, str],
) -> pd.DataFrame:
    """
    For each target user, count how many source domains have at least one event
    at or before the target timestamp.
    """
    rows = []

    if user_matrix.empty:
        return pd.DataFrame()

    for target in domains:
        target_users = set(
            user_matrix.loc[user_matrix[f"{target}_present"] == 1, "user_id"].astype(str)
        )
        n_target = len(target_users)
        if n_target == 0:
            continue

        if temporal_detail.empty:
            rows.append({
                "target_domain": target,
                "target_name": display_names[target],
                "target_users": n_target,
                "users_with_temporal_evidence": 0,
                "temporal_evidence_coverage": 0.0,
                "frac_with_ge1_past_source": None,
                "frac_with_ge2_past_sources": None,
                "frac_with_3_past_sources": None,
                "mean_past_sources_available": None,
            })
            continue

        df = temporal_detail[temporal_detail["target_domain"] == target].copy()
        if df.empty:
            continue

        count_by_user = defaultdict(int)
        evidence_users = set()

        for _, r in df.iterrows():
            uid = str(r["user_id"])
            evidence_users.add(uid)
            if int(r["has_usable_past"]) == 1:
                count_by_user[uid] += 1

        # Only users with actual pairwise temporal evidence are included in conditional fractions.
        counts = np.asarray([count_by_user[u] for u in evidence_users], dtype=int)

        rows.append({
            "target_domain": target,
            "target_name": display_names[target],
            "target_users": n_target,
            "users_with_temporal_evidence": len(evidence_users),
            "temporal_evidence_coverage": ratio(len(evidence_users), n_target),
            "frac_with_ge1_past_source": float(np.mean(counts >= 1)) if counts.size else None,
            "frac_with_ge2_past_sources": float(np.mean(counts >= 2)) if counts.size else None,
            "frac_with_3_past_sources": float(np.mean(counts >= 3)) if counts.size else None,
            "mean_past_sources_available": float(counts.mean()) if counts.size else None,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Readiness gates
# ---------------------------------------------------------------------------

def evaluate_gates(
    stats_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    temporal_ready_df: pd.DataFrame,
    primary_split: str,
) -> Dict[str, Any]:
    gates: Dict[str, Any] = {}

    # Gate 1: multi-source overlap
    if availability_df.empty:
        gates["multi_source_overlap"] = {
            "status": "FAIL",
            "reason": "No target/source availability statistics available."
        }
    else:
        ge2 = availability_df["frac_with_ge2_sources"].dropna().tolist()
        min_ge2 = min(ge2) if ge2 else 0.0
        avg_ge2 = float(np.mean(ge2)) if ge2 else 0.0

        status = "PASS" if avg_ge2 >= 0.20 else ("WARN" if avg_ge2 >= 0.05 else "FAIL")
        gates["multi_source_overlap"] = {
            "status": status,
            "avg_fraction_target_users_with_ge2_sources": avg_ge2,
            "min_fraction_target_users_with_ge2_sources": min_ge2,
            "heuristic": "PASS >=20%, WARN 5-20%, FAIL <5% (average across targets)",
        }

    # Gate 2: per-domain history
    train_stats = stats_df[stats_df["split"] == primary_split] if not stats_df.empty else pd.DataFrame()
    if train_stats.empty:
        gates["history_length"] = {
            "status": "FAIL",
            "reason": f"No stats for primary split={primary_split}"
        }
    else:
        medians = train_stats["seq_len_median"].dropna().tolist()
        avg_median = float(np.mean(medians)) if medians else 0.0
        status = "PASS" if avg_median >= 3 else ("WARN" if avg_median >= 2 else "FAIL")
        gates["history_length"] = {
            "status": status,
            "average_domain_median_history_length": avg_median,
            "heuristic": "PASS >=3, WARN 2-3, FAIL <2",
        }

    # Gate 3: timestamp availability
    if train_stats.empty:
        gates["timestamp_readiness"] = {
            "status": "FAIL",
            "reason": "No primary split stats."
        }
    else:
        cov = train_stats["time_seq_coverage"].dropna().tolist()
        tgt_cov = train_stats["target_time_coverage"].dropna().tolist()
        mono = train_stats["time_monotonic_rate"].dropna().tolist()
        avg_cov = float(np.mean(cov)) if cov else 0.0
        avg_tgt = float(np.mean(tgt_cov)) if tgt_cov else 0.0
        avg_mono = float(np.mean(mono)) if mono else 0.0

        if avg_cov >= 0.95 and avg_tgt >= 0.95 and avg_mono >= 0.95:
            status = "PASS"
        elif avg_cov > 0 or avg_tgt > 0:
            status = "WARN"
        else:
            status = "FAIL"

        gates["timestamp_readiness"] = {
            "status": status,
            "avg_time_seq_coverage": avg_cov,
            "avg_target_time_coverage": avg_tgt,
            "avg_time_monotonic_rate": avg_mono,
            "heuristic": "PASS requires >=95% coverage and monotonicity.",
        }

    # Gate 4: causal temporal source availability
    if temporal_ready_df.empty:
        gates["temporal_source_availability"] = {
            "status": "FAIL",
            "reason": "No pairwise temporal evidence could be computed."
        }
    else:
        cover = temporal_ready_df["temporal_evidence_coverage"].dropna().tolist()
        ge1 = temporal_ready_df["frac_with_ge1_past_source"].dropna().tolist()
        avg_cover = float(np.mean(cover)) if cover else 0.0
        avg_ge1 = float(np.mean(ge1)) if ge1 else 0.0

        if avg_cover >= 0.50 and avg_ge1 >= 0.70:
            status = "PASS"
        elif avg_cover >= 0.20 and avg_ge1 >= 0.50:
            status = "WARN"
        else:
            status = "FAIL"

        gates["temporal_source_availability"] = {
            "status": status,
            "avg_temporal_evidence_coverage": avg_cover,
            "avg_fraction_with_ge1_past_source_given_evidence": avg_ge1,
            "heuristic": "PASS: evidence coverage >=50% and >=70% have >=1 past source.",
        }

    return gates


# ---------------------------------------------------------------------------
# CLI / report
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit whether four-domain Amazon data is suitable for multi-source temporal CDSR."
    )
    p.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset/BEST",
        help="Directory containing *.inter files."
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="BEST",
        help="Filename prefix, e.g. BEST -> BEST.dom1.train.inter"
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=["dom1", "dom2", "dom3", "dom4"],
        help="Domain IDs."
    )
    p.add_argument(
        "--domain-names",
        nargs="+",
        default=None,
        help="Optional readable names aligned with --domains, e.g. Books Electronics Sports Tools"
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to audit."
    )
    p.add_argument(
        "--primary-split",
        type=str,
        default="train",
        help="Split used for overlap/router feasibility."
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="save/multisource_audit",
        help="Output directory."
    )
    p.add_argument(
        "--no-user-matrix",
        action="store_true",
        help="Do not save the potentially large user_domain_matrix.csv."
    )
    p.add_argument(
        "--no-temporal-detail",
        action="store_true",
        help="Do not save per-user pairwise temporal_alignment_detail.csv."
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    root = Path.cwd().resolve()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (root / dataset_dir).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = list(args.domains)
    if args.domain_names is not None:
        if len(args.domain_names) != len(domains):
            raise SystemExit("--domain-names must have the same length as --domains")
        display_names = dict(zip(domains, args.domain_names))
    else:
        display_names = {d: d for d in domains}

    print("=" * 88)
    print("Multi-source CDSR data audit")
    print("=" * 88)
    print(f"Project root   : {root}")
    print(f"Dataset dir    : {dataset_dir}")
    print(f"Prefix         : {args.prefix}")
    print(f"Domains        : {domains}")
    print(f"Domain names   : {[display_names[d] for d in domains]}")
    print(f"Splits         : {args.splits}")
    print(f"Primary split  : {args.primary_split}")
    print(f"Output dir     : {output_dir}")
    print()

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory does not exist: {dataset_dir}")
        return 2

    records, schemas, paths = load_all(
        dataset_dir=dataset_dir,
        prefix=args.prefix,
        domains=domains,
        splits=args.splits,
    )

    if not records:
        print("[ERROR] No .inter files could be loaded.")
        return 2

    # Basic stats
    stats_df = domain_stats(records, domains, args.splits, display_names)
    stats_path = output_dir / "domain_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    # Primary split overlap
    dusers = user_sets(records, domains, args.primary_split)
    overlap_df = pairwise_overlap_df(dusers, domains, display_names)
    overlap_path = output_dir / "pairwise_user_overlap.csv"
    overlap_df.to_csv(overlap_path, index=False)

    user_matrix = user_domain_matrix_df(
        records, domains, args.primary_split, display_names
    )
    if not args.no_user_matrix:
        user_matrix.to_csv(output_dir / "user_domain_matrix.csv", index=False)

    signature_df = exact_signature_df(user_matrix, domains, display_names)
    signature_df.to_csv(output_dir / "user_domain_signatures.csv", index=False)

    availability_df, combo_df = target_source_availability_df(
        user_matrix, domains, display_names
    )
    availability_df.to_csv(output_dir / "target_source_availability.csv", index=False)
    combo_df.to_csv(output_dir / "target_source_combinations.csv", index=False)

    # Temporal audit
    temporal_summary_df, temporal_detail_df = temporal_alignment_df(
        records, domains, args.primary_split, display_names
    )
    temporal_summary_df.to_csv(output_dir / "temporal_alignment_summary.csv", index=False)
    if not args.no_temporal_detail:
        temporal_detail_df.to_csv(output_dir / "temporal_alignment_detail.csv", index=False)

    temporal_ready_df = temporal_router_target_readiness_df(
        temporal_detail_df, user_matrix, domains, display_names
    )
    temporal_ready_df.to_csv(output_dir / "temporal_router_readiness.csv", index=False)

    gates = evaluate_gates(
        stats_df=stats_df,
        availability_df=availability_df,
        temporal_ready_df=temporal_ready_df,
        primary_split=args.primary_split,
    )

    # Schema report
    schema_report = {}
    for (d, sp), schema in schemas.items():
        schema_report[f"{d}.{sp}"] = {
            "path": paths[(d, sp)],
            **asdict(schema),
        }

    # Summary
    summary = {
        "project_root": str(root),
        "dataset_dir": str(dataset_dir),
        "prefix": args.prefix,
        "domains": domains,
        "display_names": display_names,
        "splits": list(args.splits),
        "primary_split": args.primary_split,
        "assumptions": {
            "user_id_global_consistency_required": True,
            "note": (
                "Cross-domain overlap assumes the same user_id denotes the same user "
                "across domains. This cannot be proven from processed .inter files alone."
            ),
        },
        "schemas": schema_report,
        "gates": gates,
        "output_files": {
            "domain_stats": str(stats_path),
            "pairwise_user_overlap": str(overlap_path),
            "user_domain_matrix": None if args.no_user_matrix else str(output_dir / "user_domain_matrix.csv"),
            "user_domain_signatures": str(output_dir / "user_domain_signatures.csv"),
            "target_source_availability": str(output_dir / "target_source_availability.csv"),
            "target_source_combinations": str(output_dir / "target_source_combinations.csv"),
            "temporal_alignment_summary": str(output_dir / "temporal_alignment_summary.csv"),
            "temporal_alignment_detail": None if args.no_temporal_detail else str(output_dir / "temporal_alignment_detail.csv"),
            "temporal_router_readiness": str(output_dir / "temporal_router_readiness.csv"),
        },
    }

    summary_path = output_dir / "audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(ensure_jsonable(summary), f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # Human-readable terminal report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("A. Domain statistics")
    print("=" * 88)
    if stats_df.empty:
        print("No statistics.")
    else:
        show = stats_df[
            [
                "domain", "domain_name", "split", "users",
                "seq_len_median", "time_seq_coverage",
                "target_time_coverage", "time_monotonic_rate"
            ]
        ].copy()
        print(show.to_string(index=False))

    print("\n" + "=" * 88)
    print(f"B. Pairwise user overlap ({args.primary_split})")
    print("=" * 88)
    if overlap_df.empty:
        print("No overlap statistics.")
    else:
        for _, r in overlap_df.iterrows():
            print(
                f"{r['domain_a_name']:>14s} <-> {r['domain_b_name']:<14s} "
                f"intersection={int(r['intersection']):>7d} | "
                f"Jaccard={r['jaccard']:.4f} | "
                f"A covered={fmt_pct(r['coverage_a_by_b'])} | "
                f"B covered={fmt_pct(r['coverage_b_by_a'])}"
            )

    print("\n" + "=" * 88)
    print("C. Exact domain signatures")
    print("=" * 88)
    if signature_df.empty:
        print("No signature statistics.")
    else:
        print(
            signature_df[
                ["signature_names", "n_domains", "users", "fraction_all_users"]
            ].head(20).to_string(index=False)
        )

    print("\n" + "=" * 88)
    print("D. Source-domain availability for each target")
    print("=" * 88)
    if availability_df.empty:
        print("No source availability statistics.")
    else:
        for _, r in availability_df.iterrows():
            print(
                f"Target={r['target_name']:<14s} users={int(r['target_users']):>7d} | "
                f">=1 source {fmt_pct(r['frac_with_ge1_source'])} | "
                f">=2 sources {fmt_pct(r['frac_with_ge2_sources'])} | "
                f"3 sources {fmt_pct(r['frac_with_3_sources'])} | "
                f"mean sources={r['mean_sources_available']:.3f}"
            )

    print("\n" + "=" * 88)
    print("E. Temporal alignment")
    print("=" * 88)
    if temporal_summary_df.empty:
        print("No temporal alignment could be computed.")
    else:
        for _, r in temporal_summary_df.iterrows():
            print(
                f"{r['source_name']:>14s} -> {r['target_name']:<14s} "
                f"common={int(r['common_users']):>7d} | "
                f"both-time={fmt_pct(r['both_time_coverage'])} | "
                f"usable-past={fmt_pct(r['usable_past_rate_given_both_time'])} | "
                f"future-contam={fmt_pct(r['future_contamination_rate_given_both_time'])}"
            )

    print("\n" + "=" * 88)
    print("F. Research-readiness gates")
    print("=" * 88)
    for name, info in gates.items():
        print(f"[{info.get('status', 'NA'):>4s}] {name}")
        for k, v in info.items():
            if k == "status":
                continue
            print(f"       {k}: {v}")

    print("\n" + "=" * 88)
    print("Interpretation")
    print("=" * 88)
    print(
        "1) If multi_source_overlap=PASS, the processed data contains enough users with "
        "multiple auxiliary domains to justify multi-source routing.\n"
        "2) If timestamp_readiness=PASS and temporal_source_availability=PASS, a true "
        "user-time-source router can be constructed without obvious future leakage.\n"
        "3) If overlap passes but timestamps fail, do NOT fake a Temporal Router from row "
        "order. Build a user-level multi-source router first, or return to raw timestamped data.\n"
        "4) The next experiment after this audit should be a Multi-source Utility Audit, "
        "not the full router."
    )

    print("\n[DONE] Summary :", summary_path)
    print("[DONE] Outputs :", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
