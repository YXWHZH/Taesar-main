#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_multisource_utility.py
============================

多源跨域序列推荐（Multi-source CDSR）边际效用审计。

目的
----
在不训练完整 Router 的前提下，验证下面四个核心假设：

Q1. 不同源域的平均边际效用是否不同？
Q2. 每个源域是否都存在用户级负迁移（utility < 0）？
Q3. 不同用户的最佳源域是否不同？
Q4. Oracle 稀疏选择（Top-1 / Top-2）是否优于全部源域 Dense Fusion？

方法
----
1. 复用 Taesar 已训练好的四个 target-only SASRec (train_type=sim) checkpoint，
   将它们作为 frozen domain experts。
2. 对每个用户抽取各域最后位置的 SASRec 隐表示 h_u^d。
3. 固定所有 expert，不更新。
4. 针对一个 target domain，只训练一个很小的 mask-conditioned fusion probe：

       h = h_target + sum_d m_d * W_d(h_source_d)

   训练时随机采样 source mask，因此同一个 probe 学会处理：
       000 (target only)
       100 / 010 / 001
       110 / 101 / 011
       111 (all sources)

5. 在固定 probe 参数下，对每个用户计算：

       U_{u,d} = Loss(u; 000) - Loss(u; only source d)

   U > 0：加入该源域使目标预测 loss 降低（有益）
   U < 0：加入该源域使目标预测 loss 升高（负迁移）

重要限制
--------
当前 Taesar 处理后的 BEST *.inter 没有 timestamp。
因此这个脚本只审计：
    user × source-domain
层面的个性化效用。

它不能证明：
    user × time × source-domain
层面的时序效用，也不能用于宣称 Temporal Router。

数据使用原则
------------
- Probe 训练：target-domain train 样本。
- Probe 验证：target-domain valid 样本。
- 最终审计：target-domain test 样本（默认）。
- Source expert representation 始终来自各 source domain 的 TRAIN 历史，
  不读取 source valid/test target，尽量减少额外标签泄漏。
- 由于处理后的四域数据没有跨域 timestamp，仍无法证明 source train 历史
  在真实时间上严格早于 target valid/test 时刻；这属于数据本身限制。

默认项目结构
------------
Taesar-main/
  config/overall.yaml
  model/seq2seq_sasrec.py
  dataset/BEST/
    BEST.dom1.train.inter
    BEST.dom1.valid.inter
    BEST.dom1.test.inter
    ...
  checkpoint/
    SASRec-BEST-2025.tune.ckpt-dom1-sim
    ...
    SASRec-BEST-2025.tune.ckpt-dom4-sim

推荐首次运行
------------
conda activate Taesar
cd /root/autodl-tmp/Taesar-main

python audit_multisource_utility.py \
  --target-dom dom1 \
  --seed 2025 \
  --gpu-id 0 \
  --epochs 10 \
  --max-train-users 20000 \
  --max-valid-users 5000 \
  --audit-split test

先只检查路径、checkpoint、用户对齐：
python audit_multisource_utility.py --target-dom dom1 --dry-run

全量训练：
python audit_multisource_utility.py \
  --target-dom dom1 \
  --max-train-users 0 \
  --max-valid-users 0 \
  --epochs 20

输出
----
save/multisource_utility/<target_dom>/
  config.json
  alignment_summary.json
  probe_best.pt
  training_history.json
  utility_<split>.csv
  utility_summary_<split>.json
  cache/
    <domain>_<split>_embeddings.npz

依赖
----
Taesar 环境本身 + numpy + torch + hydra + omegaconf + recbole
不需要 scipy / sklearn。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Basic utilities
# =============================================================================

DOMAINS = ("dom1", "dom2", "dom3", "dom4")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, ensure_ascii=False, indent=2)


def resolve_device(config: Any, requested: str) -> torch.device:
    if requested == "auto":
        device = config["device"]
        if not isinstance(device, torch.device):
            device = torch.device(str(device))
    else:
        requested = "cuda:0" if requested == "cuda" else requested
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")

    config["device"] = device
    config["use_gpu"] = device.type == "cuda"
    return device


def find_project_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "config" / "overall.yaml").exists():
            raise FileNotFoundError(f"No config/overall.yaml under --project-root: {root}")
        return root

    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen = set()
    for start in starts:
        for p in [start, *start.parents]:
            p = p.resolve()
            if p in seen:
                continue
            seen.add(p)
            if (
                (p / "config" / "overall.yaml").exists()
                and (p / "model" / "seq2seq_sasrec.py").exists()
            ):
                return p

    raise FileNotFoundError(
        "Cannot locate Taesar project root. Put this script in Taesar-main/ "
        "or pass --project-root /root/autodl-tmp/Taesar-main"
    )


def ensure_project_importable(root: Path) -> None:
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    os.chdir(root)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen-domain-expert multi-source utility audit for Taesar BEST data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--dataset-dir", type=str, default=None)
    p.add_argument("--dataset-name", type=str, default="BEST")

    p.add_argument("--target-dom", type=str, default="dom1", choices=DOMAINS)
    p.add_argument("--domains", nargs="+", default=list(DOMAINS))
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument(
        "--checkpoint-template",
        type=str,
        default=None,
        help=(
            "Optional template, e.g. "
            "'checkpoint/SASRec-BEST-2025.tune.ckpt-{domain}-sim'. "
            "If omitted, uses config['tune_ckpt']-<domain>-sim."
        ),
    )
    p.add_argument("--non-strict", action="store_true")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--encode-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--masks-per-sample", type=int, default=2)
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Logit temperature for probe classification.",
    )

    p.add_argument(
        "--max-train-users",
        type=int,
        default=20000,
        help="0 = all aligned train users.",
    )
    p.add_argument(
        "--max-valid-users",
        type=int,
        default=5000,
        help="0 = all aligned valid users.",
    )
    p.add_argument(
        "--max-audit-users",
        type=int,
        default=0,
        help="0 = all aligned users in audit split.",
    )
    p.add_argument(
        "--audit-split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
    )
    p.add_argument(
        "--allow-missing-sources",
        action="store_true",
        help=(
            "By default only users with all 3 source-train representations are used, "
            "for a clean 8-mask comparison. Set this flag to retain users with missing sources."
        ),
    )

    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Embedding cache directory. Default: <output-dir>/cache",
    )
    p.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse cached expert embeddings when cache files exist.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Default: save/multisource_utility/<target_dom>",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--progress-every", type=int, default=20)

    return p.parse_args()


# =============================================================================
# Taesar / RecBole config and expert loading
# =============================================================================

def build_recbole_config(
    project_root: Path,
    target_dom: str,
    seed: int,
    gpu_id: int,
) -> Any:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from recbole.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "Missing hydra/omegaconf/recbole. Activate the Taesar conda environment first."
        ) from exc

    config_dir = project_root / "config"
    overrides = [
        f"target_dom={target_dom}",
        f"seed={seed}",
        "stage=tun",
        "train_type=sim",
        f"gpu_id={gpu_id}",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        hydra_cfg = compose(config_name="overall", overrides=overrides)

    cfg_copy = OmegaConf.create(OmegaConf.to_container(hydra_cfg, resolve=False))
    for removable in ("base", "hydra"):
        if removable in cfg_copy:
            del cfg_copy[removable]

    cfg_dict = OmegaConf.to_container(cfg_copy, resolve=True)
    config = Config(
        model=cfg_dict["model_name"],
        dataset=cfg_dict["dataset"],
        config_dict=cfg_dict,
    )
    config["benchmark_filename"] = [
        f"{target_dom}.train",
        f"{target_dom}.valid",
        f"{target_dom}.test",
    ]
    return config


def create_domain_dataset(config: Any) -> Any:
    from recbole.data.dataset import SequentialDataset
    return SequentialDataset(config)


def create_domain_model(
    config: Any,
    dataset: Any,
    device: torch.device,
) -> torch.nn.Module:
    from model.seq2seq_sasrec import SASRec

    model = SASRec(config, dataset).to(device)
    model.flag = "sim"
    return model


def resolve_checkpoint(
    project_root: Path,
    config: Any,
    domain: str,
    template: Optional[str],
) -> Path:
    candidates: List[Path] = []

    if template:
        rendered = template.format(domain=domain)
        p = Path(rendered).expanduser()
        if not p.is_absolute():
            p = project_root / p
        candidates.append(p.resolve())

    expected = Path(f"{config['tune_ckpt']}-{domain}-sim")
    if not expected.is_absolute():
        expected = project_root / expected
    candidates.append(expected.resolve())

    candidates.append(
        (project_root / "checkpoint" / f"SASRec-BEST-2025.tune.ckpt-{domain}-sim").resolve()
    )

    for p in candidates:
        if p.exists():
            return p

    found: List[Path] = []
    pattern = f"*tune.ckpt-{domain}-sim"
    for base in (project_root / "checkpoint", project_root / "save"):
        if base.exists():
            found.extend(x.resolve() for x in base.rglob(pattern) if x.is_file())

    if found:
        found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return found[0]

    raise FileNotFoundError(
        f"Cannot find sim checkpoint for {domain}. Tried:\n"
        + "\n".join(f"  {x}" for x in candidates)
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    strict: bool,
) -> Dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, Mapping) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, Mapping):
        state_dict = ckpt
    else:
        raise TypeError(f"Unknown checkpoint type: {type(ckpt)}")

    # tolerate DDP module. prefix
    if state_dict and all(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k)[7:]: v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(
            f"[LOAD non-strict] missing={len(result.missing_keys)} "
            f"unexpected={len(result.unexpected_keys)}"
        )
    return dict(ckpt) if isinstance(ckpt, Mapping) else {}


# =============================================================================
# .inter parsing
# =============================================================================

@dataclass(frozen=True)
class InterRow:
    user: str
    seq: Tuple[str, ...]
    target: str
    line_no: int


def base_field_name(header: str) -> str:
    return header.split(":", 1)[0].strip()


def read_inter(path: Path) -> List[InterRow]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[InterRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"No header: {path}")

        base = {base_field_name(h): h for h in reader.fieldnames}
        required = ["user_id", "item_id_list", "item_id"]
        missing = [x for x in required if x not in base]
        if missing:
            raise ValueError(
                f"{path} missing fields {missing}. Header={reader.fieldnames}"
            )

        uc, sc, tc = base["user_id"], base["item_id_list"], base["item_id"]

        for line_no, r in enumerate(reader, start=2):
            u = (r.get(uc) or "").strip()
            s = (r.get(sc) or "").strip()
            t = (r.get(tc) or "").strip()
            if not u or not s or not t:
                continue
            seq = tuple(x for x in s.split() if x)
            if seq:
                rows.append(InterRow(u, seq, t, line_no))

    return rows


def index_unique_by_user(rows: Sequence[InterRow]) -> Tuple[Dict[str, InterRow], int]:
    grouped: Dict[str, List[InterRow]] = {}
    for r in rows:
        grouped.setdefault(r.user, []).append(r)

    out: Dict[str, InterRow] = {}
    ambiguous = 0
    for u, rs in grouped.items():
        if len(rs) == 1:
            out[u] = rs[0]
        else:
            ambiguous += 1
            # deterministic: keep longest history, then earliest line
            out[u] = sorted(rs, key=lambda x: (-len(x.seq), x.line_no))[0]
    return out, ambiguous


def resolve_inter_path(
    dataset_dir: Path,
    dataset_name: str,
    domain: str,
    split: str,
) -> Path:
    aliases = [split]
    if split == "valid":
        aliases += ["val", "validation"]

    candidates: List[Path] = []
    for sp in aliases:
        candidates += [
            dataset_dir / f"{dataset_name}.{domain}.{sp}.inter",
            dataset_dir / f"{domain}.{sp}.inter",
        ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        f"Cannot locate {domain}/{split}. Tried:\n"
        + "\n".join(f"  {x}" for x in candidates)
    )


# =============================================================================
# Token mapping and frozen expert encoding
# =============================================================================

def get_token_mapping(dataset: Any, field: str) -> Mapping[Any, Any]:
    m = getattr(dataset, "field2token_id", {}).get(field)
    if m is None:
        raise KeyError(f"Dataset has no token mapping for field {field!r}")
    return m


def token_to_inner_id(mapping: Mapping[Any, Any], token: str) -> int:
    candidates: List[Any] = [token]
    try:
        candidates.append(int(token))
    except Exception:
        pass

    for c in candidates:
        if c in mapping:
            return int(mapping[c])
    raise KeyError(token)


@dataclass
class EncodedSplit:
    users: np.ndarray            # dtype str
    embeddings: np.ndarray       # [N, H], float32
    target_inner_ids: np.ndarray # [N], int64; -1 when not requested / invalid
    seq_lens: np.ndarray         # [N], int32

    def to_user_dict(self) -> Dict[str, np.ndarray]:
        return {str(u): self.embeddings[i] for i, u in enumerate(self.users)}


def save_encoded(path: Path, data: EncodedSplit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        users=data.users,
        embeddings=data.embeddings.astype(np.float32),
        target_inner_ids=data.target_inner_ids.astype(np.int64),
        seq_lens=data.seq_lens.astype(np.int32),
    )


def load_encoded(path: Path) -> EncodedSplit:
    z = np.load(path, allow_pickle=False)
    return EncodedSplit(
        users=z["users"],
        embeddings=z["embeddings"].astype(np.float32),
        target_inner_ids=z["target_inner_ids"].astype(np.int64),
        seq_lens=z["seq_lens"].astype(np.int32),
    )


@torch.no_grad()
def encode_rows(
    model: torch.nn.Module,
    dataset: Any,
    rows: Sequence[InterRow],
    device: torch.device,
    batch_size: int,
    need_target_ids: bool,
    progress_prefix: str,
) -> EncodedSplit:
    model.eval()

    item_map = get_token_mapping(dataset, model.ITEM_ID)
    max_len = int(model.max_seq_length)

    valid_users: List[str] = []
    valid_seq_ids: List[List[int]] = []
    valid_target_ids: List[int] = []
    valid_lens: List[int] = []
    skipped_token = 0

    for r in rows:
        tokens = r.seq[-max_len:]
        try:
            ids = [token_to_inner_id(item_map, x) for x in tokens]
            target_id = token_to_inner_id(item_map, r.target) if need_target_ids else -1
        except KeyError:
            skipped_token += 1
            continue

        if not ids:
            continue

        valid_users.append(r.user)
        valid_seq_ids.append(ids)
        valid_target_ids.append(target_id)
        valid_lens.append(len(ids))

    if skipped_token:
        print(f"[WARN] {progress_prefix}: skipped {skipped_token} rows due to unknown tokens")

    all_emb: List[np.ndarray] = []
    n = len(valid_users)
    started = time.time()

    for start in range(0, n, batch_size):
        batch_ids = valid_seq_ids[start:start + batch_size]
        lens = torch.tensor(
            [len(x) for x in batch_ids], dtype=torch.long, device=device
        )
        padded = torch.zeros(
            (len(batch_ids), max_len), dtype=torch.long, device=device
        )
        for i, ids in enumerate(batch_ids):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        output = model.forward(padded, lens)
        last = model.gather_indexes(output, lens - 1)
        all_emb.append(last.detach().float().cpu().numpy())

        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= n:
            done = min(start + batch_size, n)
            print(
                f"[ENCODE] {progress_prefix}: {done}/{n} "
                f"elapsed={time.time() - started:.1f}s"
            )

    if all_emb:
        embeddings = np.concatenate(all_emb, axis=0)
    else:
        embeddings = np.zeros((0, int(model.hidden_size)), dtype=np.float32)

    return EncodedSplit(
        users=np.asarray(valid_users, dtype=str),
        embeddings=embeddings,
        target_inner_ids=np.asarray(valid_target_ids, dtype=np.int64),
        seq_lens=np.asarray(valid_lens, dtype=np.int32),
    )


# =============================================================================
# Alignment
# =============================================================================

@dataclass
class AlignedData:
    users: np.ndarray       # [N]
    h_target: np.ndarray    # [N,H]
    h_sources: np.ndarray   # [N,S,H]
    available: np.ndarray   # [N,S] bool
    y_class: np.ndarray     # [N]
    target_seq_len: np.ndarray
    source_seq_lens: np.ndarray  # [N,S]


def build_source_lookup(
    encoded: EncodedSplit,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    emb = {}
    lens = {}
    for i, u in enumerate(encoded.users):
        k = str(u)
        emb[k] = encoded.embeddings[i]
        lens[k] = int(encoded.seq_lens[i])
    return emb, lens


def align_for_target(
    target_encoded: EncodedSplit,
    source_encoded_train: Dict[str, EncodedSplit],
    source_domains: Sequence[str],
    target_begin_id: int,
    target_end_id: int,
    require_all_sources: bool,
) -> Tuple[AlignedData, Dict[str, int]]:
    source_lookup = {}
    source_len_lookup = {}
    for d in source_domains:
        source_lookup[d], source_len_lookup[d] = build_source_lookup(
            source_encoded_train[d]
        )

    users = []
    ht = []
    hs = []
    avail = []
    ys = []
    tlens = []
    slens = []

    stats = {
        "target_rows": len(target_encoded.users),
        "kept": 0,
        "skip_target_outside_domain": 0,
        "skip_missing_source_when_required": 0,
        "users_with_all_sources": 0,
        "users_with_2_sources": 0,
        "users_with_1_source": 0,
        "users_with_0_sources": 0,
    }

    hidden = target_encoded.embeddings.shape[1]

    for i, u0 in enumerate(target_encoded.users):
        u = str(u0)
        target_id = int(target_encoded.target_inner_ids[i])
        if target_id < target_begin_id or target_id > target_end_id:
            stats["skip_target_outside_domain"] += 1
            continue

        src_vecs = []
        src_avail = []
        src_lens = []

        for d in source_domains:
            if u in source_lookup[d]:
                src_vecs.append(source_lookup[d][u])
                src_avail.append(True)
                src_lens.append(source_len_lookup[d][u])
            else:
                src_vecs.append(np.zeros(hidden, dtype=np.float32))
                src_avail.append(False)
                src_lens.append(0)

        n_avail = int(sum(src_avail))
        if n_avail == len(source_domains):
            stats["users_with_all_sources"] += 1
        elif n_avail == 2:
            stats["users_with_2_sources"] += 1
        elif n_avail == 1:
            stats["users_with_1_source"] += 1
        else:
            stats["users_with_0_sources"] += 1

        if require_all_sources and n_avail != len(source_domains):
            stats["skip_missing_source_when_required"] += 1
            continue

        users.append(u)
        ht.append(target_encoded.embeddings[i])
        hs.append(np.stack(src_vecs, axis=0))
        avail.append(np.asarray(src_avail, dtype=np.bool_))
        ys.append(target_id - target_begin_id)
        tlens.append(int(target_encoded.seq_lens[i]))
        slens.append(np.asarray(src_lens, dtype=np.int32))

    stats["kept"] = len(users)

    if not users:
        raise RuntimeError("No aligned users remained after alignment.")

    return (
        AlignedData(
            users=np.asarray(users, dtype=str),
            h_target=np.stack(ht).astype(np.float32),
            h_sources=np.stack(hs).astype(np.float32),
            available=np.stack(avail).astype(np.bool_),
            y_class=np.asarray(ys, dtype=np.int64),
            target_seq_len=np.asarray(tlens, dtype=np.int32),
            source_seq_lens=np.stack(slens).astype(np.int32),
        ),
        stats,
    )


def subsample_aligned(data: AlignedData, max_n: int, seed: int) -> AlignedData:
    n = len(data.users)
    if max_n <= 0 or n <= max_n:
        return data

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_n, replace=False))
    return AlignedData(
        users=data.users[idx],
        h_target=data.h_target[idx],
        h_sources=data.h_sources[idx],
        available=data.available[idx],
        y_class=data.y_class[idx],
        target_seq_len=data.target_seq_len[idx],
        source_seq_lens=data.source_seq_lens[idx],
    )


# =============================================================================
# Torch dataset / probe
# =============================================================================

class ProbeDataset(Dataset):
    def __init__(self, data: AlignedData):
        self.data = data

    def __len__(self) -> int:
        return len(self.data.users)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.data.h_target[idx]),
            torch.from_numpy(self.data.h_sources[idx]),
            torch.from_numpy(self.data.available[idx]),
            torch.tensor(self.data.y_class[idx], dtype=torch.long),
            idx,
        )


class MaskFusionProbe(nn.Module):
    """
    Lightweight probe, NOT the final Router.

    target representation:
        h_t' = h_t + Delta_t(h_t)

    source contribution:
        c_d = W_d(h_d)

    fusion:
        h = LN(h_t' + sum_d mask_d * c_d / sqrt(1 + #active_sources))

    item classifier:
        logits = h @ frozen_target_item_embeddings.T / temperature
    """

    def __init__(
        self,
        hidden_size: int,
        n_sources: int,
        target_item_embeddings: torch.Tensor,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_sources = n_sources
        self.temperature = float(temperature)

        self.target_delta = nn.Linear(hidden_size, hidden_size)
        self.source_adapters = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_sources)]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

        # Keep target-domain item classifier frozen.
        self.register_buffer(
            "target_item_embeddings",
            target_item_embeddings.detach().clone().float(),
            persistent=True,
        )

        # Start near the frozen target expert.
        nn.init.zeros_(self.target_delta.weight)
        nn.init.zeros_(self.target_delta.bias)
        for layer in self.source_adapters:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def fuse(
        self,
        h_target: torch.Tensor,       # [B,H]
        h_sources: torch.Tensor,      # [B,S,H]
        mask: torch.Tensor,           # [B,S] float/bool
        available: torch.Tensor,      # [B,S] bool
    ) -> torch.Tensor:
        mask = mask.float() * available.float()

        base = h_target + self.target_delta(h_target)
        base = self.dropout(base)

        contrib = torch.zeros_like(base)
        for d, adapter in enumerate(self.source_adapters):
            c = adapter(h_sources[:, d, :])
            c = self.dropout(c)
            contrib = contrib + mask[:, d:d+1] * c

        active = mask.sum(dim=1, keepdim=True)
        scale = torch.sqrt(1.0 + active)
        fused = self.norm(base + contrib / scale)
        return fused

    def forward(
        self,
        h_target: torch.Tensor,
        h_sources: torch.Tensor,
        mask: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fuse(h_target, h_sources, mask, available)
        logits = fused @ self.target_item_embeddings.t()
        logits = logits / max(self.temperature, 1e-8)
        return logits


def all_binary_masks(n_sources: int) -> torch.Tensor:
    rows = []
    for x in range(2 ** n_sources):
        rows.append([(x >> d) & 1 for d in range(n_sources)])
    return torch.tensor(rows, dtype=torch.float32)


def sample_training_masks(
    available: torch.Tensor,
    n_repeats: int,
) -> torch.Tensor:
    """
    Bernoulli(0.5) source masks intersected with availability.
    Returns [B*n_repeats, S].
    """
    b, s = available.shape
    avail = available.repeat_interleave(n_repeats, dim=0)
    m = torch.bernoulli(
        torch.full((b * n_repeats, s), 0.5, device=available.device)
    )
    return m * avail.float()


# =============================================================================
# Training
# =============================================================================

@torch.no_grad()
def validation_mask_average_loss(
    model: MaskFusionProbe,
    loader: DataLoader,
    device: torch.device,
    n_sources: int,
) -> float:
    model.eval()

    # Audit-relevant masks for validation:
    # target-only, each single source, dense-all.
    masks = [torch.zeros(n_sources)]
    for d in range(n_sources):
        m = torch.zeros(n_sources)
        m[d] = 1
        masks.append(m)
    masks.append(torch.ones(n_sources))

    total = 0.0
    count = 0

    for ht, hs, avail, y, _ in loader:
        ht = ht.to(device)
        hs = hs.to(device)
        avail = avail.to(device)
        y = y.to(device)

        batch_loss = 0.0
        for m0 in masks:
            m = m0.to(device).unsqueeze(0).expand(len(y), -1)
            logits = model(ht, hs, m, avail)
            batch_loss = batch_loss + F.cross_entropy(
                logits, y, reduction="mean"
            )
        batch_loss = batch_loss / len(masks)

        total += float(batch_loss.item()) * len(y)
        count += len(y)

    return total / max(count, 1)


def train_probe(
    model: MaskFusionProbe,
    train_data: AlignedData,
    valid_data: AlignedData,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    train_loader = DataLoader(
        ProbeDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        ProbeDataset(valid_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = output_dir / "probe_best.pt"

    print("\n" + "=" * 88)
    print("Training mask-conditioned fusion probe")
    print("=" * 88)
    print(f"Train users       : {len(train_data.users)}")
    print(f"Valid users       : {len(valid_data.users)}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Masks/sample      : {args.masks_per_sample}")
    print(f"Epochs            : {args.epochs}")
    print(f"Trainable params  : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running = 0.0
        seen = 0

        for ht, hs, avail, y, _ in train_loader:
            ht = ht.to(device, non_blocking=True)
            hs = hs.to(device, non_blocking=True)
            avail = avail.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            r = max(1, int(args.masks_per_sample))
            masks = sample_training_masks(avail, r)

            ht_r = ht.repeat_interleave(r, dim=0)
            hs_r = hs.repeat_interleave(r, dim=0)
            avail_r = avail.repeat_interleave(r, dim=0)
            y_r = y.repeat_interleave(r, dim=0)

            optimizer.zero_grad(set_to_none=True)
            logits = model(ht_r, hs_r, masks, avail_r)
            loss = F.cross_entropy(logits, y_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += float(loss.item()) * len(y_r)
            seen += len(y_r)

        train_loss = running / max(seen, 1)
        val_loss = validation_mask_average_loss(
            model, valid_loader, device, len(model.source_adapters)
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_mask_avg_loss": val_loss,
            "seconds": time.time() - started,
        }
        history.append(row)

        print(
            f"[EPOCH {epoch:02d}] train={train_loss:.6f} "
            f"valid(mask-avg)={val_loss:.6f} "
            f"time={row['seconds']:.1f}s"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "valid_mask_avg_loss": val_loss,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"[EARLY STOP] no validation improvement for {args.patience} epochs."
                )
                break

    print(f"[BEST] epoch={best_epoch}, valid_mask_avg_loss={best_val:.6f}")
    write_json(output_dir / "training_history.json", history)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return history


# =============================================================================
# Utility evaluation
# =============================================================================

def mask_name(mask: Sequence[int], source_domains: Sequence[str]) -> str:
    active = [source_domains[i] for i, x in enumerate(mask) if int(x) == 1]
    return "target_only" if not active else "+".join(active)


@torch.no_grad()
def audit_utility(
    model: MaskFusionProbe,
    data: AlignedData,
    source_domains: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> Tuple[Path, Path]:
    model.eval()

    loader = DataLoader(
        ProbeDataset(data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    n_sources = len(source_domains)
    masks_tensor = all_binary_masks(n_sources)  # [M,S]
    masks_np = masks_tensor.numpy().astype(int)
    mask_names = [mask_name(m, source_domains) for m in masks_np]

    # index helpers
    zero_idx = next(i for i, m in enumerate(masks_np) if m.sum() == 0)
    dense_idx = next(i for i, m in enumerate(masks_np) if m.sum() == n_sources)
    single_indices = [
        next(i for i, m in enumerate(masks_np) if m.sum() == 1 and m[d] == 1)
        for d in range(n_sources)
    ]
    pair_indices = [
        i for i, m in enumerate(masks_np) if m.sum() == min(2, n_sources)
    ]

    all_losses: List[np.ndarray] = []
    all_ranks: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []

    started = time.time()
    for batch_no, (ht, hs, avail, y, idx) in enumerate(loader, start=1):
        ht = ht.to(device, non_blocking=True)
        hs = hs.to(device, non_blocking=True)
        avail = avail.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        batch_losses = []
        batch_ranks = []

        for m0 in masks_tensor:
            m = m0.to(device).unsqueeze(0).expand(len(y), -1)
            # For missing-source mode, a requested source can be unavailable.
            # The model intersects mask with availability internally.
            logits = model(ht, hs, m, avail)
            losses = F.cross_entropy(logits, y, reduction="none")

            target_scores = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            ranks = 1 + (logits > target_scores.unsqueeze(1)).sum(dim=1)

            batch_losses.append(losses.detach().cpu().numpy())
            batch_ranks.append(ranks.detach().cpu().numpy())

        all_losses.append(np.stack(batch_losses, axis=1))  # [B,M]
        all_ranks.append(np.stack(batch_ranks, axis=1))    # [B,M]
        all_indices.append(idx.numpy())

        if batch_no % args.progress_every == 0 or batch_no == len(loader):
            print(
                f"[AUDIT] batch {batch_no}/{len(loader)} "
                f"elapsed={time.time() - started:.1f}s"
            )

    losses = np.concatenate(all_losses, axis=0)
    ranks = np.concatenate(all_ranks, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    # Restore original dataset order, though loader is already shuffle=False.
    order = np.argsort(indices)
    losses = losses[order]
    ranks = ranks[order]

    base_loss = losses[:, zero_idx]
    dense_loss = losses[:, dense_idx]
    single_losses = losses[:, single_indices]

    utilities = base_loss[:, None] - single_losses  # [N,S]
    best_source_local = np.argmin(single_losses, axis=1)
    best_single_loss = np.min(single_losses, axis=1)
    best_single_mask_index = np.asarray(single_indices)[best_source_local]

    if pair_indices:
        pair_losses = losses[:, pair_indices]
        best_pair_local = np.argmin(pair_losses, axis=1)
        best_pair_loss = np.min(pair_losses, axis=1)
        best_pair_mask_index = np.asarray(pair_indices)[best_pair_local]
    else:
        best_pair_loss = best_single_loss.copy()
        best_pair_mask_index = best_single_mask_index.copy()

    # Safe oracle may choose no source.
    safe1_candidates = np.column_stack([base_loss, single_losses])
    safe1_choice = np.argmin(safe1_candidates, axis=1)
    safe1_loss = np.min(safe1_candidates, axis=1)

    up_to_2_indices = [
        i for i, m in enumerate(masks_np) if m.sum() <= min(2, n_sources)
    ]
    safe2_subset = losses[:, up_to_2_indices]
    safe2_local = np.argmin(safe2_subset, axis=1)
    safe2_loss = np.min(safe2_subset, axis=1)
    safe2_mask_index = np.asarray(up_to_2_indices)[safe2_local]

    oracle_any_idx = np.argmin(losses, axis=1)
    oracle_any_loss = np.min(losses, axis=1)

    # For rank metrics, apply the same CE-selected routing decision.
    arange_n = np.arange(len(data.users))
    base_rank = ranks[:, zero_idx]
    dense_rank = ranks[:, dense_idx]
    oracle1_rank = ranks[arange_n, best_single_mask_index]
    oracle2_rank = ranks[arange_n, best_pair_mask_index]
    safe1_mask_idx = np.empty(len(data.users), dtype=int)
    # safe1_choice 0 means target-only, 1..S means respective single source
    safe1_mask_idx[safe1_choice == 0] = zero_idx
    for d in range(n_sources):
        safe1_mask_idx[safe1_choice == d + 1] = single_indices[d]
    safe1_rank = ranks[arange_n, safe1_mask_idx]
    safe2_rank = ranks[arange_n, safe2_mask_index]
    oracle_any_rank = ranks[arange_n, oracle_any_idx]

    # -------------------------------------------------------------------------
    # Per-user CSV
    # -------------------------------------------------------------------------
    csv_path = output_dir / f"utility_{args.audit_split}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        header = [
            "user_id",
            "target_seq_len",
            *[f"{d}_seq_len" for d in source_domains],
            "loss_target_only",
            "loss_dense_all",
        ]
        for d in source_domains:
            header += [f"loss_only_{d}", f"utility_{d}"]

        header += [
            "best_source",
            "best_source_utility",
            "best_single_loss",
            "best_pair_mask",
            "best_pair_loss",
            "safe_top1_loss",
            "safe_top2_loss",
            "oracle_any_mask",
            "oracle_any_loss",
            "rank_target_only",
            "rank_dense_all",
            "rank_oracle_top1",
            "rank_oracle_top2",
            "rank_safe_top1",
            "rank_safe_top2",
            "rank_oracle_any",
            "utility_std_across_sources",
            "utility_range_across_sources",
        ]
        writer.writerow(header)

        for i, u in enumerate(data.users):
            best_d = int(best_source_local[i])
            row = [
                str(u),
                int(data.target_seq_len[i]),
                *[int(x) for x in data.source_seq_lens[i]],
                float(base_loss[i]),
                float(dense_loss[i]),
            ]
            for d in range(n_sources):
                row += [
                    float(single_losses[i, d]),
                    float(utilities[i, d]),
                ]

            row += [
                source_domains[best_d],
                float(utilities[i, best_d]),
                float(best_single_loss[i]),
                mask_names[int(best_pair_mask_index[i])],
                float(best_pair_loss[i]),
                float(safe1_loss[i]),
                float(safe2_loss[i]),
                mask_names[int(oracle_any_idx[i])],
                float(oracle_any_loss[i]),
                int(base_rank[i]),
                int(dense_rank[i]),
                int(oracle1_rank[i]),
                int(oracle2_rank[i]),
                int(safe1_rank[i]),
                int(safe2_rank[i]),
                int(oracle_any_rank[i]),
                float(np.std(utilities[i])),
                float(np.max(utilities[i]) - np.min(utilities[i])),
            ]
            writer.writerow(row)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    def rank_metrics(r: np.ndarray, k: int = 10) -> Dict[str, float]:
        r = np.asarray(r, dtype=float)
        hit = (r <= k).astype(float)
        ndcg = np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0)
        rr = 1.0 / r
        return {
            f"Recall@{k}": float(hit.mean()),
            f"NDCG@{k}": float(ndcg.mean()),
            "MRR": float(rr.mean()),
        }

    negative_rates = {
        d: float(np.mean(utilities[:, j] < 0.0))
        for j, d in enumerate(source_domains)
    }
    positive_rates = {
        d: float(np.mean(utilities[:, j] > 0.0))
        for j, d in enumerate(source_domains)
    }
    mean_utilities = {
        d: float(np.mean(utilities[:, j]))
        for j, d in enumerate(source_domains)
    }
    median_utilities = {
        d: float(np.median(utilities[:, j]))
        for j, d in enumerate(source_domains)
    }

    best_counts = {
        d: int(np.sum(best_source_local == j))
        for j, d in enumerate(source_domains)
    }
    best_fracs = {
        d: float(np.mean(best_source_local == j))
        for j, d in enumerate(source_domains)
    }

    no_source_best = np.all(single_losses >= base_loss[:, None], axis=1)

    summary = {
        "target_domain": args.target_dom,
        "audit_split": args.audit_split,
        "n_users": len(data.users),
        "source_domains": list(source_domains),
        "definition": {
            "utility": "CE_loss(target_only) - CE_loss(target + one source)",
            "positive": "source lowers CE loss",
            "negative": "source raises CE loss",
            "note": (
                "Utility is an instantaneous fixed-probe quantity, not a claim about "
                "full retraining gain."
            ),
        },
        "loss": {
            "target_only_mean": float(np.mean(base_loss)),
            "dense_all_mean": float(np.mean(dense_loss)),
            "oracle_top1_source_mean": float(np.mean(best_single_loss)),
            "oracle_top2_exact_mean": float(np.mean(best_pair_loss)),
            "safe_top1_mean": float(np.mean(safe1_loss)),
            "safe_top2_mean": float(np.mean(safe2_loss)),
            "oracle_any_mask_mean": float(np.mean(oracle_any_loss)),
            "dense_minus_oracle_top1": float(np.mean(dense_loss - best_single_loss)),
            "dense_minus_safe_top1": float(np.mean(dense_loss - safe1_loss)),
            "dense_minus_safe_top2": float(np.mean(dense_loss - safe2_loss)),
            "target_only_minus_dense": float(np.mean(base_loss - dense_loss)),
        },
        "utility_by_source": {
            d: {
                "mean": mean_utilities[d],
                "median": median_utilities[d],
                "negative_rate": negative_rates[d],
                "positive_rate": positive_rates[d],
                "p10": float(np.quantile(utilities[:, j], 0.10)),
                "p25": float(np.quantile(utilities[:, j], 0.25)),
                "p75": float(np.quantile(utilities[:, j], 0.75)),
                "p90": float(np.quantile(utilities[:, j], 0.90)),
            }
            for j, d in enumerate(source_domains)
        },
        "best_source_distribution": {
            d: {
                "count": best_counts[d],
                "fraction": best_fracs[d],
            }
            for d in source_domains
        },
        "heterogeneity": {
            "mean_within_user_utility_std": float(
                np.mean(np.std(utilities, axis=1))
            ),
            "median_within_user_utility_range": float(
                np.median(np.max(utilities, axis=1) - np.min(utilities, axis=1))
            ),
            "no_source_beats_all_single_sources_rate": float(no_source_best.mean()),
            "best_source_not_global_majority_rate": None,
        },
        "selection_advantage": {
            "oracle_top1_beats_dense_rate": float(
                np.mean(best_single_loss < dense_loss)
            ),
            "safe_top1_beats_dense_rate": float(
                np.mean(safe1_loss < dense_loss)
            ),
            "safe_top2_beats_dense_rate": float(
                np.mean(safe2_loss < dense_loss)
            ),
            "dense_beats_target_only_rate": float(
                np.mean(dense_loss < base_loss)
            ),
        },
        "ranking_metrics": {
            "target_only": rank_metrics(base_rank),
            "dense_all": rank_metrics(dense_rank),
            "oracle_top1_source_by_CE": rank_metrics(oracle1_rank),
            "oracle_top2_exact_by_CE": rank_metrics(oracle2_rank),
            "safe_top1_by_CE": rank_metrics(safe1_rank),
            "safe_top2_by_CE": rank_metrics(safe2_rank),
            "oracle_any_mask_by_CE": rank_metrics(oracle_any_rank),
        },
        "mask_mean_losses": {
            mask_names[m]: float(np.mean(losses[:, m]))
            for m in range(len(mask_names))
        },
    }

    # How non-trivial is personalized routing relative to always choosing
    # the globally most common "best" source?
    majority_source = max(best_counts, key=best_counts.get)
    majority_idx = source_domains.index(majority_source)
    summary["heterogeneity"]["global_majority_best_source"] = majority_source
    summary["heterogeneity"]["best_source_not_global_majority_rate"] = float(
        np.mean(best_source_local != majority_idx)
    )

    summary_path = output_dir / f"utility_summary_{args.audit_split}.json"
    write_json(summary_path, summary)

    # Pretty terminal output
    print("\n" + "=" * 88)
    print("Multi-source utility audit summary")
    print("=" * 88)
    print(f"Target domain                : {args.target_dom}")
    print(f"Audit split                  : {args.audit_split}")
    print(f"Users                        : {len(data.users)}")
    print()
    print("Mean utility / negative rate:")
    for d in source_domains:
        x = summary["utility_by_source"][d]
        print(
            f"  {d:>5s}: mean={x['mean']:+.6f} "
            f"median={x['median']:+.6f} "
            f"negative={100*x['negative_rate']:.2f}% "
            f"positive={100*x['positive_rate']:.2f}%"
        )

    print("\nBest source distribution:")
    for d in source_domains:
        x = summary["best_source_distribution"][d]
        print(f"  {d:>5s}: {x['count']:>6d} ({100*x['fraction']:.2f}%)")

    print("\nMean CE loss:")
    print(f"  Target only                : {summary['loss']['target_only_mean']:.6f}")
    print(f"  Dense all sources          : {summary['loss']['dense_all_mean']:.6f}")
    print(f"  Oracle exact Top-1 source  : {summary['loss']['oracle_top1_source_mean']:.6f}")
    print(f"  Oracle exact Top-2 sources : {summary['loss']['oracle_top2_exact_mean']:.6f}")
    print(f"  Safe Top-1 (allow none)    : {summary['loss']['safe_top1_mean']:.6f}")
    print(f"  Safe Top-2 (allow <=2)     : {summary['loss']['safe_top2_mean']:.6f}")
    print(f"  Oracle any mask            : {summary['loss']['oracle_any_mask_mean']:.6f}")

    print("\nRouting motivation:")
    print(
        "  No source beats all singles : "
        f"{100*summary['heterogeneity']['no_source_beats_all_single_sources_rate']:.2f}%"
    )
    print(
        "  Best source != global majority: "
        f"{100*summary['heterogeneity']['best_source_not_global_majority_rate']:.2f}%"
    )
    print(
        "  Oracle Top-1 beats dense     : "
        f"{100*summary['selection_advantage']['oracle_top1_beats_dense_rate']:.2f}%"
    )
    print(
        "  Safe Top-2 beats dense       : "
        f"{100*summary['selection_advantage']['safe_top2_beats_dense_rate']:.2f}%"
    )
    print(
        "  Dense - Oracle Top1 CE gap   : "
        f"{summary['loss']['dense_minus_oracle_top1']:+.6f}"
    )

    print("\nRanking metrics @10:")
    for name, metrics in summary["ranking_metrics"].items():
        print(
            f"  {name:28s} "
            f"R@10={metrics['Recall@10']:.6f} "
            f"NDCG@10={metrics['NDCG@10']:.6f} "
            f"MRR={metrics['MRR']:.6f}"
        )

    print("=" * 88)
    print(f"[DONE] Per-user CSV : {csv_path}")
    print(f"[DONE] Summary JSON : {summary_path}")

    return csv_path, summary_path


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = parse_args()
    set_all_seeds(args.seed)

    root = find_project_root(args.project_root)
    ensure_project_importable(root)

    dataset_dir = (
        Path(args.dataset_dir).expanduser()
        if args.dataset_dir
        else root / "dataset" / args.dataset_name
    )
    if not dataset_dir.is_absolute():
        dataset_dir = (root / dataset_dir).resolve()
    else:
        dataset_dir = dataset_dir.resolve()

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else root / "save" / "multisource_utility" / args.target_dom
    )
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else output_dir / "cache"
    )
    if not cache_dir.is_absolute():
        cache_dir = (root / cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    domains = list(args.domains)
    if args.target_dom not in domains:
        raise ValueError("--target-dom must be included in --domains")
    if len(domains) < 2:
        raise ValueError("At least 2 domains are required.")

    source_domains = [d for d in domains if d != args.target_dom]

    print("=" * 88)
    print("Frozen Expert Multi-source Utility Audit")
    print("=" * 88)
    print(f"Project root      : {root}")
    print(f"Dataset dir       : {dataset_dir}")
    print(f"Target domain     : {args.target_dom}")
    print(f"Source domains    : {source_domains}")
    print(f"Audit split       : {args.audit_split}")
    print(f"Output dir        : {output_dir}")
    print(f"Reuse cache       : {args.reuse_cache}")
    print()
    print("[IMPORTANT] This is USER-level routing audit, not temporal routing.")
    print("[IMPORTANT] Source representations use source TRAIN histories only.")
    print()

    # Resolve all .inter files needed.
    needed_files: Dict[str, Dict[str, Path]] = {}
    for d in domains:
        needed_files[d] = {}
        # source only needs train; target needs train/valid/audit
        splits = {"train"}
        if d == args.target_dom:
            splits.update({"valid", args.audit_split})
        for sp in sorted(splits):
            needed_files[d][sp] = resolve_inter_path(
                dataset_dir, args.dataset_name, d, sp
            )

    print("Input files:")
    for d in domains:
        for sp, p in needed_files[d].items():
            print(f"  {d:>5s} {sp:>5s}: {p}")

    # Parse rows now so dry-run can audit user alignment without model loading.
    raw_rows: Dict[str, Dict[str, List[InterRow]]] = {}
    raw_user_maps: Dict[str, Dict[str, Dict[str, InterRow]]] = {}
    raw_ambiguous: Dict[str, Dict[str, int]] = {}

    for d in domains:
        raw_rows[d] = {}
        raw_user_maps[d] = {}
        raw_ambiguous[d] = {}
        for sp, p in needed_files[d].items():
            rows = read_inter(p)
            umap, ambiguous = index_unique_by_user(rows)
            raw_rows[d][sp] = rows
            raw_user_maps[d][sp] = umap
            raw_ambiguous[d][sp] = ambiguous
            print(
                f"[PARSE] {d} {sp}: rows={len(rows)} "
                f"users={len(umap)} ambiguous_users={ambiguous}"
            )

    # User overlap before embeddings.
    alignment_pre = {}
    for sp in ["train", "valid", args.audit_split]:
        if sp not in raw_user_maps[args.target_dom]:
            continue
        tusers = set(raw_user_maps[args.target_dom][sp])
        src_counts = []
        for u in tusers:
            c = sum(u in raw_user_maps[d]["train"] for d in source_domains)
            src_counts.append(c)

        src_counts = np.asarray(src_counts, dtype=int)
        alignment_pre[sp] = {
            "target_users": len(tusers),
            "with_all_sources": int(np.sum(src_counts == len(source_domains))),
            "with_ge2_sources": int(np.sum(src_counts >= min(2, len(source_domains)))),
            "with_ge1_source": int(np.sum(src_counts >= 1)),
            "fraction_all_sources": (
                float(np.mean(src_counts == len(source_domains)))
                if len(src_counts) else None
            ),
        }

    print("\nPre-embedding alignment:")
    for sp, st in alignment_pre.items():
        print(
            f"  {sp:>5s}: target_users={st['target_users']} "
            f"all_sources={st['with_all_sources']} "
            f"({100*st['fraction_all_sources']:.2f}%)"
        )

    # Resolve checkpoints/config paths in dry-run by constructing configs.
    config_by_domain = {}
    ckpt_by_domain = {}

    for d in domains:
        cfg = build_recbole_config(root, d, args.seed, args.gpu_id)
        config_by_domain[d] = cfg
        ckpt = resolve_checkpoint(
            root, cfg, d, args.checkpoint_template
        )
        ckpt_by_domain[d] = ckpt
        print(f"[CKPT] {d}: {ckpt}")

    basic_config_dump = {
        "args": vars(args),
        "project_root": str(root),
        "dataset_dir": str(dataset_dir),
        "target_domain": args.target_dom,
        "source_domains": source_domains,
        "input_files": {
            d: {sp: str(p) for sp, p in x.items()}
            for d, x in needed_files.items()
        },
        "checkpoints": {d: str(p) for d, p in ckpt_by_domain.items()},
        "pre_embedding_alignment": alignment_pre,
        "raw_ambiguous_users": raw_ambiguous,
        "limitations": [
            "Processed BEST .inter files have no timestamp.",
            "This script audits user-level source utility only.",
            "Source train histories cannot be proven temporally prior to target samples.",
            "Utility is a fixed-probe instantaneous CE quantity, not full-retraining gain.",
        ],
    }
    write_json(output_dir / "config.json", basic_config_dump)

    if args.dry_run:
        print("\n[DONE] Dry run passed.")
        print(f"[DONE] Config: {output_dir / 'config.json'}")
        return 0

    # -------------------------------------------------------------------------
    # Encode frozen experts one domain at a time.
    # -------------------------------------------------------------------------
    encoded: Dict[str, Dict[str, EncodedSplit]] = {}
    target_item_embeddings_cpu: Optional[torch.Tensor] = None
    target_range: Optional[Tuple[int, int]] = None
    target_hidden_size: Optional[int] = None
    device_final: Optional[torch.device] = None

    for d in domains:
        print("\n" + "=" * 88)
        print(f"Loading frozen expert: {d}")
        print("=" * 88)

        cfg = config_by_domain[d]
        device = resolve_device(cfg, args.device)
        device_final = device

        dataset = create_domain_dataset(cfg)
        model = create_domain_model(cfg, dataset, device)
        load_checkpoint(
            model, ckpt_by_domain[d], device, strict=not args.non_strict
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        print(f"Dataset item_num : {dataset.item_num}")
        print(f"Dataset user_num : {dataset.user_num}")
        print(f"Hidden size      : {model.hidden_size}")
        print(f"Checkpoint       : {ckpt_by_domain[d]}")
        print(f"Device           : {device}")

        if d == args.target_dom:
            begin, end = model.domain_ranges[d]
            target_range = (int(begin), int(end))
            target_hidden_size = int(model.hidden_size)
            target_item_embeddings_cpu = (
                model.item_embedding.weight[begin:end + 1]
                .detach()
                .float()
                .cpu()
            )
            print(
                f"Target item range: [{begin}, {end}] "
                f"classes={end - begin + 1}"
            )

        encoded[d] = {}
        splits = ["train"] if d != args.target_dom else sorted(
            set(["train", "valid", args.audit_split])
        )

        for sp in splits:
            cache_path = cache_dir / f"{d}_{sp}_embeddings.npz"
            if args.reuse_cache and cache_path.exists():
                enc = load_encoded(cache_path)
                print(
                    f"[CACHE] {d}/{sp}: N={len(enc.users)}, "
                    f"H={enc.embeddings.shape[1]}"
                )
            else:
                need_targets = (d == args.target_dom)
                enc = encode_rows(
                    model=model,
                    dataset=dataset,
                    rows=raw_rows[d][sp],
                    device=device,
                    batch_size=args.encode_batch_size,
                    need_target_ids=need_targets,
                    progress_prefix=f"{d}/{sp}",
                )
                save_encoded(cache_path, enc)
                print(f"[SAVE CACHE] {cache_path}")

            encoded[d][sp] = enc

        del model
        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert target_item_embeddings_cpu is not None
    assert target_range is not None
    assert target_hidden_size is not None
    assert device_final is not None

    # -------------------------------------------------------------------------
    # Align train / valid / audit data.
    # -------------------------------------------------------------------------
    source_train_encoded = {
        d: encoded[d]["train"] for d in source_domains
    }

    aligned = {}
    alignment_stats = {}

    splits_to_align = sorted(set(["train", "valid", args.audit_split]))
    for sp in splits_to_align:
        a, st = align_for_target(
            target_encoded=encoded[args.target_dom][sp],
            source_encoded_train=source_train_encoded,
            source_domains=source_domains,
            target_begin_id=target_range[0],
            target_end_id=target_range[1],
            require_all_sources=not args.allow_missing_sources,
        )
        aligned[sp] = a
        alignment_stats[sp] = st
        print(
            f"[ALIGN] {sp}: kept={st['kept']}/{st['target_rows']} "
            f"all_sources={st['users_with_all_sources']} "
            f"skip_target={st['skip_target_outside_domain']} "
            f"skip_missing={st['skip_missing_source_when_required']}"
        )

    aligned["train"] = subsample_aligned(
        aligned["train"], args.max_train_users, args.seed + 101
    )
    aligned["valid"] = subsample_aligned(
        aligned["valid"], args.max_valid_users, args.seed + 102
    )
    aligned[args.audit_split] = subsample_aligned(
        aligned[args.audit_split], args.max_audit_users, args.seed + 103
    )

    write_json(
        output_dir / "alignment_summary.json",
        {
            "stats_before_subsample": alignment_stats,
            "after_subsample": {
                sp: len(aligned[sp].users)
                for sp in splits_to_align
            },
            "target_item_range": list(target_range),
            "n_target_classes": target_range[1] - target_range[0] + 1,
            "hidden_size": target_hidden_size,
        },
    )

    # -------------------------------------------------------------------------
    # Train probe.
    # -------------------------------------------------------------------------
    device = device_final
    target_item_embeddings = target_item_embeddings_cpu.to(device)

    probe = MaskFusionProbe(
        hidden_size=target_hidden_size,
        n_sources=len(source_domains),
        target_item_embeddings=target_item_embeddings,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)

    train_probe(
        model=probe,
        train_data=aligned["train"],
        valid_data=aligned["valid"],
        args=args,
        device=device,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------------
    # Final utility audit.
    # -------------------------------------------------------------------------
    audit_utility(
        model=probe,
        data=aligned[args.audit_split],
        source_domains=source_domains,
        args=args,
        device=device,
        output_dir=output_dir,
    )

    print("\nNext decision rule:")
    print("  Strong motivation if:")
    print("    - each source has a non-trivial negative utility rate;")
    print("    - best-source distribution is not collapsed to one domain;")
    print("    - Oracle/Safe sparse selection materially beats dense fusion.")
    print("  Weak motivation if:")
    print("    - one source dominates almost all users; or")
    print("    - dense fusion is already as good as oracle sparse selection.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
