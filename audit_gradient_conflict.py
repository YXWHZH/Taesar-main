#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taesar 原始序列—再生序列梯度冲突审计
======================================

用途
----
1. 读取原始目标域训练文件：
       BEST.<dom>.train.inter
2. 读取 Taesar 合并训练文件：
       BEST.<seed>.<dom>.train.inter
3. 通过“合并文件 - 原始文件”的多重集合差，识别新增再生样本；
4. 按 user_id 将原始样本与新增再生样本配对；
5. 加载 Taesar 下游 SASRec checkpoint；
6. 对每一对样本计算指定参数范围内的损失梯度余弦相似度；
7. 输出逐用户 JSONL、汇总 JSON、配对预览和参数清单。

默认策略
--------
- checkpoint：原始目标域训练得到的 sim 模型；
- target_mode：same_only，只比较目标 item 相同的原始/再生样本；
- loss_mode：full，直接调用项目 SASRec.calculate_loss()；
- param_scope：last_layer，最后一个 Transformer block + 顶层 LayerNorm；
- model_mode：eval，关闭 dropout，减少随机噪声；
- sample_size：500。

推荐命令
--------
python audit_gradient_conflict.py \
  --target-dom dom1 \
  --seed 2025 \
  --checkpoint-train-type sim \
  --sample-size 500 \
  --target-mode same_only \
  --loss-mode full \
  --param-scope last_layer \
  --gpu-id 0

只检查数据配对，不加载模型：
python audit_gradient_conflict.py --target-dom dom1 --dry-run

检查最后一个目标物品的梯度，而非完整自回归序列损失：
python audit_gradient_conflict.py \
  --target-dom dom1 \
  --loss-mode last \
  --checkpoint-train-type sim

说明
----
- 必须从 Taesar 项目目录内运行，或显式传入 --project-root。
- 本脚本不会修改任何 .inter 文件。
- 本脚本不使用测试集，也不进行模型参数更新。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InterRow:
    """从 .inter TSV 文件中读取的一条序列样本。"""

    user_token: str
    item_tokens: Tuple[str, ...]
    target_token: str
    line_no: int
    source_file: str

    @property
    def signature(self) -> Tuple[Tuple[str, ...], str]:
        """用于判定原始行是否已包含在合并文件中的签名。"""

        return self.item_tokens, self.target_token


@dataclass(frozen=True)
class PairRecord:
    """一个用户的原始—再生训练样本对。"""

    user_token: str
    original: InterRow
    regenerated: InterRow
    target_same: bool
    sequence_jaccard: float
    original_length: int
    regenerated_length: int


@dataclass
class PairBuildStats:
    original_rows: int = 0
    combined_rows: int = 0
    original_users: int = 0
    combined_users: int = 0
    extra_rows: int = 0
    extra_users: int = 0
    users_without_original: int = 0
    users_without_extra: int = 0
    ambiguous_original_users: int = 0
    ambiguous_extra_users: int = 0
    target_same_candidates: int = 0
    target_different_candidates: int = 0
    pairs_built: int = 0
    pairs_skipped_target_mismatch: int = 0


# -----------------------------------------------------------------------------
# 参数解析与路径
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit gradient conflict between original and Taesar-regenerated sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--project-root", type=str, default=None, help="Taesar 项目根目录。")
    parser.add_argument("--target-dom", type=str, default="dom1", choices=["dom1", "dom2", "dom3", "dom4"])
    parser.add_argument("--seed", type=int, default=2025, help="Taesar 再生数据与 checkpoint 使用的随机种子。")
    parser.add_argument("--gpu-id", type=int, default=0, help="写入 RecBole/Hydra 配置的 GPU ID。")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto、cpu、cuda、cuda:0 等。auto 使用 RecBole 配置计算出的 device。",
    )

    parser.add_argument("--original-inter", type=str, default=None, help="原始目标域 train.inter；默认自动定位。")
    parser.add_argument("--combined-inter", type=str, default=None, help="原始+再生 train.inter；默认自动定位。")

    parser.add_argument(
        "--checkpoint-train-type",
        type=str,
        default="sim",
        choices=["sim", "new", "full", "none"],
        help="加载哪一种下游 checkpoint；none 表示随机初始化模型。",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="显式 checkpoint 路径，优先级最高。")
    parser.add_argument("--non-strict", action="store_true", help="以 strict=False 加载 state_dict。")

    parser.add_argument(
        "--target-mode",
        type=str,
        default="same_only",
        choices=["same_only", "original", "own"],
        help=(
            "same_only：只保留原始/再生 target 相同的样本；"
            "original：两条序列都使用原始 target；"
            "own：分别使用各自文件中的 target。"
        ),
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="full",
        choices=["full", "last"],
        help="full 调用 calculate_loss；last 只计算最后一个目标物品的交叉熵。",
    )
    parser.add_argument(
        "--param-scope",
        type=str,
        default="last_layer",
        choices=["last_layer", "transformer", "all_no_embedding", "all"],
        help="参与梯度余弦计算的参数范围。",
    )
    parser.add_argument(
        "--param-regex",
        type=str,
        default=None,
        help="自定义参数名正则；提供后覆盖 --param-scope。",
    )
    parser.add_argument(
        "--model-mode",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="eval 关闭 dropout；train 更接近真实训练，但随机性更强。",
    )

    parser.add_argument("--sample-size", type=int, default=500, help="抽取多少个用户；0 表示全部。")
    parser.add_argument("--sample-seed", type=int, default=2025, help="用户抽样随机种子。")
    parser.add_argument("--max-print-pairs", type=int, default=10, help="终端和预览文件打印的配对数量。")
    parser.add_argument("--progress-every", type=int, default=25, help="每处理多少对样本打印一次进度。")
    parser.add_argument("--dry-run", action="store_true", help="只做路径、文件和配对检查，不加载模型。")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；默认 save/gradient_audit/<target_dom>/。",
    )
    parser.add_argument(
        "--save-param-cosines",
        action="store_true",
        help="在逐用户 JSONL 中额外保存每个参数张量的 cosine，会显著增大文件。",
    )

    return parser.parse_args()


def find_project_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "config" / "overall.yaml").exists():
            raise FileNotFoundError(f"--project-root 下未找到 config/overall.yaml: {root}")
        return root

    start = Path(__file__).resolve().parent
    candidates = [start, *start.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    seen: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "config" / "overall.yaml").exists() and (candidate / "model" / "seq2seq_sasrec.py").exists():
            return candidate

    raise FileNotFoundError(
        "无法自动找到 Taesar 项目根目录。请把脚本复制到项目根目录，"
        "或传入 --project-root /root/autodl-tmp/Taesar-main"
    )


def ensure_project_importable(project_root: Path) -> None:
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(project_root)


# -----------------------------------------------------------------------------
# Hydra + RecBole 配置、Dataset 与 checkpoint
# -----------------------------------------------------------------------------


def build_recbole_config(
    project_root: Path,
    target_dom: str,
    seed: int,
    gpu_id: int,
    dataset_train_type: str,
) -> Any:
    """按 finetune.py 的方式构造 RecBole Config，但不启动 wandb/Hydra job。"""

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from recbole.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "缺少 hydra/omegaconf/recbole。请先激活 Taesar conda 环境再运行。"
        ) from exc

    config_dir = project_root / "config"
    overrides = [
        f"target_dom={target_dom}",
        f"seed={seed}",
        "stage=tun",
        f"train_type={dataset_train_type}",
        f"gpu_id={gpu_id}",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        hydra_cfg = compose(config_name="overall", overrides=overrides)

    # base/hydra 中含 ${now:...}，审计脚本不需要这些字段；移除后再 resolve。
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

    if dataset_train_type == "sim":
        config["benchmark_filename"] = [
            f"{target_dom}.train",
            f"{target_dom}.valid",
            f"{target_dom}.test",
        ]
    elif dataset_train_type == "new":
        config["benchmark_filename"] = [
            f"{seed}.{target_dom}.train",
            f"{target_dom}.valid",
            f"{target_dom}.test",
        ]
    elif dataset_train_type == "full":
        config["benchmark_filename"] = [
            "full.train",
            f"{target_dom}.valid",
            f"{target_dom}.test",
        ]
    else:
        raise ValueError(f"Unsupported dataset_train_type: {dataset_train_type}")

    return config


def resolve_device(config: Any, requested: str) -> torch.device:
    if requested == "auto":
        device = config["device"]
        if not isinstance(device, torch.device):
            device = torch.device(str(device))
    else:
        requested = "cuda:0" if requested == "cuda" else requested
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求使用 {device}，但 torch.cuda.is_available() 为 False。")

    config["device"] = device
    config["use_gpu"] = device.type == "cuda"
    return device


def create_dataset(config: Any, dataset_train_type: str) -> Any:
    if dataset_train_type == "full":
        from data.sequential_dataset import SequentialDataset
    else:
        from recbole.data.dataset import SequentialDataset

    return SequentialDataset(config)


def create_model(config: Any, dataset: Any, flag: str, device: torch.device) -> torch.nn.Module:
    from model.seq2seq_sasrec import SASRec

    model = SASRec(config, dataset).to(device)
    model.flag = flag if flag != "none" else "sim"
    return model


def resolve_checkpoint_path(
    project_root: Path,
    config: Any,
    target_dom: str,
    train_type: str,
    explicit: Optional[str],
) -> Optional[Path]:
    if train_type == "none":
        return None

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"显式 checkpoint 不存在: {path}")
        return path

    expected = Path(f"{config['tune_ckpt']}-{target_dom}-{train_type}")
    if not expected.is_absolute():
        expected = project_root / expected
    expected = expected.resolve()
    if expected.exists():
        return expected

    pattern = f"*tune.ckpt-{target_dom}-{train_type}"
    found: List[Path] = []
    for base in (project_root / "checkpoint", project_root / "save"):
        if base.exists():
            found.extend(p.resolve() for p in base.rglob(pattern) if p.is_file())

    if not found:
        raise FileNotFoundError(
            "未找到 checkpoint。预期路径为：\n"
            f"  {expected}\n"
            "也未在 checkpoint/ 与 save/ 下找到匹配文件。请用 --checkpoint 显式指定。"
        )

    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(found) > 1:
        print("[WARN] 找到多个候选 checkpoint，自动选择最新修改的一个：")
        for candidate in found[:5]:
            print(f"       {candidate}")
    return found[0]


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device, strict: bool) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, Mapping):
        state_dict = checkpoint
    else:
        raise TypeError(f"无法识别 checkpoint 内容类型: {type(checkpoint)}")

    load_result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"[INFO] non-strict load missing_keys={len(load_result.missing_keys)}, unexpected_keys={len(load_result.unexpected_keys)}")
        if load_result.missing_keys:
            print("       missing example:", load_result.missing_keys[:10])
        if load_result.unexpected_keys:
            print("       unexpected example:", load_result.unexpected_keys[:10])

    return dict(checkpoint) if isinstance(checkpoint, Mapping) else {}


# -----------------------------------------------------------------------------
# .inter TSV 读取、差集与配对
# -----------------------------------------------------------------------------


def base_field_name(header: str) -> str:
    return header.split(":", 1)[0].strip()


def read_inter_rows(path: Path) -> List[InterRow]:
    if not path.exists():
        raise FileNotFoundError(f".inter 文件不存在: {path}")

    rows: List[InterRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"文件没有表头: {path}")

        base_to_header = {base_field_name(h): h for h in reader.fieldnames}
        required = ("user_id", "item_id_list", "item_id")
        missing = [field for field in required if field not in base_to_header]
        if missing:
            raise ValueError(
                f"文件缺少必要字段 {missing}: {path}\n"
                f"实际表头: {reader.fieldnames}"
            )

        user_col = base_to_header["user_id"]
        seq_col = base_to_header["item_id_list"]
        target_col = base_to_header["item_id"]

        for line_no, raw in enumerate(reader, start=2):
            user = (raw.get(user_col) or "").strip()
            seq_text = (raw.get(seq_col) or "").strip()
            target = (raw.get(target_col) or "").strip()

            if not user or not target:
                continue

            items = tuple(token for token in seq_text.split() if token)
            if not items:
                # SASRec 至少需要一个历史物品。
                continue

            rows.append(
                InterRow(
                    user_token=user,
                    item_tokens=items,
                    target_token=target,
                    line_no=line_no,
                    source_file=str(path),
                )
            )

    return rows


def group_by_user(rows: Iterable[InterRow]) -> Dict[str, List[InterRow]]:
    grouped: Dict[str, List[InterRow]] = defaultdict(list)
    for row in rows:
        grouped[row.user_token].append(row)
    return dict(grouped)


def sequence_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def find_extra_rows(
    original_by_user: Mapping[str, Sequence[InterRow]],
    combined_by_user: Mapping[str, Sequence[InterRow]],
) -> Dict[str, List[InterRow]]:
    """对每个用户执行多重集合差 combined - original。"""

    extras: Dict[str, List[InterRow]] = {}
    for user, combined_rows in combined_by_user.items():
        original_counter = Counter(row.signature for row in original_by_user.get(user, ()))
        user_extras: List[InterRow] = []

        for row in combined_rows:
            signature = row.signature
            if original_counter[signature] > 0:
                original_counter[signature] -= 1
            else:
                user_extras.append(row)

        if user_extras:
            extras[user] = user_extras

    return extras


def choose_pair_for_user(
    user: str,
    original_rows: Sequence[InterRow],
    extra_rows: Sequence[InterRow],
    target_mode: str,
) -> Optional[PairRecord]:
    candidates: List[Tuple[Tuple[float, float, float, float], InterRow, InterRow]] = []

    for original in original_rows:
        for regenerated in extra_rows:
            target_same = original.target_token == regenerated.target_token
            if target_mode == "same_only" and not target_same:
                continue

            jaccard = sequence_jaccard(original.item_tokens, regenerated.item_tokens)
            # 稳定、确定性的选择规则：
            # 1) target 相同优先；2) 再生序列更长优先；3) 上下文相似度高优先；4) 行号靠前优先。
            score = (
                1.0 if target_same else 0.0,
                float(len(regenerated.item_tokens)),
                jaccard,
                -float(regenerated.line_no),
            )
            candidates.append((score, original, regenerated))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, original, regenerated = candidates[0]
    return PairRecord(
        user_token=user,
        original=original,
        regenerated=regenerated,
        target_same=original.target_token == regenerated.target_token,
        sequence_jaccard=sequence_jaccard(original.item_tokens, regenerated.item_tokens),
        original_length=len(original.item_tokens),
        regenerated_length=len(regenerated.item_tokens),
    )


def build_pairs(
    original_rows: Sequence[InterRow],
    combined_rows: Sequence[InterRow],
    target_mode: str,
) -> Tuple[List[PairRecord], PairBuildStats]:
    stats = PairBuildStats(
        original_rows=len(original_rows),
        combined_rows=len(combined_rows),
    )

    original_by_user = group_by_user(original_rows)
    combined_by_user = group_by_user(combined_rows)
    extras_by_user = find_extra_rows(original_by_user, combined_by_user)

    stats.original_users = len(original_by_user)
    stats.combined_users = len(combined_by_user)
    stats.extra_rows = sum(len(rows) for rows in extras_by_user.values())
    stats.extra_users = len(extras_by_user)
    stats.users_without_original = sum(1 for user in extras_by_user if user not in original_by_user)
    stats.users_without_extra = sum(1 for user in original_by_user if user not in extras_by_user)
    stats.ambiguous_original_users = sum(1 for rows in original_by_user.values() if len(rows) > 1)
    stats.ambiguous_extra_users = sum(1 for rows in extras_by_user.values() if len(rows) > 1)

    pairs: List[PairRecord] = []
    for user in sorted(set(original_by_user) & set(extras_by_user)):
        original_list = original_by_user[user]
        extra_list = extras_by_user[user]

        for original in original_list:
            for regenerated in extra_list:
                if original.target_token == regenerated.target_token:
                    stats.target_same_candidates += 1
                else:
                    stats.target_different_candidates += 1

        pair = choose_pair_for_user(user, original_list, extra_list, target_mode)
        if pair is None:
            if target_mode == "same_only":
                stats.pairs_skipped_target_mismatch += 1
            continue
        pairs.append(pair)

    stats.pairs_built = len(pairs)
    return pairs, stats


def locate_dataset_dir(project_root: Path, config: Any) -> Path:
    dataset_name = str(config["dataset"])
    configured = Path(str(config["data_path"]))
    if not configured.is_absolute():
        configured = project_root / configured

    candidates = [
        configured / dataset_name,
        configured,
        project_root / "dataset" / dataset_name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "无法定位数据集目录。检查过：\n" + "\n".join(f"  {p}" for p in candidates)
    )


def resolve_inter_paths(
    project_root: Path,
    config: Any,
    target_dom: str,
    seed: int,
    explicit_original: Optional[str],
    explicit_combined: Optional[str],
) -> Tuple[Path, Path]:
    data_dir = locate_dataset_dir(project_root, config)
    dataset_name = str(config["dataset"])

    def resolve_one(explicit: Optional[str], default_name: str) -> Path:
        if explicit:
            path = Path(explicit).expanduser()
            if not path.is_absolute():
                path = project_root / path
            return path.resolve()
        return (data_dir / default_name).resolve()

    original = resolve_one(explicit_original, f"{dataset_name}.{target_dom}.train.inter")
    combined = resolve_one(explicit_combined, f"{dataset_name}.{seed}.{target_dom}.train.inter")
    return original, combined


# -----------------------------------------------------------------------------
# Token 映射与 Interaction 构造
# -----------------------------------------------------------------------------


def get_token_mapping(dataset: Any, field: str) -> Mapping[str, int]:
    mapping = getattr(dataset, "field2token_id", {}).get(field)
    if mapping is None:
        raise KeyError(f"Dataset 中没有字段 {field!r} 的 token→id 映射。")
    return mapping


def token_to_inner_id(mapping: Mapping[Any, Any], token: str) -> int:
    candidates: List[Any] = [token]
    try:
        candidates.append(int(token))
    except (TypeError, ValueError):
        pass

    for candidate in candidates:
        if candidate in mapping:
            return int(mapping[candidate])

    raise KeyError(token)


def truncate_history(tokens: Sequence[str], max_length: int) -> Tuple[str, ...]:
    if len(tokens) <= max_length:
        return tuple(tokens)
    return tuple(tokens[-max_length:])


def make_interaction(
    model: torch.nn.Module,
    dataset: Any,
    item_tokens: Sequence[str],
    target_token: str,
    device: torch.device,
) -> Any:
    from recbole.data.interaction import Interaction

    item_mapping = get_token_mapping(dataset, model.ITEM_ID)
    max_length = int(model.max_seq_length)
    truncated = truncate_history(item_tokens, max_length)

    item_ids = [token_to_inner_id(item_mapping, token) for token in truncated]
    target_id = token_to_inner_id(item_mapping, target_token)

    seq_len = len(item_ids)
    if seq_len < 1:
        raise ValueError("历史序列为空。")

    padded = torch.zeros((1, max_length), dtype=torch.long)
    padded[0, :seq_len] = torch.tensor(item_ids, dtype=torch.long)

    interaction = Interaction(
        {
            model.ITEM_SEQ: padded,
            model.ITEM_SEQ_LEN: torch.tensor([seq_len], dtype=torch.long),
            model.POS_ITEM_ID: torch.tensor([target_id], dtype=torch.long),
        }
    )
    return interaction.to(device)


# -----------------------------------------------------------------------------
# 参数选择、损失与梯度统计
# -----------------------------------------------------------------------------


def select_named_parameters(
    model: torch.nn.Module,
    scope: str,
    custom_regex: Optional[str],
) -> List[Tuple[str, torch.nn.Parameter]]:
    all_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    if custom_regex:
        pattern = re.compile(custom_regex)
        selected = [(name, param) for name, param in all_named if pattern.search(name)]
    elif scope == "all":
        selected = all_named
    elif scope == "all_no_embedding":
        selected = [
            (name, param)
            for name, param in all_named
            if "embedding" not in name.lower()
        ]
    elif scope == "transformer":
        selected = [
            (name, param)
            for name, param in all_named
            if name.startswith("trm_encoder.") or name.startswith("LayerNorm.")
        ]
    elif scope == "last_layer":
        layer_indices: List[int] = []
        for name, _ in all_named:
            match = re.search(r"(?:^|\.)trm_encoder\.layer\.(\d+)\.", name)
            if match:
                layer_indices.append(int(match.group(1)))

        if layer_indices:
            last_idx = max(layer_indices)
            prefix = f"trm_encoder.layer.{last_idx}."
            selected = [
                (name, param)
                for name, param in all_named
                if name.startswith(prefix) or name.startswith("LayerNorm.")
            ]
        else:
            # 不同 RecBole 版本参数命名可能变化；找不到层号时退化为整个 Transformer。
            selected = [
                (name, param)
                for name, param in all_named
                if name.startswith("trm_encoder.") or name.startswith("LayerNorm.")
            ]
    else:
        raise ValueError(f"Unknown param scope: {scope}")

    if not selected:
        sample_names = "\n".join(f"  {name}" for name, _ in all_named[:100])
        raise RuntimeError(
            "参数筛选结果为空。模型前100个参数名如下：\n" + sample_names
        )

    return selected


def compute_last_item_loss(model: torch.nn.Module, interaction: Any) -> torch.Tensor:
    item_seq = interaction[model.ITEM_SEQ]
    item_seq_len = interaction[model.ITEM_SEQ_LEN]
    pos_items = interaction[model.POS_ITEM_ID]

    output = model.forward(item_seq, item_seq_len)
    seq_output = model.gather_indexes(output, item_seq_len - 1)
    logits = torch.matmul(seq_output, model.item_embedding.weight.transpose(0, 1))
    return torch.nn.functional.cross_entropy(logits, pos_items)


def compute_loss(model: torch.nn.Module, interaction: Any, loss_mode: str) -> torch.Tensor:
    if loss_mode == "full":
        loss = model.calculate_loss(interaction)
    elif loss_mode == "last":
        loss = compute_last_item_loss(model, interaction)
    else:
        raise ValueError(loss_mode)

    if isinstance(loss, tuple):
        loss = sum(loss)
    if loss.ndim != 0:
        loss = loss.mean()
    return loss


def compute_gradients(
    model: torch.nn.Module,
    interaction: Any,
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    loss_mode: str,
) -> Tuple[float, List[Optional[torch.Tensor]]]:
    model.zero_grad(set_to_none=True)
    loss = compute_loss(model, interaction, loss_mode)

    if not torch.isfinite(loss):
        raise FloatingPointError(f"loss 非有限值: {loss.detach().cpu().item()}")

    params = [param for _, param in named_params]
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    detached: List[Optional[torch.Tensor]] = []
    for grad in grads:
        detached.append(None if grad is None else grad.detach())

    return float(loss.detach().cpu().item()), detached


def gradient_pair_statistics(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grads_o: Sequence[Optional[torch.Tensor]],
    grads_r: Sequence[Optional[torch.Tensor]],
    save_param_cosines: bool,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    dot = 0.0
    norm_o_sq = 0.0
    norm_r_sq = 0.0
    used_tensors = 0
    unused_both = 0
    unused_original = 0
    unused_regenerated = 0
    param_cosines: Dict[str, Optional[float]] = {}

    for (name, _), grad_o, grad_r in zip(named_params, grads_o, grads_r):
        if grad_o is None and grad_r is None:
            unused_both += 1
            if save_param_cosines:
                param_cosines[name] = None
            continue
        if grad_o is None:
            unused_original += 1
            grad_o = torch.zeros_like(grad_r)
        if grad_r is None:
            unused_regenerated += 1
            grad_r = torch.zeros_like(grad_o)

        go = grad_o.float()
        gr = grad_r.float()
        tensor_dot = float(torch.sum(go * gr).detach().cpu().item())
        tensor_o_sq = float(torch.sum(go * go).detach().cpu().item())
        tensor_r_sq = float(torch.sum(gr * gr).detach().cpu().item())

        dot += tensor_dot
        norm_o_sq += tensor_o_sq
        norm_r_sq += tensor_r_sq
        used_tensors += 1

        if save_param_cosines:
            denom = math.sqrt(max(tensor_o_sq, 0.0) * max(tensor_r_sq, 0.0))
            param_cosines[name] = tensor_dot / (denom + eps) if denom > eps else None

    norm_o = math.sqrt(max(norm_o_sq, 0.0))
    norm_r = math.sqrt(max(norm_r_sq, 0.0))
    denominator = norm_o * norm_r
    cosine = dot / (denominator + eps) if denominator > eps else float("nan")

    # 当 dot<0 时，PCGrad 式投影会移除 g_r 在 g_o 方向上的负分量。
    if norm_r_sq <= eps:
        projection_retention = float("nan")
    elif dot >= 0.0 or norm_o_sq <= eps:
        projection_retention = 1.0
    else:
        projected_norm_sq = max(norm_r_sq - (dot * dot) / (norm_o_sq + eps), 0.0)
        projection_retention = math.sqrt(projected_norm_sq / (norm_r_sq + eps))

    result: Dict[str, Any] = {
        "gradient_dot": dot,
        "gradient_cosine": cosine,
        "grad_norm_original": norm_o,
        "grad_norm_regenerated": norm_r,
        "is_conflict": bool(math.isfinite(cosine) and cosine < 0.0),
        "is_strong_conflict_01": bool(math.isfinite(cosine) and cosine < -0.1),
        "is_strong_conflict_03": bool(math.isfinite(cosine) and cosine < -0.3),
        "projection_retention": projection_retention,
        "used_parameter_tensors": used_tensors,
        "unused_both": unused_both,
        "unused_original": unused_original,
        "unused_regenerated": unused_regenerated,
    }
    if save_param_cosines:
        result["parameter_cosines"] = param_cosines
    return result


# -----------------------------------------------------------------------------
# 汇总与输出
# -----------------------------------------------------------------------------


def finite_values(results: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values: List[float] = []
    for row in results:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            values.append(value_f)
    return np.asarray(values, dtype=np.float64)


def describe_array(values: np.ndarray) -> Dict[str, Optional[float]]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def make_summary(
    results: Sequence[Mapping[str, Any]],
    pair_stats: PairBuildStats,
    metadata: Mapping[str, Any],
    skipped: Counter,
) -> Dict[str, Any]:
    cosine = finite_values(results, "gradient_cosine")
    loss_o = finite_values(results, "loss_original")
    loss_r = finite_values(results, "loss_regenerated")
    norm_o = finite_values(results, "grad_norm_original")
    norm_r = finite_values(results, "grad_norm_regenerated")
    retention = finite_values(results, "projection_retention")

    valid_cos = cosine[np.isfinite(cosine)]
    conflict_rate = float(np.mean(valid_cos < 0.0)) if valid_cos.size else None
    conflict_rate_01 = float(np.mean(valid_cos < -0.1)) if valid_cos.size else None
    conflict_rate_03 = float(np.mean(valid_cos < -0.3)) if valid_cos.size else None

    return {
        "metadata": dict(metadata),
        "pair_build": asdict(pair_stats),
        "processed_results": len(results),
        "skipped": dict(skipped),
        "gradient_cosine": describe_array(cosine),
        "conflict_rate": conflict_rate,
        "conflict_rate_below_minus_0_1": conflict_rate_01,
        "conflict_rate_below_minus_0_3": conflict_rate_03,
        "loss_original": describe_array(loss_o),
        "loss_regenerated": describe_array(loss_r),
        "grad_norm_original": describe_array(norm_o),
        "grad_norm_regenerated": describe_array(norm_r),
        "projection_retention": describe_array(retention),
    }


def json_dump(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    return value


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(safe_json_value(dict(row)), ensure_ascii=False) + "\n")


def pair_preview_lines(pairs: Sequence[PairRecord], limit: int) -> List[str]:
    lines: List[str] = []
    for index, pair in enumerate(pairs[:limit], start=1):
        lines.extend(
            [
                f"[{index}] user_id={pair.user_token}",
                f"    target_same={pair.target_same}  jaccard={pair.sequence_jaccard:.4f}",
                f"    original_len={pair.original_length} target={pair.original.target_token}",
                f"    original_seq={' '.join(pair.original.item_tokens)}",
                f"    regenerated_len={pair.regenerated_length} target={pair.regenerated.target_token}",
                f"    regenerated_seq={' '.join(pair.regenerated.item_tokens)}",
                "",
            ]
        )
    return lines


def print_summary(summary: Mapping[str, Any]) -> None:
    cosine = summary["gradient_cosine"]
    print("\n" + "=" * 76)
    print("Gradient conflict audit summary")
    print("=" * 76)
    print(f"Processed pairs            : {summary['processed_results']}")
    print(f"Skipped                    : {summary['skipped']}")
    print(f"Cosine mean                : {cosine['mean']}")
    print(f"Cosine median              : {cosine['median']}")
    print(f"Cosine P10 / P90           : {cosine['p10']} / {cosine['p90']}")
    print(f"Conflict rate cosine < 0   : {summary['conflict_rate']}")
    print(f"Strong conflict < -0.1     : {summary['conflict_rate_below_minus_0_1']}")
    print(f"Strong conflict < -0.3     : {summary['conflict_rate_below_minus_0_3']}")
    print(f"Projection retention mean  : {summary['projection_retention']['mean']}")
    print("=" * 76)


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    started_at = time.time()

    project_root = find_project_root(args.project_root)
    ensure_project_importable(project_root)

    # 随机初始化模型时仍用 sim 数据集构造模型与 token 映射。
    dataset_train_type = args.checkpoint_train_type if args.checkpoint_train_type != "none" else "sim"
    config = build_recbole_config(
        project_root=project_root,
        target_dom=args.target_dom,
        seed=args.seed,
        gpu_id=args.gpu_id,
        dataset_train_type=dataset_train_type,
    )

    original_path, combined_path = resolve_inter_paths(
        project_root=project_root,
        config=config,
        target_dom=args.target_dom,
        seed=args.seed,
        explicit_original=args.original_inter,
        explicit_combined=args.combined_inter,
    )

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else project_root / "save" / "gradient_audit" / args.target_dom
    )
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_tag = (
        f"{args.checkpoint_train_type}_{args.loss_mode}_{args.param_scope}_"
        f"{args.target_mode}_n{args.sample_size if args.sample_size > 0 else 'all'}"
    )
    jsonl_path = output_dir / f"gradient_conflict_{run_tag}.jsonl"
    summary_path = output_dir / f"gradient_conflict_{run_tag}.summary.json"
    preview_path = output_dir / f"gradient_conflict_{run_tag}.pairs.txt"
    params_path = output_dir / f"gradient_conflict_{run_tag}.parameters.txt"

    print(f"Project root      : {project_root}")
    print(f"Original .inter   : {original_path}")
    print(f"Combined .inter   : {combined_path}")
    print(f"Output directory  : {output_dir}")

    original_rows = read_inter_rows(original_path)
    combined_rows = read_inter_rows(combined_path)
    pairs, pair_stats = build_pairs(original_rows, combined_rows, args.target_mode)

    print("\nPair construction statistics:")
    for key, value in asdict(pair_stats).items():
        print(f"  {key:32s}: {value}")

    if not pairs:
        raise RuntimeError(
            "没有构造出可用样本对。若 target_same_candidates 为 0，可先用 "
            "--target-mode original 做诊断；但正式结论建议优先 same_only。"
        )

    rng = random.Random(args.sample_seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    if args.sample_size > 0:
        pairs = pairs[: min(args.sample_size, len(pairs))]

    preview = pair_preview_lines(pairs, args.max_print_pairs)
    preview_path.write_text("\n".join(preview), encoding="utf-8")
    print("\n" + "\n".join(preview))

    dry_metadata = {
        "project_root": str(project_root),
        "target_dom": args.target_dom,
        "seed": args.seed,
        "original_inter": str(original_path),
        "combined_inter": str(combined_path),
        "target_mode": args.target_mode,
        "sample_size_requested": args.sample_size,
        "sample_size_selected": len(pairs),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        dry_summary = {
            "metadata": dry_metadata,
            "pair_build": asdict(pair_stats),
            "message": "Dry run finished; model was not loaded.",
        }
        json_dump(summary_path, dry_summary)
        print(f"[DONE] Dry run summary: {summary_path}")
        print(f"[DONE] Pair preview   : {preview_path}")
        return 0

    device = resolve_device(config, args.device)

    try:
        from recbole.utils import init_seed
    except ImportError as exc:
        raise RuntimeError("无法导入 recbole.utils.init_seed") from exc
    init_seed(args.sample_seed, config["reproducibility"])

    print(f"\nBuilding dataset ({dataset_train_type}) ...")
    dataset = create_dataset(config, dataset_train_type)
    print(f"Dataset item_num : {getattr(dataset, 'item_num', 'unknown')}")
    print(f"Dataset user_num : {getattr(dataset, 'user_num', 'unknown')}")

    model = create_model(
        config=config,
        dataset=dataset,
        flag=args.checkpoint_train_type,
        device=device,
    )

    checkpoint_path = resolve_checkpoint_path(
        project_root=project_root,
        config=config,
        target_dom=args.target_dom,
        train_type=args.checkpoint_train_type,
        explicit=args.checkpoint,
    )

    checkpoint_meta: Dict[str, Any] = {}
    if checkpoint_path is not None:
        print(f"Loading checkpoint : {checkpoint_path}")
        checkpoint_meta = load_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            strict=not args.non_strict,
        )
    else:
        print("[WARN] 使用随机初始化模型，不加载 checkpoint。")

    if args.model_mode == "eval":
        model.eval()
    else:
        model.train()

    named_params = select_named_parameters(model, args.param_scope, args.param_regex)
    selected_numel = sum(param.numel() for _, param in named_params)
    params_lines = [
        f"scope={args.param_scope}",
        f"custom_regex={args.param_regex}",
        f"tensor_count={len(named_params)}",
        f"parameter_count={selected_numel}",
        "",
    ]
    params_lines.extend(f"{name}\t{tuple(param.shape)}\t{param.numel()}" for name, param in named_params)
    params_path.write_text("\n".join(params_lines) + "\n", encoding="utf-8")

    print(f"Device             : {device}")
    print(f"Model mode         : {args.model_mode}")
    print(f"Loss mode          : {args.loss_mode}")
    print(f"Selected tensors   : {len(named_params)}")
    print(f"Selected parameters: {selected_numel:,}")
    print("Selected examples  :")
    for name, param in named_params[:12]:
        print(f"  {name:72s} {tuple(param.shape)}")

    results: List[Dict[str, Any]] = []
    skipped: Counter = Counter()

    for index, pair in enumerate(pairs, start=1):
        try:
            if args.target_mode == "same_only":
                target_o = pair.original.target_token
                target_r = pair.regenerated.target_token
            elif args.target_mode == "original":
                target_o = pair.original.target_token
                target_r = pair.original.target_token
            elif args.target_mode == "own":
                target_o = pair.original.target_token
                target_r = pair.regenerated.target_token
            else:
                raise ValueError(args.target_mode)

            original_interaction = make_interaction(
                model=model,
                dataset=dataset,
                item_tokens=pair.original.item_tokens,
                target_token=target_o,
                device=device,
            )
            regenerated_interaction = make_interaction(
                model=model,
                dataset=dataset,
                item_tokens=pair.regenerated.item_tokens,
                target_token=target_r,
                device=device,
            )

            # train 模式下为两次前向使用同一起始随机种子，降低纯 dropout 噪声。
            pair_seed = args.sample_seed + index
            torch.manual_seed(pair_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(pair_seed)
            loss_o, grads_o = compute_gradients(
                model=model,
                interaction=original_interaction,
                named_params=named_params,
                loss_mode=args.loss_mode,
            )

            torch.manual_seed(pair_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(pair_seed)
            loss_r, grads_r = compute_gradients(
                model=model,
                interaction=regenerated_interaction,
                named_params=named_params,
                loss_mode=args.loss_mode,
            )

            grad_stats = gradient_pair_statistics(
                named_params=named_params,
                grads_o=grads_o,
                grads_r=grads_r,
                save_param_cosines=args.save_param_cosines,
            )

            row: Dict[str, Any] = {
                "user_id": pair.user_token,
                "target_mode": args.target_mode,
                "target_same_in_files": pair.target_same,
                "target_original": pair.original.target_token,
                "target_regenerated": pair.regenerated.target_token,
                "target_used_original": target_o,
                "target_used_regenerated": target_r,
                "original_length": pair.original_length,
                "regenerated_length": pair.regenerated_length,
                "length_delta": pair.regenerated_length - pair.original_length,
                "sequence_jaccard": pair.sequence_jaccard,
                "loss_original": loss_o,
                "loss_regenerated": loss_r,
                "loss_delta_regen_minus_original": loss_r - loss_o,
                **grad_stats,
            }
            results.append(row)

        except KeyError as exc:
            skipped["token_not_in_dataset_mapping"] += 1
            if skipped["token_not_in_dataset_mapping"] <= 5:
                print(f"[WARN] user={pair.user_token} token 不在 Dataset 映射中: {exc}")
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            skipped[type(exc).__name__] += 1
            if sum(skipped.values()) <= 10:
                print(f"[WARN] user={pair.user_token} skipped: {type(exc).__name__}: {exc}")

        if index % args.progress_every == 0 or index == len(pairs):
            elapsed = time.time() - started_at
            print(
                f"[{index:5d}/{len(pairs):5d}] valid={len(results):5d} "
                f"skipped={sum(skipped.values()):4d} elapsed={elapsed:.1f}s"
            )

    metadata = {
        "project_root": str(project_root),
        "target_dom": args.target_dom,
        "seed": args.seed,
        "device": str(device),
        "checkpoint_train_type": args.checkpoint_train_type,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_best_valid_score": checkpoint_meta.get("best_valid_score"),
        "dataset_train_type": dataset_train_type,
        "original_inter": str(original_path),
        "combined_inter": str(combined_path),
        "target_mode": args.target_mode,
        "loss_mode": args.loss_mode,
        "param_scope": args.param_scope,
        "param_regex": args.param_regex,
        "model_mode": args.model_mode,
        "selected_parameter_tensors": len(named_params),
        "selected_parameter_count": selected_numel,
        "sample_size_requested": args.sample_size,
        "sample_size_selected": len(pairs),
        "sample_seed": args.sample_seed,
        "elapsed_seconds": time.time() - started_at,
    }

    summary = make_summary(results, pair_stats, metadata, skipped)
    write_jsonl(jsonl_path, results)
    json_dump(summary_path, safe_json_value(summary))
    print_summary(summary)

    print(f"[DONE] Per-user JSONL : {jsonl_path}")
    print(f"[DONE] Summary JSON   : {summary_path}")
    print(f"[DONE] Pair preview   : {preview_path}")
    print(f"[DONE] Parameters     : {params_path}")

    if not results:
        print("[ERROR] 没有任何成功结果，请检查 summary 中的 skipped。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
