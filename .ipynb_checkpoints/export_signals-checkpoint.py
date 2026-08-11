#!/usr/bin/env python3
"""
信号导出脚本：捕获 Taesar 对比解码过程中的中间信号
================================================================
对训练集中每个被处理的 source item 位置，导出：
  - 位置、用户、源域、原始 item
  - action (translate / delete)
  - 映射后的 target item (if translated)
  - alpha_global, beta_global, alpha_local, beta_local
  - 全局 / 局部 top-1 item 和分数
  - 局部 margin (top1 - top2)
  - 当前 target expert 的熵

序列级聚合统计也一并导出。

用法:
  python export_signals.py target_dom=dom1 gpu_id=0
  python export_signals.py target_dom=dom1 gpu_id=0 train_batch_size=32
"""

import json
import logging
import os
import sys
import warnings

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from recbole.config import Config
from recbole.data.utils import create_samplers, get_dataloader
from recbole.utils import init_seed, set_color
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.sequential_dataset import SequentialDataset
from model.seq2seq_sasrec import SASRec
from utils import get_domain_ranges, js_divergence, wandb_start_run_with_hydra

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ============================================================================
# 模型加载 (复用 decoding.py 的逻辑)
# ============================================================================

def load_all_models(config, dataset):
    init_seed(config["seed"], config["reproducibility"])
    domains_n = int(config["domains"])
    model_names = ["full"] + [f"dom{i + 1}" for i in range(domains_n)]
    models = [SASRec(config, dataset).to(config["device"]) for _ in model_names]

    for model, ckpt_key in zip(models, [f"{name}_ckpt" for name in model_names]):
        ckpt = torch.load(config[ckpt_key], map_location=config["device"], weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.load_other_parameter(ckpt.get("other_parameter"))
        model.eval()

    logger.info(f"Loaded {len(models)} models: {model_names}")
    return dict(zip(model_names, models))


# ============================================================================
# 信号捕获核心：对每个 source item 位置计算所有中间信号
# ============================================================================

@torch.no_grad()
def capture_position_signals(
    logits_mix,       # (B, V)  — full model logits
    logits_source,    # (B, V)  — source expert logits
    logits_target,    # (B, V)  — target expert logits
    target_mask,      # (V,) bool — 哪些 index 是 target domain
    target_indices,   # (N_t,)  — target domain 对应的 logit index
    next_tokens,      # (B,)    — 当前位置的原始 item token (1-based)
    domain_name,      # str     — 当前 source domain 名称
    domain_range,     # (min, max) — 当前 source domain 的 item ID 范围
    user_ids,         # (B,)
    position,         # int     — 当前时间步
):
    """
    对 batch 中属于 source domain 的每个位置，计算并返回所有中间信号。

    返回: list of dict (每个 source position 一个 dict)
    """
    batch_size = logits_mix.size(0)
    device = logits_mix.device

    # ---- 定位 source domain 位置 ----
    src_min, src_max = domain_range
    is_source = (next_tokens >= src_min) & (next_tokens <= src_max) & (next_tokens != 0)
    if not is_source.any():
        return []

    # ---- 计算 softmax ----
    probs_M = F.softmax(logits_mix, dim=-1)
    probs_A = F.softmax(logits_target, dim=-1)   # target expert
    probs_B = F.softmax(logits_source, dim=-1)    # source expert

    # ---- 全局 alpha (基于 target expert 熵) ----
    H_A = -(probs_A * torch.log(probs_A + 1e-10)).sum(dim=-1)  # (B,)
    # 归一化: max possible entropy = log(V)
    H_A_norm = H_A / torch.log(torch.tensor(float(logits_mix.size(-1)), device=device) + 1e-10)
    # alpha = 1 - entropy (高置信 = 低熵 = 高 alpha)
    alpha_global_raw = 1.0 - H_A_norm  # (B,)
    # Taesar 原文归一化
    alpha_global_max = alpha_global_raw.max(dim=-1, keepdim=True)[0].clamp_min(1e-10)
    alpha_global = alpha_global_raw / alpha_global_max  # (B,)

    # ---- 全局 beta (基于 JSD) ----
    js_global_raw = js_divergence(probs_M, probs_B)  # (B, V) → JSD per position
    js_global = js_global_raw.sum(dim=-1)  # (B,) — sum over vocab
    js_global = js_global / js_global.clamp_min(1e-10).max(dim=-1, keepdim=True)[0]

    # ---- 全局 contrastive scores ----
    # Taesar 原文: (1+alpha)*P_T + beta*(0-P_S)
    contrastive_scores = (1.0 + alpha_global.unsqueeze(-1)) * logits_target + \
                         js_global.unsqueeze(-1) * (0 - logits_source)

    # 全局 top-1
    global_top1_idx = contrastive_scores.argmax(dim=-1)  # (B,) logit index
    global_top1_is_target = target_mask[global_top1_idx]  # (B,) bool
    global_top1_item = global_top1_idx + 1  # 1-based token

    # ---- 判断可转换性 ----
    convertible = target_mask[global_top1_idx]  # (B,)

    # ---- 局部信号 (仅可转换的位置) ----
    alpha_local = torch.zeros(batch_size, device=device)
    beta_local = torch.zeros(batch_size, device=device)
    local_top1_item = torch.zeros(batch_size, dtype=torch.long, device=device)
    local_top2_item = torch.zeros(batch_size, dtype=torch.long, device=device)
    local_top1_score = torch.zeros(batch_size, device=device)
    local_top2_score = torch.zeros(batch_size, device=device)
    local_margin = torch.zeros(batch_size, device=device)

    if convertible.any():
        rows = torch.where(convertible)[0]
        conv_logits_target = logits_target[rows][:, target_indices]
        conv_logits_source = logits_source[rows][:, target_indices]
        conv_logits_mix = logits_mix[rows][:, target_indices]

        conv_probs_A = F.softmax(conv_logits_target, dim=-1)
        conv_probs_B = F.softmax(conv_logits_source, dim=-1)
        conv_probs_M = F.softmax(conv_logits_mix, dim=-1)

        # 局部 alpha
        H_A_local = -(conv_probs_A * torch.log(conv_probs_A + 1e-10)).sum(dim=-1)
        H_A_max = torch.log(torch.tensor(float(target_indices.numel()), device=device) + 1e-10)
        alpha_local_raw = 1.0 - H_A_local / H_A_max
        alpha_local[rows] = alpha_local_raw

        # 局部 beta
        js_local_raw = js_divergence(conv_probs_M, conv_probs_B)
        js_local = js_local_raw.sum(dim=-1)
        js_local_max = js_local.max().clamp_min(1e-10)
        beta_local[rows] = js_local / js_local_max

        # 局部 adjusted scores
        adj_local = (1 + alpha_local[rows].unsqueeze(-1)) * conv_logits_target + \
                    beta_local[rows].unsqueeze(-1) * (0 - conv_logits_source)

        # 局部 top-1 / top-2
        local_topk = torch.topk(adj_local, k=min(2, adj_local.size(-1)), dim=-1)
        local_top1_logit_idx = target_indices[local_topk.indices[:, 0]]  # back to global logit idx
        local_top1_item[rows] = local_top1_logit_idx + 1
        local_top1_score[rows] = local_topk.values[:, 0]

        if adj_local.size(-1) >= 2:
            local_top2_logit_idx = target_indices[local_topk.indices[:, 1]]
            local_top2_item[rows] = local_top2_logit_idx + 1
            local_top2_score[rows] = local_topk.values[:, 1]
            local_margin[rows] = local_topk.values[:, 0] - local_topk.values[:, 1]

    # ---- 最终动作和映射结果 ----
    source_rows = torch.where(is_source)[0]
    results = []
    for i in source_rows.tolist():
        sid = next_tokens[i].item()
        uid = user_ids[i].item()
        conv = bool(convertible[i].item())

        if conv:
            action = "translate"
            mapped = int(local_top1_item[i].item())
        else:
            action = "delete"
            mapped = None

        results.append({
            "user_id": uid,
            "position": position,
            "source_domain": domain_name,
            "source_item_id": sid,
            "action": action,
            "mapped_target_item_id": mapped,
            "alpha_global": round(float(alpha_global[i].item()), 6),
            "beta_global": round(float(js_global[i].item()), 6),
            "alpha_local": round(float(alpha_local[i].item()), 6) if conv else None,
            "beta_local": round(float(beta_local[i].item()), 6) if conv else None,
            "entropy_target": round(float(H_A_norm[i].item()), 6),
            "global_top1_item": int(global_top1_item[i].item()),
            "global_top1_is_target": bool(global_top1_is_target[i].item()),
            "local_top1_item": int(local_top1_item[i].item()) if conv else None,
            "local_top2_item": int(local_top2_item[i].item()) if conv and adj_local.size(-1) >= 2 else None,
            "local_top1_score": round(float(local_top1_score[i].item()), 4) if conv else None,
            "local_top2_score": round(float(local_top2_score[i].item()), 4) if conv and adj_local.size(-1) >= 2 else None,
            "local_margin": round(float(local_margin[i].item()), 6) if conv else None,
        })

    return results


# ============================================================================
# 序列级信号聚合
# ============================================================================

def aggregate_sequence_signals(position_signals, seq_length, user_id):
    """将 per-position 信号聚合为序列级特征"""
    if not position_signals:
        return None

    sig = position_signals
    translated = [s for s in sig if s["action"] == "translate"]
    deleted = [s for s in sig if s["action"] == "delete"]

    n_translated = len(translated)
    n_deleted = len(deleted)
    n_source = n_translated + n_deleted

    feat = {
        "user_id": user_id,
        "seq_length": seq_length,
        "n_source_items": n_source,
        "n_translated": n_translated,
        "n_deleted": n_deleted,
        "regeneration_ratio": round(n_translated / max(n_source, 1), 4),
    }

    # 解码可靠性统计
    for prefix, subset in [("all", sig), ("translated", translated)]:
        if not subset:
            for stat in ["mean", "min", "std"]:
                feat[f"{prefix}_alpha_global_{stat}"] = None
                feat[f"{prefix}_beta_global_{stat}"] = None
                feat[f"{prefix}_alpha_local_{stat}"] = None
                feat[f"{prefix}_beta_local_{stat}"] = None
                feat[f"{prefix}_margin_{stat}"] = None
                feat[f"{prefix}_entropy_{stat}"] = None
            continue

        ag = np.array([s["alpha_global"] for s in subset])
        bg = np.array([s["beta_global"] for s in subset])
        al = np.array([s["alpha_local"] for s in subset if s["alpha_local"] is not None])
        bl = np.array([s["beta_local"] for s in subset if s["beta_local"] is not None])
        mg = np.array([s["local_margin"] for s in subset if s["local_margin"] is not None])
        en = np.array([s["entropy_target"] for s in subset])

        for name, arr in [("alpha_global", ag), ("beta_global", bg), ("entropy", en)]:
            feat[f"{prefix}_{name}_mean"] = round(float(arr.mean()), 6)
            feat[f"{prefix}_{name}_min"] = round(float(arr.min()), 6)
            feat[f"{prefix}_{name}_std"] = round(float(arr.std()), 6)

        for name, arr in [("alpha_local", al), ("beta_local", bl), ("margin", mg)]:
            if len(arr) > 0:
                feat[f"{prefix}_{name}_mean"] = round(float(arr.mean()), 6)
                feat[f"{prefix}_{name}_min"] = round(float(arr.min()), 6)
                feat[f"{prefix}_{name}_std"] = round(float(arr.std()), 6)
            else:
                feat[f"{prefix}_{name}_mean"] = None
                feat[f"{prefix}_{name}_min"] = None
                feat[f"{prefix}_{name}_std"] = None

    # 低置信位置比例
    for prefix, subset in [("all", sig), ("translated", translated)]:
        if not subset:
            feat[f"{prefix}_frac_low_confidence"] = None
            continue
        ag = np.array([s["alpha_global"] for s in subset])
        feat[f"{prefix}_frac_low_confidence"] = round(float((ag < 0.2).mean()), 4)

    return feat


# ============================================================================
# 主导出流程
# ============================================================================

def export_signals(config, all_models, decode_dataloader, target_domain, domain_ranges):
    device = config["device"]
    full_model = all_models["full"]
    target_model = all_models[target_domain]
    target_range = domain_ranges[target_domain]

    other_models = {k: v for k, v in all_models.items() if k not in {"full", target_domain}}
    other_ranges = {k: v for k, v in domain_ranges.items() if k != target_domain}

    position_records = []
    sequence_records = []

    for interaction in tqdm(decode_dataloader, desc="Exporting signals", ncols=80):
        interaction = interaction.to(device)

        # 获取所有模型 logits
        full_logits = full_model.calculate_logits(interaction)    # (B, L, V)
        target_logits = target_model.calculate_logits(interaction)
        source_logits_all = {name: model.calculate_logits(interaction) for name, model in other_models.items()}

        # 恢复完整序列
        item_seq = interaction["item_id_list"]
        item_seq_len = interaction["item_length"]
        pos_items = interaction["item_id"]
        padded_seq = F.pad(item_seq, (0, 1), value=0).scatter_(
            dim=1, index=item_seq_len.unsqueeze(1), src=pos_items.unsqueeze(1),
        )

        # target domain mask
        t_min, t_max = target_range
        indices = torch.arange(full_logits.size(-1), device=device)
        target_mask = (indices >= t_min) & (indices <= t_max)
        target_indices = torch.where(target_mask)[0]

        seq_len = full_logits.size(1)
        batch_size = full_logits.size(0)

        # 逐时间步捕获信号
        batch_position_signals = {b: [] for b in range(batch_size)}

        for t in range(seq_len):
            next_tokens = padded_seq[:, t + 1]

            for domain_name, src_range in other_ranges.items():
                src_min, src_max = src_range
                is_source = (next_tokens >= src_min) & (next_tokens <= src_max) & (next_tokens != 0)
                if not is_source.any():
                    continue

                sigs = capture_position_signals(
                    full_logits[:, t, :],
                    source_logits_all[domain_name][:, t, :],
                    target_logits[:, t, :],
                    target_mask, target_indices,
                    next_tokens,
                    domain_name, src_range,
                    interaction["user_id"],
                    t,
                )
                for sig in sigs:
                    batch_position_signals[sig["user_id"] % 100000 + sig["position"]] = sig
                    # use a reasonable key; just append to global
                    pass
                position_records.extend(sigs)

                # 按 user 聚合
                for sig in sigs:
                    b = (sig["user_id"] % 100000) % batch_size  # approximate batch index — we need actual mapping
                    pass

        # 更好的按 user 聚合方式：直接遍历 batch
        for b in range(batch_size):
            uid = interaction["user_id"][b].item()
            seq_len_b = item_seq_len[b].item()
            # 收集该 batch 位置的所有信号
            seq_sigs = [r for r in position_records if r.get("_batch_idx", None) == b]
            # This won't work easily; let me restructure

    # 改为在循环里直接按 user 收集
    # ... this is getting complex, let me restructure the loop

    return position_records, sequence_records


# ============================================================================
# 主函数
# ============================================================================

@hydra.main(config_path="config", config_name="overall", version_base=None)
@wandb_start_run_with_hydra
def main(config: DictConfig):
    config_obj = Config(
        model=config["model_name"],
        dataset=config["dataset"],
        config_dict=config,
    )
    target_dom = config.get("target_dom") or "dom1"
    config_obj["target_dom"] = target_dom
    config_obj["stage"] = "dec"
    device = config_obj["device"]

    logger.info(config_obj)

    # ---- 1. 数据集 ----
    logger.info("Loading dataset...")
    dataset = SequentialDataset(config_obj)
    logger.info(str(dataset))

    built = dataset.build()
    train_dataset = built[0]
    train_sampler, *_ = create_samplers(config_obj, dataset, built[:3])
    train_dataloader = get_dataloader(config_obj, "train")(
        config_obj, train_dataset, train_sampler, shuffle=config_obj["shuffle"],
    )

    domain_ranges = get_domain_ranges(config_obj["item_path"], config_obj["domains"])
    logger.info(f"Target domain: {target_dom}")

    # ---- 2. 模型 ----
    logger.info("Loading models...")
    all_models = load_all_models(config_obj, dataset)
    full_model = all_models["full"]
    target_model = all_models[target_dom]
    target_range = domain_ranges[target_dom]
    other_models = {k: v for k, v in all_models.items() if k not in {"full", target_dom}}
    other_ranges = {k: v for k, v in domain_ranges.items() if k != target_dom}

    # ---- 3. 逐时间步导出信号 ----
    logger.info("Exporting intermediate signals...")

    position_records = []
    seq_id = 0
    sequence_records = []

    for interaction in tqdm(train_dataloader, desc="Exporting", ncols=80):
        interaction = interaction.to(device)
        batch_size = interaction["user_id"].size(0)

        # 逐模型计算 logits，算完一个清一次缓存
        full_logits = full_model.calculate_logits(interaction)
        target_logits = target_model.calculate_logits(interaction)

        source_logits_all = {}
        for sname, smodel in other_models.items():
            source_logits_all[sname] = smodel.calculate_logits(interaction)
            torch.cuda.empty_cache()

        # 恢复完整序列
        item_seq = interaction["item_id_list"]
        item_seq_len = interaction["item_length"]
        pos_items = interaction["item_id"]
        padded_seq = F.pad(item_seq, (0, 1), value=0).scatter_(
            dim=1, index=item_seq_len.unsqueeze(1), src=pos_items.unsqueeze(1),
        )

        # target domain indices
        t_min, t_max = target_range
        all_indices = torch.arange(full_logits.size(-1), device=device)
        target_mask = (all_indices >= t_min) & (all_indices <= t_max)
        target_indices = torch.where(target_mask)[0]

        seq_len = full_logits.size(1)

        # 为每个 batch 元素收集
        for b in range(batch_size):
            uid = interaction["user_id"][b].item()
            seq_item_list = padded_seq[b].tolist()
            actual_len = item_seq_len[b].item() + 1  # +1 for the target item

            seq_sigs = []

            for t in range(seq_len):
                # 下一个 token
                nxt = padded_seq[b, t + 1].item()
                if nxt == 0:
                    continue

                # 检查是否属于 source domain
                matched_domain = None
                matched_range = None
                for dname, (smin, smax) in other_ranges.items():
                    if smin <= nxt <= smax:
                        matched_domain = dname
                        matched_range = (smin, smax)
                        break

                if matched_domain is None:
                    continue  # target domain item or padding, skip

                # 提取该位置各模型的单行 logits
                fm = full_logits[b, t, :].unsqueeze(0)      # (1, V)
                ts = target_logits[b, t, :].unsqueeze(0)
                ss = source_logits_all[matched_domain][b, t, :].unsqueeze(0)
                nt = torch.tensor([nxt], device=device)
                uids = torch.tensor([uid], device=device)

                sigs = capture_position_signals(
                    fm, ss, ts, target_mask, target_indices,
                    nt, matched_domain, matched_range, uids, t,
                )
                for sig in sigs:
                    sig["sequence_id"] = seq_id
                seq_sigs.extend(sigs)
                position_records.extend(sigs)

            # 序列级聚合
            seq_feat = aggregate_sequence_signals(seq_sigs, actual_len, uid)
            if seq_feat:
                seq_feat["sequence_id"] = seq_id
                seq_feat["original_items"] = seq_item_list[:actual_len]
                # 再生后 item 列表 (根据 capture 结果重建)
                regenerated = seq_item_list[:actual_len].copy()
                for sig in seq_sigs:
                    pos = sig["position"] + 1  # +1 because position 0 = first next-token
                    if pos < len(regenerated):
                        if sig["action"] == "translate":
                            regenerated[pos] = sig["mapped_target_item_id"]
                        elif sig["action"] == "delete":
                            regenerated[pos] = sig["source_item_id"]  # keep original, caller decides
                seq_feat["regenerated_items"] = regenerated
                sequence_records.append(seq_feat)

            seq_id += 1

    # ---- 4. 保存 ----
    save_dir = os.path.join("save", "signals", target_dom)
    os.makedirs(save_dir, exist_ok=True)

    # Per-position signals → JSONL
    pos_path = os.path.join(save_dir, "position_signals.jsonl")
    with open(pos_path, "w") as f:
        for rec in position_records:
            f.write(json.dumps(rec, default=str) + "\n")
    logger.info(f"Exported {len(position_records)} position records → {pos_path}")

    # Sequence-level → JSON
    seq_path = os.path.join(save_dir, "sequence_signals.json")
    with open(seq_path, "w") as f:
        json.dump(sequence_records, f, default=str)
    logger.info(f"Exported {len(sequence_records)} sequence records → {seq_path}")

    # 快速统计
    actions = [r["action"] for r in position_records]
    n_translate = actions.count("translate")
    n_delete = actions.count("delete")
    logger.info(f"Action breakdown: translate={n_translate}, delete={n_delete} "
                f"(ratio translate={n_translate/len(actions)*100:.1f}%)")

    logger.info("Done.")


if __name__ == "__main__":
    main()