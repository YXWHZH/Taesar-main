#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, subprocess, sys
from pathlib import Path

DOMAINS = ["dom1","dom2","dom3","dom4"]

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get(d, *ks, default=None):
    x=d
    for k in ks:
        if not isinstance(x, dict) or k not in x:
            return default
        x=x[k]
    return x

def pct(x):
    return "NA" if x is None else f"{100*x:.2f}%"

def num(x, n=5):
    return "NA" if x is None else f"{x:.{n}f}"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--targets", nargs="+", default=["dom2","dom3","dom4"], choices=DOMAINS)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--encode-batch-size", type=int, default=1024)
    p.add_argument("--max-train-users", type=int, default=20000)
    p.add_argument("--max-valid-users", type=int, default=5000)
    p.add_argument("--max-audit-users", type=int, default=0)
    p.add_argument("--audit-split", default="test")
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only-summarize", action="store_true")
    return p.parse_args()

def summary_path(root, target, split):
    return root/"save"/"multisource_utility"/target/f"utility_summary_{split}.json"

def run_target(root, args, target):
    sp=summary_path(root,target,args.audit_split)
    if sp.exists() and not args.force:
        print(f"[SKIP] {target}: existing summary {sp}")
        return
    audit=root/"audit_multisource_utility.py"
    if not audit.exists():
        raise FileNotFoundError(audit)
    cmd=[
        sys.executable,str(audit),
        "--target-dom",target,
        "--seed",str(args.seed),
        "--gpu-id",str(args.gpu_id),
        "--epochs",str(args.epochs),
        "--patience",str(args.patience),
        "--batch-size",str(args.batch_size),
        "--encode-batch-size",str(args.encode_batch_size),
        "--max-train-users",str(args.max_train_users),
        "--max-valid-users",str(args.max_valid_users),
        "--max-audit-users",str(args.max_audit_users),
        "--audit-split",args.audit_split
    ]
    if args.reuse_cache:
        cmd.append("--reuse-cache")
    print("\n"+"="*90)
    print("[RUN]",target)
    print("="*90)
    subprocess.run(cmd,cwd=root,check=True)

def aggregate(root, split):
    rows=[]
    for target in DOMAINS:
        p=summary_path(root,target,split)
        if not p.exists():
            print("[WARN] missing",target,p)
            continue
        s=load_json(p)
        src=s.get("source_domains",[])
        util=s.get("utility_by_source",{})
        best=s.get("best_source_distribution",{})
        neg=[util[d]["negative_rate"] for d in src if d in util]
        means=[util[d]["mean"] for d in src if d in util]
        bestf=[best[d]["fraction"] for d in src if d in best]
        dense=get(s,"ranking_metrics","dense_all",default={}) or {}
        safe2=get(s,"ranking_metrics","safe_top2_by_CE",default={}) or {}
        oracle1=get(s,"ranking_metrics","oracle_top1_source_by_CE",default={}) or {}
        r={
            "target":target,
            "n_users":s.get("n_users"),
            "avg_negative_rate":sum(neg)/len(neg) if neg else None,
            "min_negative_rate":min(neg) if neg else None,
            "max_negative_rate":max(neg) if neg else None,
            "avg_mean_utility":sum(means)/len(means) if means else None,
            "max_best_source_fraction":max(bestf) if bestf else None,
            "best_source_not_global_majority_rate":get(s,"heterogeneity","best_source_not_global_majority_rate"),
            "no_source_beats_all_singles_rate":get(s,"heterogeneity","no_source_beats_all_single_sources_rate"),
            "dense_minus_oracle_top1_ce":get(s,"loss","dense_minus_oracle_top1"),
            "dense_minus_safe_top2_ce":get(s,"loss","dense_minus_safe_top2"),
            "oracle_top1_beats_dense_rate":get(s,"selection_advantage","oracle_top1_beats_dense_rate"),
            "safe_top2_beats_dense_rate":get(s,"selection_advantage","safe_top2_beats_dense_rate"),
            "dense_recall10":dense.get("Recall@10"),
            "safe2_recall10":safe2.get("Recall@10"),
            "oracle1_recall10":oracle1.get("Recall@10"),
            "dense_ndcg10":dense.get("NDCG@10"),
            "safe2_ndcg10":safe2.get("NDCG@10"),
            "oracle1_ndcg10":oracle1.get("NDCG@10"),
            "dense_mrr":dense.get("MRR"),
            "safe2_mrr":safe2.get("MRR"),
            "oracle1_mrr":oracle1.get("MRR")
        }
        for m in ["recall10","ndcg10","mrr"]:
            d=r.get("dense_"+m); q=r.get("safe2_"+m); o=r.get("oracle1_"+m)
            r["safe2_minus_dense_"+m]=None if d is None or q is None else q-d
            r["oracle1_minus_dense_"+m]=None if d is None or o is None else o-d
        rows.append(r)
    return rows

def decide(rows):
    checks=[]
    for r in rows:
        neg = r["avg_negative_rate"] is not None and r["avg_negative_rate"] >= .20
        bal = r["max_best_source_fraction"] is not None and r["max_best_source_fraction"] <= .70
        ceg = r["dense_minus_oracle_top1_ce"] is not None and r["dense_minus_oracle_top1_ce"] > 0
        rate = r["oracle_top1_beats_dense_rate"] is not None and r["oracle_top1_beats_dense_rate"] >= .60
        rankvals=[r.get("safe2_minus_dense_recall10"),r.get("safe2_minus_dense_ndcg10"),
                  r.get("oracle1_minus_dense_recall10"),r.get("oracle1_minus_dense_ndcg10")]
        rank=any(x is not None and x>0 for x in rankvals)
        checks.append({"target":r["target"],"negative_utility":neg,"best_source_not_collapsed":bal,
                       "oracle_ce_gap":ceg,"oracle_beats_dense_rate":rate,"ranking_gap":rank,
                       "passed_checks":sum([neg,bal,ceg,rate,rank])})
    strong=sum(c["passed_checks"]>=4 for c in checks)
    core=sum(c["negative_utility"] and c["best_source_not_collapsed"] and c["oracle_ce_gap"] for c in checks)
    n=len(rows)
    if n>=4 and strong>=3 and core>=3:
        status="STRONG_GO"
        reason="至少3/4目标域同时支持负效用异质性、非单域垄断与稀疏选择优势。"
    elif strong>=max(2,n//2):
        status="GO_WITH_CAUTION"
        reason="多个目标域支持该方向，但跨域一致性或推荐指标优势仍需增强。"
    else:
        status="STOP_OR_REDESIGN"
        reason="当前跨目标域证据不足，不建议立即进入正式Router。"
    return {"status":status,"reason":reason,"n_domains_available":n,
            "strong_domains":strong,"core_domains":core,"per_target_checks":checks}

def write_outputs(root, rows, decision):
    out=root/"save"/"multisource_utility"
    out.mkdir(parents=True,exist_ok=True)
    cp=out/"stage1_multidomain_summary.csv"
    if rows:
        with open(cp,"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    jp=out/"stage1_multidomain_summary.json"
    save_json(jp,{"rows":rows,"decision":decision})
    mp=out/"stage1_decision.md"
    lines=["# Stage-1 Multi-domain Utility Decision","",f"**Decision: {decision['status']}**","",decision["reason"],"",
           "|Target|Avg negative utility|Max best-source share|Dense-Oracle1 CE|Oracle1 beats Dense|Safe2-Dense R@10|Safe2-Dense NDCG@10|",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['target']}|{pct(r['avg_negative_rate'])}|{pct(r['max_best_source_fraction'])}|"
                     f"{num(r['dense_minus_oracle_top1_ce'],6)}|{pct(r['oracle_top1_beats_dense_rate'])}|"
                     f"{num(r['safe2_minus_dense_recall10'],6)}|{num(r['safe2_minus_dense_ndcg10'],6)}|")
    lines += ["","判据：负效用率≥20%；最大Best-source占比≤70%；Dense-Oracle Top1 CE gap>0；"
              "Oracle Top1在≥60%用户上优于Dense；Recall/NDCG至少有一个正向Oracle/Safe gap。"]
    mp.write_text("\n".join(lines),encoding="utf-8")
    return cp,jp,mp

def print_table(rows,decision):
    print("\n"+"="*112)
    print("Stage-1 cross-target summary")
    print("="*112)
    print(f"{'Target':<8}{'NegRate':>11}{'MaxBest':>11}{'D-O1 CE':>12}{'O1>Dense':>12}{'S2-D R10':>12}{'S2-D N10':>12}")
    for r in rows:
        print(f"{r['target']:<8}{pct(r['avg_negative_rate']):>11}{pct(r['max_best_source_fraction']):>11}"
              f"{num(r['dense_minus_oracle_top1_ce']):>12}{pct(r['oracle_top1_beats_dense_rate']):>12}"
              f"{num(r['safe2_minus_dense_recall10']):>12}{num(r['safe2_minus_dense_ndcg10']):>12}")
    print("-"*112)
    print("DECISION:",decision["status"])
    print(decision["reason"])
    print("="*112)

def main():
    args=parse_args()
    root=Path(args.project_root).resolve()
    if not args.only_summarize:
        for t in args.targets:
            run_target(root,args,t)
    rows=aggregate(root,args.audit_split)
    decision=decide(rows)
    cp,jp,mp=write_outputs(root,rows,decision)
    print_table(rows,decision)
    print("\n[DONE]",cp)
    print("[DONE]",jp)
    print("[DONE]",mp)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
