#!/usr/bin/env python3
"""Validation-only candidate coverage sweep with OOF-probe logit ensembles."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from audit_multisource_utility import MaskFusionProbe, ProbeDataset
from stage2_train_utility_router import load_stage1, save_json, subset, torch_load

TARGET_MS = (100, 500, 1000, 2000, 5000)
UNION_KS = (20, 100, 200, 400, 1000)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-dom", choices=("dom1", "dom2"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-valid-users", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def load_fold_probes(root, target, base, sources, device):
    probes=[]; item=base.target_item_embeddings.detach()
    for fold in range(5):
        ckpt=torch_load(root/"save/oof_candidate_utility"/target/"teachers"/f"fold_{fold}"/"probe_best.pt",device)
        model=MaskFusionProbe(item.shape[1],len(sources),item,dropout=0.1,
                              temperature=base.temperature).to(device)
        model.load_state_dict(ckpt["state_dict"],strict=True); model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        probes.append(model)
    return probes


@torch.no_grad()
def main():
    args=parse_args(); root=Path(__file__).resolve().parent; device=torch.device(args.device)
    base,splits,sources=load_stage1(root/"save/multisource_utility"/args.target_dom,
                                    args.target_dom,device)
    valid=subset(splits["valid"],args.max_valid_users,args.seed+1)
    probes=load_fold_probes(root,args.target_dom,base,sources,device)
    s=len(sources)
    masks=[torch.zeros(s)]
    masks += [torch.eye(s)[i] for i in range(s)]
    masks += [torch.ones(s)]
    max_target=min(max(TARGET_MS),base.target_item_embeddings.shape[0])
    max_union=min(max(UNION_KS),base.target_item_embeddings.shape[0])
    target_hits={m:0 for m in TARGET_MS}
    union_hits={k:0 for k in UNION_KS}
    matched_target_hits={k:0 for k in UNION_KS}
    union_sizes={k:[] for k in UNION_KS}
    n=0
    loader=DataLoader(ProbeDataset(valid),batch_size=args.batch_size,shuffle=False)
    for ht,hs,avail,y,_ in loader:
        ht,hs,avail,y=ht.to(device),hs.to(device),avail.to(device),y.to(device)
        avg=[]
        for m0 in masks:
            acc=None; m=m0.to(device).expand(len(y),-1)
            for probe in probes:
                z=probe(ht,hs,m,avail)
                acc=z if acc is None else acc+z
            avg.append(acc/len(probes))
        tops=[torch.topk(z,max_union,dim=1).indices.cpu().numpy() for z in avg]
        target_top=torch.topk(avg[0],max_target,dim=1).indices.cpu().numpy()
        yy=y.cpu().numpy()
        for m in TARGET_MS:
            mm=min(m,target_top.shape[1])
            target_hits[m]+=int(np.any(target_top[:,:mm]==yy[:,None],axis=1).sum())
        for i,yi in enumerate(yy):
            for k in UNION_KS:
                kk=min(k,max_union)
                pool=np.unique(np.concatenate([x[i,:kk] for x in tops]))
                union_sizes[k].append(len(pool))
                matched_target_hits[k]+=int(np.any(target_top[i,:len(pool)]==yi))
                union_hits[k]+=int(np.any(pool==yi))
        n+=len(y)
        print(f"[{args.target_dom}] {n}/{len(valid.users)}")
    report={"protocol":{
        "split":"valid only","logits":"mean of five OOF fold probes",
        "views":["000",*[f"single_{x}" for x in sources],"111"],
        "selection":"No test inspection; choose candidate policy from this report."
    },"target":args.target_dom,"n_users":n,
      "target_only":{str(m):{"nominal_budget":m,"coverage":target_hits[m]/n,
                             "positive_users":target_hits[m]} for m in TARGET_MS},
      "multi_view_union":{}}
    for k in UNION_KS:
        a=np.asarray(union_sizes[k])
        report["multi_view_union"][str(k)]={
          "per_view_k":k,"max_budget":5*k,"mean_pool_size":float(a.mean()),
          "median_pool_size":float(np.median(a)),"p10_pool_size":float(np.quantile(a,.1)),
          "p90_pool_size":float(np.quantile(a,.9)),"coverage":union_hits[k]/n,
          "matched_target_coverage":matched_target_hits[k]/n,
          "positive_users":union_hits[k]}
    out=Path(args.output_dir).resolve() if args.output_dir else root/"save/retrieval_coverage"/args.target_dom
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/"valid_coverage.json",report)
    print(report)
    print(f"[REPORT] {out/'valid_coverage.json'}")


if __name__=="__main__": main()
