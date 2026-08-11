#!/usr/bin/env python3
"""End-to-end soft candidate gate with OOF pairwise utility as auxiliary supervision."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from audit_multisource_utility import ProbeDataset
from stage2_train_utility_router import load_stage1,subset,save_json
from stage4_pairwise_candidate_reranker import folds_load,fold_ids,make_pair_cache,mask_logits,paired_ndcg_significance,rank_values


def parse_args():
 p=argparse.ArgumentParser();p.add_argument("--target-dom",choices=("dom1","dom2"),required=True)
 p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=2025)
 p.add_argument("--candidate-m",type=int,default=5000);p.add_argument("--hard-k",type=int,default=10)
 p.add_argument("--neg-strata",type=int,nargs=3,default=(5,3,2));p.add_argument("--batch-size",type=int,default=256)
 p.add_argument("--score-batch-size",type=int,default=32);p.add_argument("--candidate-chunk",type=int,default=512)
 p.add_argument("--epochs",type=int,default=25);p.add_argument("--patience",type=int,default=5)
 p.add_argument("--hidden",type=int,default=128);p.add_argument("--lr",type=float,default=1e-3)
 p.add_argument("--weight-decay",type=float,default=1e-4);p.add_argument("--utility-weight",type=float,default=.5)
 p.add_argument("--reuse-checkpoints",action="store_true")
 p.add_argument("--output-dir",default=None);return p.parse_args()


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


class PairUsers(Dataset):
 def __init__(self,cache):self.c=cache
 def __len__(self):return len(self.c["neg"])
 def __getitem__(self,u):
  c=self.c
  return u,c["neg"][u],c["pair"][u],c["base_y"][u],c["base_j"][u],c["uy"][u],c["uj"][u]


class SoftGate(nn.Module):
 def __init__(self,kind,h,s,hidden):
  super().__init__();self.kind=kind;self.s=s;dim=h+s if kind=="P1" else 5*h+s
  self.net=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(.1),nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(.1),nn.Linear(hidden,1))
 def forward(self,ht,hs,e):
  b,c,h=e.shape;out=[]
  for d in range(self.s):
   one=F.one_hot(torch.full((b,c),d,device=e.device),self.s).float()
   if self.kind=="P1":x=torch.cat([e,one],2)
   else:
    hd=hs[:,d,None,:].expand(-1,c,-1);tt=ht[:,None,:].expand(-1,c,-1)
    x=torch.cat([tt,hd,e,tt*e,hd*e,one],2)
   out.append(self.net(x).squeeze(2))
  return torch.stack(out,1)


def mixed_scores(q,base,correction):
 alpha=F.softmax(torch.cat([torch.zeros_like(q[:,:1]),q],1),1)
 return base+(alpha[:,1:]*correction).sum(1),alpha


def batch_loss(model,batch,data,item,device,utility_weight):
 u,neg,pair,by,bj,uy,uj=[x.to(device) for x in batch];ix=u.cpu().numpy()
 ht=torch.from_numpy(data.h_target[ix]).to(device);hs=torch.from_numpy(data.h_sources[ix]).to(device)
 y=torch.as_tensor(data.y_class[ix],device=device);ids=torch.cat([y[:,None],neg],1);e=item[ids]
 base=torch.cat([by[:,None],bj],1).float();corr=torch.cat([uy[:,:,None],uj],2).float();q=model(ht,hs,e)
 score,_=mixed_scores(q,base,corr);rank=F.cross_entropy(score,torch.zeros(len(u),device=device,dtype=torch.long))
 utility=F.huber_loss(q[:,:,0,None]-q[:,:,1:],pair.float())
 return rank+utility_weight*utility,rank,utility


def train_model(name,kind,utility_weight,args,data,item,train_c,val_c,device,out):
 model=SoftGate(kind,item.shape[1],train_c["pair"].shape[1],args.hidden).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 train=DataLoader(PairUsers(train_c),batch_size=args.batch_size,shuffle=True);valid=DataLoader(PairUsers(val_c),batch_size=1024)
 best=float("inf");bad=0;path=out/f"{name}_best.pt";history=[]
 for epoch in range(1,args.epochs+1):
  model.train();tr=[]
  for batch in train:
   loss,rank,utility=batch_loss(model,batch,data["train"],item,device,utility_weight);opt.zero_grad();loss.backward();opt.step();tr.append((loss.item(),rank.item(),utility.item()))
  model.eval();vals=[]
  with torch.no_grad():
   for batch in valid:vals.append(tuple(x.item() for x in batch_loss(model,batch,data["valid"],item,device,utility_weight)))
  va=np.mean(vals,0);row={"epoch":epoch,"train_total":float(np.mean(tr,0)[0]),"valid_total":float(va[0]),"valid_rank":float(va[1]),"valid_utility":float(va[2])};history.append(row);print(f"[{name} {epoch}] rank={va[1]:.6f} utility={va[2]:.6f}")
  if va[1]<best-1e-7:best=float(va[1]);bad=0;torch.save(model.state_dict(),path)
  else:
   bad+=1
   if bad>=args.patience:break
 model.load_state_dict(torch.load(path,map_location=device,weights_only=True));return model,history


def metrics(ranks):
 r=np.asarray(ranks);covered=r>0;hit=covered&(r<=10);rc=r[covered];ndcg=np.zeros(len(r));ndcg[hit]=1/np.log2(r[hit]+1);mrr=np.zeros(len(r));mrr[covered]=1/r[covered]
 return {"Recall@5000":float(covered.mean()),"positive_users":int(covered.sum()),"Recall@10":float(hit.mean()),"NDCG@10":float(ndcg.mean()),"MRR":float(mrr.mean()),"conditional_Recall@10":float((rc<=10).mean()) if len(rc) else 0.0,"conditional_NDCG@10":float(ndcg[covered].mean()) if len(rc) else 0.0,"conditional_MRR":float((1/rc).mean()) if len(rc) else 0.0,"RerankSuccess@10":float((rc<=10).mean()) if len(rc) else 0.0}


@torch.no_grad()
def evaluate(data,probes,models,item,args,device,sources):
 names=["Target","Dense","Oracle",*models];ranks={x:[] for x in names}
 for ht,hs,av,y,_ in DataLoader(ProbeDataset(data),batch_size=args.score_batch_size):
  ht,hs,av,y=[x.to(device) for x in (ht,hs,av,y)];b=len(y);z=mask_logits(probes,ht,hs,av);m=min(args.candidate_m,z[0].shape[1]);cand=torch.topk(z[0],m,1).indices
  ss=torch.stack([x.gather(1,cand) for x in z]);base=ss[0];single=ss[1:1+len(sources)];cc=cand.cpu().numpy();yy=y.cpu().numpy()
  ranks["Target"].extend(rank_values(base.cpu().numpy(),cc,yy).tolist());ranks["Dense"].extend(rank_values(ss[-1].cpu().numpy(),cc,yy).tolist())
  views=np.stack([rank_values(x.cpu().numpy(),cc,yy) for x in torch.cat([base[None],single],0)]);ok=views>0;oracle=np.where(ok.any(0),np.where(ok,views,np.iinfo(np.int64).max).min(0),0);ranks["Oracle"].extend(oracle.tolist())
  corr=single.permute(1,0,2)-base[:,None,:]
  for name,model in models.items():
   parts=[]
   for st in range(0,m,args.candidate_chunk):
    ids=cand[:,st:st+args.candidate_chunk];q=model(ht,hs,item[ids]);score,_=mixed_scores(q,base[:,st:st+args.candidate_chunk],corr[:,:,st:st+args.candidate_chunk]);parts.append(score)
   ranks[name].extend(rank_values(torch.cat(parts,1).cpu().numpy(),cc,yy).tolist())
 return {k:metrics(v) for k,v in ranks.items()},ranks


def main():
 args=parse_args();seed_all(args.seed)
 if args.candidate_m!=5000:raise ValueError("Protocol is frozen: candidate-m must remain 5000")
 root=Path(__file__).resolve().parent;device=torch.device(args.device);out=Path(args.output_dir).resolve() if args.output_dir else root/"save/soft_candidate_gate"/args.target_dom;out.mkdir(parents=True,exist_ok=True)
 base,splits,sources=load_stage1(root/"save/multisource_utility"/args.target_dom,args.target_dom,device);splits["train"]=subset(splits["train"],20000,args.seed);splits["valid"]=subset(splits["valid"],5000,args.seed+1)
 probes=folds_load(root,args.target_dom,base,sources,device);train_c=make_pair_cache(splits["train"],(fold_ids(len(splits["train"].users),args.seed),probes),device,args.score_batch_size,args.hard_k,args.neg_strata,args.seed+11);val_c=make_pair_cache(splits["valid"],probes,device,args.score_batch_size,args.hard_k,args.neg_strata,args.seed+12)
 item=base.target_item_embeddings.detach().to(device);data={"train":splits["train"],"valid":splits["valid"]};models={};history={}
 specs=(("P1-soft","P1",0.0),("P5-soft","P5",0.0),("P5-soft+utility","P5",args.utility_weight))
 for name,kind,weight in specs:
  path=out/f"{name}_best.pt"
  if args.reuse_checkpoints and path.exists():
   model=SoftGate(kind,item.shape[1],train_c["pair"].shape[1],args.hidden).to(device);model.load_state_dict(torch.load(path,map_location=device,weights_only=True));models[name]=model;history[name]=[{"reused_checkpoint":str(path)}]
  else:models[name],history[name]=train_model(name,kind,weight,args,data,item,train_c,val_c,device,out)
 valid,_=evaluate(splits["valid"],probes,models,item,args,device,sources);test,ranks=evaluate(splits["test"],probes,models,item,args,device,sources)
 significance=paired_ndcg_significance(ranks,"P5-soft",["Target","P1-soft","Dense"],args.seed+99);significance_aux=paired_ndcg_significance(ranks,"P5-soft+utility",["P5-soft","Target","P1-soft"],args.seed+199)
 report={"protocol":{"candidate_pool":"Target-only Top-5000 frozen from validation coverage","experts":"frozen OOF probe ensemble","hard_negatives":{"total":args.hard_k,"strata":list(args.neg_strata)},"integration":"s0 + sum_d softmax([no-transfer, source gates])_d * (sd-s0)","primary_loss":"sampled softmax ranking","utility_auxiliary":"OOF pairwise Huber","test":"single frozen evaluation"},"args":vars(args),"sources":sources,"history":history,"valid":valid,"test":test,"paired_significance":{"P5-soft":significance,"P5-soft+utility":significance_aux}}
 save_json(out/"soft_gate_report.json",report);print(json.dumps({"valid":valid,"test":test,"significance":report["paired_significance"]},indent=2));print(f"[REPORT] {out/'soft_gate_report.json'}")


if __name__=="__main__":main()
