#!/usr/bin/env python3
"""OOF pairwise hard-negative candidate routing and Top-M reranking."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from audit_multisource_utility import MaskFusionProbe,ProbeDataset
from stage2_train_utility_router import load_stage1,subset,save_json,torch_load
from audit_oof_candidate_utility import take

def args_parse():
 p=argparse.ArgumentParser(); p.add_argument("--target-dom",choices=("dom1","dom2"),required=True)
 p.add_argument("--device",default="cuda:0"); p.add_argument("--seed",type=int,default=2025)
 p.add_argument("--hard-k",type=int,default=10); p.add_argument("--neg-strata",type=int,nargs=3,default=(5,3,2),metavar=("TOP100","MID500","TAIL5000"))
 p.add_argument("--candidate-m",type=int,default=5000); p.add_argument("--batch-size",type=int,default=256)
 p.add_argument("--score-batch-size",type=int,default=64); p.add_argument("--epochs",type=int,default=25)
 p.add_argument("--patience",type=int,default=5); p.add_argument("--hidden",type=int,default=128)
 p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
 p.add_argument("--pair-weight",type=float,default=1.0); p.add_argument("--point-weight",type=float,default=0.5)
 p.add_argument("--output-dir",default=None); return p.parse_args()
def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def folds_load(root,target,base,sources,device):
 out=[]; item=base.target_item_embeddings.detach()
 for f in range(5):
  z=torch_load(root/"save/oof_candidate_utility"/target/"teachers"/f"fold_{f}"/"probe_best.pt",device)
  p=MaskFusionProbe(item.shape[1],len(sources),item,.1,base.temperature).to(device); p.load_state_dict(z["state_dict"]); p.eval()
  for q in p.parameters(): q.requires_grad_(False)
  out.append(p)
 return out
@torch.no_grad()
def mask_logits(probes,ht,hs,av):
 s=hs.shape[1]; masks=[torch.zeros(s,device=ht.device)]+[torch.eye(s,device=ht.device)[i] for i in range(s)]+[torch.ones(s,device=ht.device)]; out=[]
 for m0 in masks:
  m=m0.expand(len(ht),-1); z=None
  for p in probes:
   q=p(ht,hs,m,av); z=q if z is None else z+q
  out.append(z/len(probes))
 return out
def fold_ids(n,seed):
 order=np.random.default_rng(seed+700).permutation(n); ids=np.empty(n,np.int64)
 for f,x in enumerate(np.array_split(order,5)): ids[x]=f
 return ids
def stratified_negatives(order,y,k,counts,rng):
 bounds=((0,min(100,len(order))),(min(100,len(order)),min(500,len(order))),(min(500,len(order)),min(5000,len(order))))
 chosen=[]
 for (lo,hi),need in zip(bounds,counts):
  pool=order[lo:hi];pool=pool[pool!=y]
  if len(pool):chosen.extend(rng.choice(pool,size=min(need,len(pool)),replace=False).tolist())
 if len(chosen)<k:
  used=set(chosen);pool=np.asarray([x for x in order[:min(5000,len(order))] if x!=y and x not in used])
  if len(pool):chosen.extend(rng.choice(pool,size=min(k-len(chosen),len(pool)),replace=False).tolist())
 if len(chosen)!=k:raise ValueError(f"Cannot sample {k} negatives from {len(order)} ranked items")
 return chosen
@torch.no_grad()
def make_pair_cache(data,teacher_spec,device,batch,k,counts,seed):
 if sum(counts)!=k:raise ValueError(f"neg-strata must sum to hard-k ({counts} vs {k})")
 rng=np.random.default_rng(seed)
 n=len(data.users); s=data.h_sources.shape[1]; neg=np.empty((n,k),np.int64); uy=np.empty((n,s),np.float32); uj=np.empty((n,s,k),np.float32)
 if isinstance(teacher_spec,tuple):
  ids,all_probes=teacher_spec; groups=[(np.flatnonzero(ids==f),[all_probes[f]]) for f in range(5)]
 else: groups=[(np.arange(n),teacher_spec)]
 for idx,probes in groups:
  for st in range(0,len(idx),batch):
   ii=idx[st:st+batch]; d=take(data,ii); ht,hs,av,y,_=next(iter(DataLoader(ProbeDataset(d),batch_size=len(ii))))
   ht,hs,av,y=[x.to(device) for x in (ht,hs,av,y)]; z=mask_logits(probes,ht,hs,av); top=torch.topk(z[0],min(5001,z[0].shape[1]),1).indices.cpu().numpy();yn=y.cpu().numpy()
   jj=torch.as_tensor(np.asarray([stratified_negatives(top[r],yn[r],k,counts,rng) for r in range(len(y))]),device=device); base_y=z[0].gather(1,y[:,None]); base_j=z[0].gather(1,jj)
   for q in range(s):
    uy[ii,q]=(z[1+q].gather(1,y[:,None])-base_y).squeeze(1).cpu(); uj[ii,q]=(z[1+q].gather(1,jj)-base_j).cpu()
   neg[ii]=jj.cpu()
 return {"neg":neg,"uy":uy,"uj":uj,"pair":uy[:,:,None]-uj}
class Pairs(Dataset):
 def __init__(self,c): self.c=c; self.n,self.s,self.k=c["pair"].shape
 def __len__(self): return self.n*self.s*self.k
 def __getitem__(self,x):
  u=x//(self.s*self.k); r=x%(self.s*self.k); d=r//self.k; q=r%self.k
  return u,d,self.c["neg"][u,q],self.c["pair"][u,d,q],self.c["uy"][u,d],self.c["uj"][u,d,q]
class Gate(nn.Module):
 def __init__(self,kind,h,s):
  super().__init__(); self.kind=kind; self.s=s; dim=h+s if kind=="P1" else 5*h+s
  self.net=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,1))
 def forward(self,ht,hs,e,d):
  one=F.one_hot(d,self.s).float(); sd=hs[torch.arange(len(d),device=d.device),d] if hs.ndim==3 else hs
  x=torch.cat([e,one],1) if self.kind=="P1" else torch.cat([ht,sd,e,ht*e,sd*e,one],1)
  return self.net(x).squeeze(1)
def train_gate(kind,args,data,item,train_c,val_c,device,out):
 model=Gate(kind,item.shape[1],train_c["pair"].shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 tr=DataLoader(Pairs(train_c),batch_size=args.batch_size,shuffle=True); va=DataLoader(Pairs(val_c),batch_size=1024); best=1e9;bad=0;path=out/f"{kind}_pair_best.pt"
 for ep in range(1,args.epochs+1):
  model.train()
  for u,d,j,p,uy,uj in tr:
   u,d,j,p,uy,uj=[x.to(device) for x in (u,d,j,p,uy,uj)]; ix=u.cpu().numpy()
   ht=torch.from_numpy(data["train"].h_target[ix]).to(device); hs=torch.from_numpy(data["train"].h_sources[ix]).to(device); ey=item[torch.as_tensor(data["train"].y_class[ix],device=device)]; ej=item[j]
   gy=model(ht,hs,ey,d); gj=model(ht,hs,ej,d); loss=args.pair_weight*F.huber_loss(gy-gj,p.float())+args.point_weight*(F.huber_loss(gy,uy.float())+F.huber_loss(gj,uj.float()))
   opt.zero_grad();loss.backward();opt.step()
  model.eval();total=count=0
  with torch.no_grad():
   for u,d,j,p,uy,uj in va:
    u,d,j,p,uy,uj=[x.to(device) for x in (u,d,j,p,uy,uj)]; ix=u.cpu().numpy(); ht=torch.from_numpy(data["valid"].h_target[ix]).to(device);hs=torch.from_numpy(data["valid"].h_sources[ix]).to(device)
    gy=model(ht,hs,item[torch.as_tensor(data["valid"].y_class[ix],device=device)],d);gj=model(ht,hs,item[j],d);l=F.huber_loss(gy-gj,p.float())+.5*(F.huber_loss(gy,uy.float())+F.huber_loss(gj,uj.float()));total+=l.item()*len(u);count+=len(u)
  v=total/count;print(f"[{kind} {ep}] valid={v:.6f}")
  if v<best-1e-7:best=v;bad=0;torch.save(model.state_dict(),path)
  else:
   bad+=1
   if bad>=args.patience:break
 model.load_state_dict(torch.load(path,map_location=device,weights_only=True));return model,best
def read_user_choice(path,users,sources):
 rows=list(csv.DictReader(path.open())); by_user={r["user_id"]:r for r in rows}
 assert all(str(x) in by_user for x in users)
 return np.array([[float(by_user[str(x)][f"pred_utility_{s}"]) for s in sources] for x in users]).argmax(1)
def rank_values(scores,cand,y):
 out=[]
 for i in range(len(y)):
  pos=np.flatnonzero(cand[i]==y[i])
  out.append(0 if not len(pos) else 1+int((scores[i]>scores[i,pos[0]]).sum()))
 return np.asarray(out)
def add_rank(acc,name,scores,cand,y):
 acc[name].extend(rank_values(scores,cand,y).tolist())
@torch.no_grad()
def evaluate(data,probes,models,item,args,device,sources,user_choice,thresholds):
 names=["target","dense","oracle","user"]+[f"{k}:{t:.6g}" for k in models for t in thresholds]+[f"fixed{d}" for d in range(len(sources))];acc={x:[] for x in names};off=0
 for ht,hs,av,y,_ in DataLoader(ProbeDataset(data),batch_size=args.score_batch_size):
  b=len(y);ht,hs,av,y=[x.to(device) for x in (ht,hs,av,y)];z=mask_logits(probes,ht,hs,av);m=min(args.candidate_m,z[0].shape[1]);cand=torch.topk(z[0],m,1).indices
  ss=torch.stack([q.gather(1,cand) for q in z]);yy=y.cpu().numpy();cc=cand.cpu().numpy();add_rank(acc,"target",ss[0].cpu().numpy(),cc,yy);add_rank(acc,"dense",ss[-1].cpu().numpy(),cc,yy)
  singles=ss[1:1+len(sources)]
  view_ranks=np.stack([rank_values(q.cpu().numpy(),cc,yy) for q in torch.cat([ss[0:1],singles],0)]);valid=view_ranks>0;oracle=np.where(valid.any(0),np.where(valid,view_ranks,np.iinfo(np.int64).max).min(0),0);acc["oracle"].extend(oracle.tolist())
  uc=torch.as_tensor(user_choice[off:off+b],device=device);us=singles.permute(1,0,2)[torch.arange(b,device=device),uc];add_rank(acc,"user",us.cpu().numpy(),cc,yy)
  for d in range(len(sources)):add_rank(acc,f"fixed{d}",singles[d].cpu().numpy(),cc,yy)
  for kind,model in models.items():
   gs=[]
   for d in range(len(sources)):
    dom=torch.full((b*m,),d,device=device,dtype=torch.long);g=model(ht[:,None].expand(-1,m,-1).reshape(b*m,-1),hs[:,None].expand(-1,m,-1,-1).reshape(b*m,len(sources),-1),item[cand.reshape(-1)],dom);gs.append(g.reshape(b,m))
   g=torch.stack(gs,1);gv,gd=g.max(1);routed=singles.permute(1,0,2)[torch.arange(b,device=device)[:,None],gd,torch.arange(m,device=device)[None]]
   for t in thresholds:add_rank(acc,f"{kind}:{t:.6g}",torch.where(gv>t,routed,ss[0]).cpu().numpy(),cc,yy)
  off+=b
 def met(r):
  r=np.asarray(r);c=r>0;hit=c&(r<=10);rc=r[c]
  ndcg=np.zeros(len(r),dtype=float);ndcg[hit]=1/np.log2(r[hit]+1);mrr=np.zeros(len(r),dtype=float);mrr[c]=1/r[c]
  return {"Recall@5000":float(c.mean()),"positive_users":int(c.sum()),"Recall@10":float(hit.mean()),"NDCG@10":float(ndcg.mean()),"MRR":float(mrr.mean()),"conditional_Recall@10":float((rc<=10).mean()) if len(rc) else 0.0,"conditional_NDCG@10":float(ndcg[c].mean()) if len(rc) else 0.0,"conditional_MRR":float((1/rc).mean()) if len(rc) else 0.0,"RerankSuccess@10":float((rc<=10).mean()) if len(rc) else 0.0}
 return {k:met(v) for k,v in acc.items()},acc
def paired_ndcg_significance(ranks,reference,comparators,seed,samples=5000):
 rng=np.random.default_rng(seed);out={};ref=np.asarray(ranks[reference]);covered=ref>0
 def contrib(r):
  r=np.asarray(r);z=np.zeros(len(r),dtype=float);hit=(r>0)&(r<=10);z[hit]=1/np.log2(r[hit]+1);return z
 yp=contrib(ref)
 for name in comparators:
  delta=yp-contrib(np.asarray(ranks[name]));rows={}
  for label,x in (("end_to_end",delta),("conditional",delta[covered])):
   observed=float(x.mean());n=len(x);boot=np.empty(samples);extreme=0
   for st in range(0,samples,250):
    q=min(250,samples-st);signs=rng.choice((-1.0,1.0),size=(q,n));extreme+=int((np.abs((signs*x).mean(1))>=abs(observed)).sum());boot[st:st+q]=x[rng.integers(0,n,size=(q,n))].mean(1)
   rows[label]={"delta_NDCG@10":observed,"permutation_p_two_sided":float((extreme+1)/(samples+1)),"bootstrap_95ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"n":n}
  out[f"{reference}_vs_{name}"]=rows
 return out

def main():
 args=args_parse();seed_all(args.seed)
 if args.candidate_m!=5000:raise ValueError("Protocol is frozen: candidate-m must remain 5000")
 root=Path(__file__).resolve().parent;device=torch.device(args.device);out=Path(args.output_dir).resolve() if args.output_dir else root/"save/pairwise_reranker"/args.target_dom;out.mkdir(parents=True,exist_ok=True)
 base,splits,sources=load_stage1(root/"save/multisource_utility"/args.target_dom,args.target_dom,device);splits["train"]=subset(splits["train"],20000,args.seed);splits["valid"]=subset(splits["valid"],5000,args.seed+1)
 probes=folds_load(root,args.target_dom,base,sources,device);train_c=make_pair_cache(splits["train"],(fold_ids(len(splits["train"].users),args.seed),probes),device,args.score_batch_size,args.hard_k,args.neg_strata,args.seed+11);val_c=make_pair_cache(splits["valid"],probes,device,args.score_batch_size,args.hard_k,args.neg_strata,args.seed+12)
 item=base.target_item_embeddings.detach().to(device);data={"train":splits["train"],"valid":splits["valid"]};models={};hist={}
 for k in ("P1","P5"):models[k],hist[k]=train_gate(k,args,data,item,train_c,val_c,device,out)
 scale=float(np.std(train_c["pair"]));thresholds=np.linspace(-scale,scale,11).tolist();uv=read_user_choice(root/"save/stage2_utility_router"/args.target_dom/"predictions_valid.csv",splits["valid"].users,sources);vr,_=evaluate(splits["valid"],probes,models,item,args,device,sources,uv,thresholds)
 chosen={k:max(thresholds,key=lambda t:vr[f"{k}:{t:.6g}"]["NDCG@10"]) for k in models};ut=read_user_choice(root/"save/stage2_utility_router"/args.target_dom/"predictions_test.csv",splits["test"].users,sources);tr,tranks=evaluate(splits["test"],probes,models,item,args,device,sources,ut,sorted(set(chosen.values())));fixed=max(range(len(sources)),key=lambda d:vr[f"fixed{d}"]["NDCG@10"])
 test={"Target-only":tr["target"],"Dense":tr["dense"],"Fixed-source":tr[f"fixed{fixed}"],"User-level Router":tr["user"],"P1":tr[f"P1:{chosen['P1']:.6g}"],"P5":tr[f"P5:{chosen['P5']:.6g}"],"Candidate Oracle":tr["oracle"]}
 significance=paired_ndcg_significance(tranks,f"P5:{chosen['P5']:.6g}",[f"P1:{chosen['P1']:.6g}","user","dense"],args.seed+99)
 report={"protocol":{"candidate_selection":"Target-only Top-5000 selected exclusively on validation coverage","candidate_pool":"OOF ensemble target Top-5000","hard_negatives":{"total":args.hard_k,"strata":{"Top100":args.neg_strata[0],"rank101_500":args.neg_strata[1],"rank501_5000":args.neg_strata[2]}},"utility":"(source positive-negative margin) - (target positive-negative margin)","threshold":"validation NDCG@10","test":"frozen; no candidate-policy tuning"},"args":vars(args),"sources":sources,"valid_threshold_sweep":vr,"chosen_thresholds":chosen,"fixed_source":sources[fixed],"test":test,"paired_significance":significance};save_json(out/"pairwise_reranker_report.json",report);print(json.dumps({"chosen":chosen,"fixed":sources[fixed],"test":test},indent=2));print(f"[REPORT] {out/'pairwise_reranker_report.json'}")
if __name__=="__main__":main()
