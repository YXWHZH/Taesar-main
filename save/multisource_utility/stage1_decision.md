# Stage-1 Multi-domain Utility Decision

**Decision: STRONG_GO**

至少3/4目标域同时支持负效用异质性、非单域垄断与稀疏选择优势。

|Target|Avg negative utility|Max best-source share|Dense-Oracle1 CE|Oracle1 beats Dense|Safe2-Dense R@10|Safe2-Dense NDCG@10|
|---|---:|---:|---:|---:|---:|---:|
|dom1|49.26%|33.67%|0.119228|76.98%|0.001811|0.000993|
|dom2|50.13%|34.14%|0.087280|82.43%|0.000956|0.000550|
|dom3|50.71%|34.08%|0.134381|81.96%|0.001669|0.000835|
|dom4|46.72%|36.69%|0.045146|78.54%|0.001311|0.000658|

判据：负效用率≥20%；最大Best-source占比≤70%；Dense-Oracle Top1 CE gap>0；Oracle Top1在≥60%用户上优于Dense；Recall/NDCG至少有一个正向Oracle/Safe gap。