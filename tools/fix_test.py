import json, collections, random, math, sys, time
sys.path.insert(0,'.')
from evaluator.local_evaluator import intent_card, coarse_category
from starter.agent import _terms
import numpy as np
from sentence_transformers import SentenceTransformer

prods,cards,cats={},{},{}
for line in open('data/catalog.jsonl',encoding='utf-8'):
    p=json.loads(line); a=str(p['parent_asin']); prods[a]=p
    cards[a]=intent_card(p); cats[a]=coarse_category([str(v) for v in p.get('categories') or []])
asins=list(prods)
bkt=collections.defaultdict(set)
for a,c in cats.items(): bkt[c].add(a)
pop=lambda a: prods[a].get('rating_number') or 0
CL={a:(cards[a]['hard_constraints']+cards[a]['soft_preferences']) for a in asins}

# ---- HELD-OUT SPLIT: index on clues[:2], query with clues[2:] ----
IDXC={a:CL[a][:2] for a in asins}      # what the agent may index
QRYC={a:CL[a][2:] for a in asins}      # unseen phrasing at query time

def build_lex(scope):
    docs={a:_terms(" ".join(scope[a])) for a in asins}
    N=len(docs); avgdl=sum(len(d) for d in docs.values())/max(N,1)
    tf={a:collections.Counter(d) for a,d in docs.items()}
    post=collections.defaultdict(set)
    for a,d in docs.items():
        for t in set(d): post[t].add(a)
    idf={t: math.log(1+(N-len(s)+0.5)/(len(s)+0.5)) for t,s in post.items()}
    def score(q, base, k1=1.2, b=0.75):
        sc=collections.Counter()
        for t in q:
            if t not in post: continue
            w=idf[t]
            for a in post[t]:
                if a not in base: continue
                f=tf[a][t]; dl=len(docs[a])
                sc[a]+= w*(f*(k1+1))/(f+k1*(1-b+b*dl/avgdl))
        return [a for a,_ in sorted(sc.items(),key=lambda kv:(-kv[1],-pop(kv[0])))]
    return score

t0=time.time()
m=SentenceTransformer('BAAI/bge-small-en-v1.5', device='mps')
# doc-level embedding of the INDEXABLE clues
D=m.encode([" ; ".join(IDXC[a]) or "item" for a in asins],batch_size=256,convert_to_numpy=True,
           normalize_embeddings=True,show_progress_bar=False)
# clue-level: embed each indexable clue separately, keep per-product list
flat=[]; owner=[]
for a in asins:
    for c in (IDXC[a] or ["item"]): flat.append(c); owner.append(a)
F=m.encode(flat,batch_size=512,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
byprod=collections.defaultdict(list)
for i,a in enumerate(owner): byprod[a].append(i)
print(f"embedded {len(asins):,} docs + {len(flat):,} individual clues in {time.time()-t0:.0f}s")
didx={a:i for i,a in enumerate(asins)}

samples=[json.loads(l) for l in open('data/public_set.jsonl',encoding='utf-8')]
lex_idx=build_lex(IDXC)

qtexts=[]; meta=[]
for s in samples:
    t=str(s['ground_truth']['parent_asin'])
    q=QRYC[t] or CL[t][-1:]
    qtexts.append(" ; ".join(q)); meta.append((t,q))
QD=m.encode(qtexts,batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
qflat=[]; qown=[]
for i,(t,q) in enumerate(meta):
    for c in q: qflat.append(c); qown.append(i)
QF=m.encode(qflat,batch_size=256,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
qby=collections.defaultdict(list)
for i,qi in enumerate(qown): qby[qi].append(i)

def ev(name):
    hit=mrr=0
    for qi,(t,q) in enumerate(meta):
        base=list(bkt[cats[t]])
        if name=="bm25":
            lst=lex_idx(_terms(" ".join(q)), set(base))
        elif name=="dense_doc":
            sims=D[[didx[a] for a in base]] @ QD[qi]
            lst=[base[i] for i in np.argsort(-sims)]
        else:  # clue-level max-sim (ColBERT-style)
            qv=QF[qby[qi]]
            sc=[]
            for a in base:
                pv=F[byprod[a]]
                sc.append(float((qv @ pv.T).max(axis=1).sum()))
            lst=[base[i] for i in np.argsort(-np.array(sc))]
        lst=lst[:10]
        if t in lst: hit+=1; mrr+=1/(lst.index(t)+1)
    return hit/len(meta), mrr/len(meta)

print("\nHELD-OUT TEST — index knows clues 1-2, query uses clues 3-4 (never-seen phrasing)")
print(f"{'method':34s} {'hit@10':>7s} {'MRR':>7s}")
print("-"*52)
for n,l in [("bm25","BM25 lexical"),("dense_doc","dense, document-level"),("dense_max","dense, clue-level max-sim")]:
    h,mm=ev(n); print(f"{l:34s} {h:7.3f} {mm:7.3f}")
