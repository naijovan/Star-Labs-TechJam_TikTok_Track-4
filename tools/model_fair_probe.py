import json, collections, random, math, sys, time
sys.path.insert(0,'.')
from evaluator.local_evaluator import intent_card, coarse_category
from starter.agent import _terms
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

prods,cards,cats={},{},{}
for line in open('data/catalog.jsonl',encoding='utf-8'):
    p=json.loads(line); a=str(p['parent_asin']); prods[a]=p
    cards[a]=intent_card(p); cats[a]=coarse_category([str(v) for v in p.get('categories') or []])
asins=list(prods)
emit=collections.defaultdict(set); docs={}; text={}
for a,c in cards.items():
    cs=c['hard_constraints']+c['soft_preferences']
    for s in set(cs): emit[s].add(a)
    docs[a]=_terms(" ".join(cs)); text[a]=" ; ".join(cs)
bkt=collections.defaultdict(set)
for a,c in cats.items(): bkt[c].add(a)
pop=lambda a: prods[a].get('rating_number') or 0
N=len(docs); avgdl=sum(len(d) for d in docs.values())/N
tf={a:collections.Counter(d) for a,d in docs.items()}
post=collections.defaultdict(set)
for a,d in docs.items():
    for t in set(d): post[t].add(a)
idf={t: math.log(1+(N-len(s)+0.5)/(len(s)+0.5)) for t,s in post.items()}
def bm25(q, base, k1=1.2, b=0.75):
    sc=collections.Counter()
    for t in q:
        if t not in post: continue
        w=idf[t]
        for a in post[t]:
            if a not in base: continue
            f=tf[a][t]; dl=len(docs[a])
            sc[a]+= w*(f*(k1+1))/(f+k1*(1-b+b*dl/avgdl))
    return [a for a,_ in sorted(sc.items(),key=lambda kv:(-kv[1],-pop(kv[0])))]

t0=time.time()
emb=SentenceTransformer('BAAI/bge-small-en-v1.5', device='mps')
E=emb.encode([text[a] for a in asins],batch_size=256,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
idx={a:i for i,a in enumerate(asins)}
ce=CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='mps', max_length=256)
print(f"models loaded + 50k embedded in {time.time()-t0:.0f}s")

BGE_Q="Represent this sentence for searching relevant passages: "
samples=[json.loads(l) for l in open('data/public_set.jsonl',encoding='utf-8')]
def reword(s,rate,rng):
    w=s.split()
    if len(w)<3: return s
    k=[x for x in w if rng.random()>rate]
    return " ".join(k) if k else w[0]

def run(rate,seed=0):
    rng=random.Random(seed)
    data=[]
    for s in samples:
        t=str(s['ground_truth']['parent_asin'])
        data.append((t,[reword(c,rate,rng) for c in cards[t]['hard_constraints']+cards[t]['soft_preferences']]))
    qraw=[" ; ".join(cl) for _,cl in data]
    Qp=emb.encode([BGE_Q+q for q in qraw],batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    Qn=emb.encode(qraw,batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    out={}
    for name in ("bm25","dense_noprefix","dense_prefix","bm25_then_ce","hybrid_bm25_dense"):
        hit=mrr=0
        for qi,(t,cl) in enumerate(data):
            base=bkt[cats[t]]; q=_terms(" ".join(cl))
            if name=="bm25": lst=bm25(q,base)
            elif name.startswith("dense"):
                Q = Qn if name=="dense_noprefix" else Qp
                b=list(base); sims=E[[idx[a] for a in b]] @ Q[qi]
                lst=[b[i] for i in np.argsort(-sims)]
            elif name=="bm25_then_ce":
                top=bm25(q,base)[:20]
                if len(top)>1:
                    sc=ce.predict([(qraw[qi], text[a]) for a in top], show_progress_bar=False)
                    lst=[a for _,a in sorted(zip(sc,top),key=lambda p:-p[0])]
                else: lst=top
            else:
                bl=bm25(q,base)[:50]
                if len(bl)>1:
                    sims=E[[idx[a] for a in bl]] @ Qp[qi]
                    rr={a:1/(i+60) for i,a in enumerate(bl)}
                    for i,a in enumerate([bl[j] for j in np.argsort(-sims)]): rr[a]+=1/(i+60)
                    lst=sorted(bl,key=lambda a:-rr[a])
                else: lst=bl
            lst=lst[:10]
            if t in lst: hit+=1; mrr+=1/(lst.index(t)+1)
        out[name]=(hit/len(data),mrr/len(data))
    return out

print(f"\n{'drop':>5s} |{'BM25':^13s}|{'dense no-prefix':^15s}|{'dense +prefix':^15s}|{'BM25→cross-enc':^16s}|{'BM25+dense RRF':^16s}")
print("-"*94)
for r in (0.0,0.35):
    o=run(r)
    print(f"{int(r*100):>4d}% |"+ "|".join(f"{o[k][0]:6.3f}{o[k][1]:7.3f}" for k in
          ("bm25","dense_noprefix","dense_prefix","bm25_then_ce","hybrid_bm25_dense")))
