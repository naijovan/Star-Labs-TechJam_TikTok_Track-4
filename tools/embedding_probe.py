import json, collections, random, time, sys
sys.path.insert(0,'.')
from evaluator.local_evaluator import intent_card, coarse_category
from starter._original_bm25_agent import _terms
import numpy as np
from sentence_transformers import SentenceTransformer

prods,cards,cats={},{},{}
for line in open('data/catalog.jsonl',encoding='utf-8'):
    p=json.loads(line); a=str(p['parent_asin']); prods[a]=p
    cards[a]=intent_card(p); cats[a]=coarse_category([str(v) for v in p.get('categories') or []])
asins=list(prods)
emit=collections.defaultdict(set)
for a,c in cards.items():
    for s in set(c['hard_constraints']+c['soft_preferences']): emit[s].add(a)
bkt=collections.defaultdict(set)
for a,c in cats.items(): bkt[c].add(a)
pop=lambda a: prods[a].get('rating_number') or 0
tok=collections.defaultdict(set)
for s,x in emit.items():
    for t in set(_terms(s)): tok[t]|=x

texts=[" ; ".join(cards[a]['hard_constraints']+cards[a]['soft_preferences']) for a in asins]
t0=time.time()
m=SentenceTransformer('BAAI/bge-small-en-v1.5', device='mps')
E=m.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
print(f"embedded {len(texts):,} products with bge-small-en-v1.5 in {time.time()-t0:.0f}s  dim={E.shape[1]}")
idx={a:i for i,a in enumerate(asins)}

samples=[json.loads(l) for l in open('data/public_set.jsonl',encoding='utf-8')]
def reword(s,rate,rng):
    w=s.split()
    if len(w)<3: return s
    k=[x for x in w if rng.random()>rate]
    return " ".join(k) if k else w[0]

def run(rate,seed=0):
    rng=random.Random(seed)
    qtexts=[]; meta=[]
    for s in samples:
        t=str(s['ground_truth']['parent_asin'])
        cl=[reword(c,rate,rng) for c in cards[t]['hard_constraints']+cards[t]['soft_preferences']]
        qtexts.append(" ; ".join(cl)); meta.append((t,cl))
    Q=m.encode(qtexts,batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    out={}
    for name in ("exact","tokens","dense"):
        hit=mrr=0
        for qi,(t,cl) in enumerate(meta):
            base=bkt[cats[t]]
            if name=="exact":
                c=set(base)
                for x in cl:
                    if x in emit: c &= emit[x]
                lst=sorted(c,key=lambda a:-pop(a))
            elif name=="tokens":
                sc=collections.Counter()
                for x in cl:
                    for tt in set(_terms(x)):
                        for a in tok.get(tt,()):
                            if a in base: sc[a]+=1
                lst=[a for a,_ in sorted(sc.items(),key=lambda kv:(-kv[1],-pop(kv[0])))]
            else:
                b=list(base); sims=E[[idx[a] for a in b]] @ Q[qi]
                lst=[b[i] for i in np.argsort(-sims)]
            top=lst[:10]
            if t in top: hit+=1; mrr+=1/(top.index(t)+1)
        out[name]=(hit/len(samples), mrr/len(samples))
    return out

print(f"\n{'dropped':>8s} | {'exact ∩':^16s} | {'token overlap':^16s} | {'bge-small dense':^16s}")
print(f"{'':>8s} | {'hit':>6s} {'MRR':>8s} | {'hit':>6s} {'MRR':>8s} | {'hit':>6s} {'MRR':>8s}")
print("-"*74)
for r in (0.0,0.2,0.35,0.5):
    o=run(r)
    print(f"{int(r*100):>7d}% | {o['exact'][0]:6.3f} {o['exact'][1]:8.3f} | {o['tokens'][0]:6.3f} {o['tokens'][1]:8.3f} | {o['dense'][0]:6.3f} {o['dense'][1]:8.3f}")
