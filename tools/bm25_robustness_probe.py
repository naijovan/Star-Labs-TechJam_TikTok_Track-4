import json, collections, random, math
from evaluator.local_evaluator import intent_card, coarse_category
from starter.agent import _terms

prods,cards,cats={},{},{}
for line in open('data/catalog.jsonl',encoding='utf-8'):
    p=json.loads(line); a=str(p['parent_asin']); prods[a]=p
    cards[a]=intent_card(p); cats[a]=coarse_category([str(v) for v in p.get('categories') or []])
emit=collections.defaultdict(set)
docs={}
for a,c in cards.items():
    cs=c['hard_constraints']+c['soft_preferences']
    for s in set(cs): emit[s].add(a)
    docs[a]=_terms(" ".join(cs))                       # the emitted-constraint document
bkt=collections.defaultdict(set)
for a,c in cats.items(): bkt[c].add(a)
pop=lambda a: prods[a].get('rating_number') or 0

N=len(docs); avgdl=sum(len(d) for d in docs.values())/N
tf={a:collections.Counter(d) for a,d in docs.items()}
post=collections.defaultdict(set)
for a,d in docs.items():
    for t in set(d): post[t].add(a)
idf={t: math.log(1 + (N-len(s)+0.5)/(len(s)+0.5)) for t,s in post.items()}
print(f"corpus: {N:,} docs, avg {avgdl:.1f} tokens, {len(post):,} distinct terms")

def bm25(qterms, base, k1=1.2, b=0.75):
    sc=collections.Counter()
    for t in qterms:
        if t not in post: continue
        w=idf[t]
        for a in post[t]:
            if a not in base: continue
            f=tf[a][t]; dl=len(docs[a])
            sc[a]+= w*(f*(k1+1))/(f + k1*(1-b+b*dl/avgdl))
    return [a for a,_ in sorted(sc.items(), key=lambda kv:(-kv[1],-pop(kv[0])))]

def count(qterms, base):
    sc=collections.Counter()
    for t in set(qterms):
        for a in post.get(t,()):
            if a in base: sc[a]+=1
    return [a for a,_ in sorted(sc.items(), key=lambda kv:(-kv[1],-pop(kv[0])))]

samples=[json.loads(l) for l in open('data/public_set.jsonl',encoding='utf-8')]
def reword(s,rate,rng):
    w=s.split()
    if len(w)<3: return s
    k=[x for x in w if rng.random()>rate]
    return " ".join(k) if k else w[0]

def run(rate,seed=0):
    rng=random.Random(seed); out={}
    data=[]
    for s in samples:
        t=str(s['ground_truth']['parent_asin'])
        data.append((t,[reword(c,rate,rng) for c in cards[t]['hard_constraints']+cards[t]['soft_preferences']]))
    for name in ("count","bm25_b75","hybrid_bm25","exact"):
        hit=mrr=0
        for t,cl in data:
            base=bkt[cats[t]]; q=_terms(" ".join(cl))
            if name=="count": lst=count(q,base)
            elif name=="bm25_b75": lst=bm25(q,base,b=0.75)
            elif name=="exact":
                c=set(base)
                for x in cl:
                    if x in emit: c &= emit[x]
                lst=sorted(c,key=lambda a:-pop(a))
            else:
                c=set(base)
                for x in cl:
                    if x in emit: c &= emit[x]
                lst=list(c) if len(c)==1 else bm25(q,base,b=0.75)
            lst=lst[:10]
            if t in lst: hit+=1; mrr+=1/(lst.index(t)+1)
        out[name]=(hit/len(data), mrr/len(data))
    return out

print(f"\n{'dropped':>8s} |{'word count':^15s}|{'BM25 b=0.75':^15s}|{'singleton+BM25':^15s}|{'exact only':^15s}")
print(f"{'':>8s} |{'hit':>6s}{'MRR':>9s}|{'hit':>6s}{'MRR':>9s}|{'hit':>6s}{'MRR':>9s}|{'hit':>7s}{'MRR':>10s}")
print("-"*78)
for r in (0.0,0.2,0.35,0.5):
    o=run(r)
    print(f"{int(r*100):>7d}% |{o['count'][0]:6.3f}{o['count'][1]:9.3f}|{o['bm25_b75'][0]:6.3f}{o['bm25_b75'][1]:9.3f}|"
          f"{o['hybrid_bm25'][0]:6.3f}{o['hybrid_bm25'][1]:9.3f}|{o['exact'][0]:6.3f}{o['exact'][1]:9.3f}")
