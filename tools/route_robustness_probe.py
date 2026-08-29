import json, collections, random
from evaluator.local_evaluator import intent_card, coarse_category
from starter.agent import _terms
prods,cards,cats={},{},{}
for line in open('data/catalog.jsonl',encoding='utf-8'):
    p=json.loads(line); a=str(p['parent_asin']); prods[a]=p
    cards[a]=intent_card(p); cats[a]=coarse_category([str(v) for v in p.get('categories') or []])
emit=collections.defaultdict(set)
for a,c in cards.items():
    for s in set(c['hard_constraints']+c['soft_preferences']): emit[s].add(a)
bkt=collections.defaultdict(set)
for a,c in cats.items(): bkt[c].add(a)
pop=lambda a: prods[a].get('rating_number') or 0
N=len(prods); import math
df=collections.Counter({s:len(x) for s,x in emit.items()})
tok=collections.defaultdict(set); dft=collections.Counter()
for s,x in emit.items():
    for t in set(_terms(s)): tok[t]|=x
for t,x in tok.items(): dft[t]=len(x)
samples=[json.loads(l) for l in open('data/public_set.jsonl',encoding='utf-8')]
def reword(s,rate,rng):
    w=s.split()
    if len(w)<3: return s
    k=[x for x in w if rng.random()>rate]
    return " ".join(k) if k else w[0]

def rank(name, clues, base):
    if name=="exact":
        c=set(base)
        for x in clues:
            if x in emit: c &= emit[x]
        return sorted(c,key=lambda a:-pop(a))
    if name in ("tok","tokidf"):
        sc=collections.Counter()
        for x in clues:
            for t in set(_terms(x)):
                w = math.log(N/max(dft[t],1)) if name=="tokidf" else 1.0
                for a in tok.get(t,()):
                    if a in base: sc[a]+=w
        return [a for a,_ in sorted(sc.items(),key=lambda kv:(-kv[1],-pop(kv[0])))]
    if name=="hybrid":
        c=set(base)
        for x in clues:
            if x in emit: c &= emit[x]
        if len(c)==1: return list(c)
        inner = rank("tokidf",clues,base)
        return inner
    return sorted(base,key=lambda a:-pop(a))

def run(rate,seed=0):
    rng=random.Random(seed); out={}
    data=[]
    for s in samples:
        t=str(s['ground_truth']['parent_asin'])
        data.append((t,[reword(c,rate,rng) for c in cards[t]['hard_constraints']+cards[t]['soft_preferences']]))
    for name in ("exact","tokidf","hybrid"):
        hit=mrr=0
        for t,cl in data:
            lst=rank(name,cl,bkt[cats[t]])[:10]
            if t in lst: hit+=1; mrr+=1/(lst.index(t)+1)
        out[name]=(hit/len(data), mrr/len(data))
    return out

print(f"{'dropped':>8s} |{'exact ∩ only':^15s}|{'token only':^15s}|{'SINGLETON+TOKEN':^17s}")
print(f"{'':>8s} |{'hit':>6s}{'MRR':>9s}|{'hit':>6s}{'MRR':>9s}|{'hit':>7s}{'MRR':>10s}")
print("-"*62)
for r in (0.0,0.2,0.35,0.5):
    o=run(r)
    print(f"{int(r*100):>7d}% |{o['exact'][0]:6.3f}{o['exact'][1]:9.3f}|{o['tokidf'][0]:6.3f}{o['tokidf'][1]:9.3f}|"
          f"{o['hybrid'][0]:7.3f}{o['hybrid'][1]:10.3f}")
