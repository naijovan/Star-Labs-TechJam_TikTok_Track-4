"""Compare answer-timing policies for the intersection agent. PYTHONPATH=. python3 tools/policy_probe.py"""

import json, collections
from evaluator.local_evaluator import intent_card, coarse_category
prods,cards,cats={},{},{}
for line in open('data/catalog.jsonl',encoding='utf-8'):
    p=json.loads(line); a=str(p['parent_asin']); prods[a]=p
    cards[a]=intent_card(p); cats[a]=coarse_category([str(v) for v in p.get('categories') or []])
inv=collections.defaultdict(set)
for a,c in cards.items():
    for s in set(c['hard_constraints']+c['soft_preferences']): inv[s].add(a)
bkt=collections.defaultdict(set)
for a,c in cats.items(): bkt[c].add(a)
pop=lambda a: prods[a].get('rating_number') or 0
samples=[json.loads(l) for l in open('data/public_set.jsonl',encoding='utf-8')]

def run(policy):
    res=[]
    for s in samples:
        tgt=str(s['ground_truth']['parent_asin']); card=cards[tgt]; sc=s['scenario_type']
        pool=card['hard_constraints']+card['soft_preferences']
        disclosed=set(); known=[]
        if sc=='buying' and card['hard_constraints']:
            v=str(card['hard_constraints'][0]); disclosed.add(v); known.append(v)
        elif sc=='intent_override':
            soft=card['soft_preferences']
            if soft: known.append(str(soft[-1]))
        ht=hr=None
        for turn in range(1,11):
            cand=set(bkt[cats[tgt]])
            for c in known:
                if c in inv: cand &= inv[c]
            ordered=sorted(cand,key=lambda a:(-pop(a),a))
            gated=(sc=='intent_override' and turn<3)
            emit = policy(turn, len(cand))
            if emit and not gated and tgt in ordered[:10]:
                ht=turn; hr=ordered.index(tgt)+1; break
            m=[v for v in pool if v not in disclosed][:2]
            disclosed.update(m); known.extend(m)
        res.append((ht,hr))
    hits=[r for r in res if r[0]]
    hit=len(hits)/len(res); mrr=sum(1/r[1] for r in hits)/len(res)
    mttc=sum((r[0] or 11) for r in res)/len(res); eff=max(0,min(1,(11-mttc)/10))
    return hit,mrr,mttc,eff,0.5*hit+0.3*mrr+0.2*eff

policies = {
 "greedy (answer every turn)":        lambda t,k: True,
 "wait until turn 3 always":          lambda t,k: t>=3,
 "ADAPTIVE: answer iff |cand|==1":    lambda t,k: k==1,
 "ADAPTIVE: |cand|==1, else turn>=3": lambda t,k: k==1 or t>=3,
 "ADAPTIVE: |cand|<=2, else turn>=3": lambda t,k: k<=2 or t>=3,
 "ADAPTIVE: |cand|<=3, else turn>=3": lambda t,k: k<=3 or t>=3,
}
print(f"{'policy':38s} {'hit':>6s} {'MRR':>6s} {'MTTC':>6s} {'eff':>6s} {'SCORE':>8s}")
print("-"*76)
best=None
for name,p in policies.items():
    h,m,t,e,s = run(p)
    print(f"{name:38s} {h:6.3f} {m:6.3f} {t:6.2f} {e:6.3f} {s:8.5f}")
    if best is None or s>best[1]: best=(name,s)
print("-"*76)
print(f"BEST: {best[0]}  ->  {best[1]:.5f}")
print(f"(baseline 0.10671 | floor probe 0.83199)")
