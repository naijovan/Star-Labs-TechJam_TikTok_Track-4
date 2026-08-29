import sys, collections, statistics
sys.path.insert(0,'.')
import evaluator.local_evaluator as LE
from submission.agent import Agent
samples=LE.load_jsonl('data/public_set.jsonl'); ids,cats,prods=LE.catalog_index('data/catalog.jsonl')
ag=Agent('data/catalog.jsonl'); pop=lambda a: ag.pop.get(a,0)

best_rows=[]
for s in samples:
    t=str(s['ground_truth']['parent_asin']); sc=s['scenario_type']
    card,beh=LE.materialize_hidden_fields(s,prods)
    hard,soft=card['hard_constraints'],card['soft_preferences']; pool=hard+soft
    bucket=ag.bucket.get(LE.coarse_category(cats.get(t,[])),set())
    ovturn=int(beh.get('override',{}).get('turn',0)) if sc=='intent_override' else 0
    disclosed=set(); acc=[]
    if sc=='buying' and hard: acc.append(hard[0]); disclosed.add(hard[0])
    elif sc=='intent_override' and soft: acc.append(soft[-1])
    known={1:list(acc)}
    for k in range(2,11):
        m=[v for v in pool if v not in disclosed][:2]; disclosed.update(m); acc=acc+m; known[k]=list(acc)
    # per-session best achievable value, choosing the turn to answer
    bestv=(0.0,None,None)
    for k in range(1,11):
        if sc=='intent_override' and k<ovturn: continue
        cand=set(bucket)
        for c in known[k]:
            if c in ag.clue_to: cand &= ag.clue_to[c]
        if not cand: continue
        order=sorted(cand,key=lambda a:(-pop(a),a))[:10]
        if t not in order: continue
        r=order.index(t)+1
        v=0.5 + 0.3/r + 0.2*(11-k)/10          # this session's contribution if answered at turn k
        if v>bestv[0]: bestv=(v,k,r)
    best_rows.append((s['sample_id'],sc)+bestv[1:]+ (bestv[0],))

hits=[r for r in best_rows if r[2]]
hit=len(hits)/len(best_rows)
mrr=sum(1/r[3] for r in hits)/len(best_rows)
mttc=sum((r[2] if r[2] else 11) for r in best_rows)/len(best_rows)
eff=max(0,min(1,(11-mttc)/10))
print("TRUE CEILING — each session answered at its own optimal turn, perfect ranking within")
print("the information actually available (popularity tie-break, override gate respected)")
print(f"  hit={hit:.3f}  MRR={mrr:.3f}  MTTC={mttc:.2f}  eff={eff:.3f}")
print(f"  ==> CEILING {0.5*hit+0.3*mrr+0.2*eff:.5f}")
print(f"  current agent                    0.973589")
print(f"  ACHIEVABLE HEADROOM              {0.5*hit+0.3*mrr+0.2*eff-0.973589:+.5f}")
print(f"\n  optimal answer turn: {dict(sorted(collections.Counter(r[2] for r in best_rows).items(), key=lambda kv:(kv[0] is None,kv[0])))}")
print(f"  rank when answered : {dict(sorted(collections.Counter(r[3] for r in hits).items()))}")
print(f"\nWhy not 1.000:")
print(f"  a) 15% of sessions CANNOT convert before turn 3-4 (override gate) -> MTTC floor")
print(f"  b) browsing/boundary know only the category on turn 1 (~180 products)")
print(f"  c) {len(best_rows)-len(hits)} sessions never reach the top 10 at any turn")
print(f"  d) MRR 1.000 needs rank 1 everywhere; {sum(1 for r in hits if r[3]>1)} sessions cannot")
