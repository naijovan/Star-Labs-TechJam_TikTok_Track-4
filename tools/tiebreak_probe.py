import json, collections, statistics, random
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

amb=[]
for s in samples:
    tgt=str(s['ground_truth']['parent_asin']); card=cards[tgt]
    cand=set(bkt[cats[tgt]])
    for c in card['hard_constraints']+card['soft_preferences']:
        if c in inv: cand &= inv[c]
    if len(cand)>1: amb.append((tgt,cand))

print(f"sessions where intersection leaves >1 candidate: {len(amb)}/200 ({len(amb)/2:.1f}%)")
print(f"  ambiguous-set sizes: {sorted(collections.Counter(len(c) for _,c in amb).items())}\n")

rng=random.Random(0)
def mrr(key):
    tot=0
    for tgt,cand in amb:
        o=sorted(cand,key=key); tot += 1/(o.index(tgt)+1) if tgt in o[:10] else 0
    return tot/len(amb)

print(f"MRR on those {len(amb)} ambiguous sessions only:")
print(f"  popularity (rating_number desc) : {mrr(lambda a:(-pop(a),a)):.3f}")
print(f"  average_rating desc             : {mrr(lambda a:(-(prods[a].get('average_rating') or 0),a)):.3f}")
print(f"  arbitrary (asin order)          : {mrr(lambda a:a):.3f}")
print(f"  random shuffle                  : {mrr(lambda a: rng.random()):.3f}")

cat_med=statistics.median(pop(a) for a in prods)
tgt_med=statistics.median(pop(str(s['ground_truth']['parent_asin'])) for s in samples)
print(f"\npopularity skew: catalog median {cat_med:,.0f} reviews vs target median {tgt_med:,.0f}  ({tgt_med/max(cat_med,1):.0f}x)")
