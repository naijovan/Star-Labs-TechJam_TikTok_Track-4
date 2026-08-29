import json, collections, statistics, random
from evaluator.local_evaluator import (intent_card, coarse_category, searchable_text,
                                       MATERIAL_RE, COLOR_RE, behavior_for)

prods, cards, cats = {}, {}, {}
for line in open('data/catalog.jsonl', encoding='utf-8'):
    p = json.loads(line); a = str(p['parent_asin']); prods[a] = p
    cards[a] = intent_card(p)
    cats[a] = coarse_category([str(v) for v in p.get('categories') or []])

# inverted index: constraint string -> set of asins EMITTING it
inv = collections.defaultdict(set)
for a, c in cards.items():
    for s in set(c['hard_constraints'] + c['soft_preferences']):
        inv[s].add(a)

bucket = collections.defaultdict(set)
for a, cat in cats.items(): bucket[cat].add(a)

samples = [json.loads(l) for l in open('data/public_set.jsonl', encoding='utf-8')]

def harvest(card, scenario, sample_id):
    """Constraints the customer will have disclosed by the time info saturates."""
    pool = card['hard_constraints'] + card['soft_preferences']
    seen, disclosed = [], set()
    if scenario == 'buying' and card['hard_constraints']:
        v = str(card['hard_constraints'][0]); disclosed.add(v); seen.append(v)
    elif scenario == 'intent_override':
        soft = card['soft_preferences']
        v = str(soft[-1]) if soft else None
        if v: seen.append(v)          # spoken but NOT added to `disclosed`
    for _ in range(3):                 # up to 3 "other" asks
        m = [v for v in pool if v not in disclosed][:2]
        if not m: break
        disclosed.update(m); seen.extend(m)
    return list(dict.fromkeys(seen))

rows = []
for s in samples:
    tgt = str(s['ground_truth']['parent_asin']); card = cards[tgt]
    hv = harvest(card, s['scenario_type'], s['sample_id'])
    cand_all = bucket[cats[tgt]]                       # category filter (turn 1)
    cand_int = set(cand_all)
    for c in hv:                                        # conjunctive intersection
        if c in inv: cand_int &= inv[c]
    if tgt not in cand_int: cand_int = set()            # sanity
    rows.append(dict(sid=s['sample_id'], sc=s['scenario_type'],
                     n_harvest=len(hv), bucket=len(cand_all), inter=len(cand_int)))

def pct(f, rs=rows): return 100*sum(1 for r in rs if f(r))/len(rs)
print(f"sessions: {len(rows)}")
print(f"\n[A] CATEGORY BUCKET size      median {statistics.median(r['bucket'] for r in rows):8.0f}")
print(f"[B] AFTER conjunctive intersection with harvested constraints:")
print(f"    median candidates        {statistics.median(r['inter'] for r in rows):8.0f}")
print(f"    EXACTLY 1 (solved)       {pct(lambda r: r['inter']==1):8.1f}%")
print(f"    <= 2                     {pct(lambda r: 1<=r['inter']<=2):8.1f}%")
print(f"    <= 5                     {pct(lambda r: 1<=r['inter']<=5):8.1f}%")
print(f"    <=10 (guaranteed hit)    {pct(lambda r: 1<=r['inter']<=10):8.1f}%")
print(f"    0 (BROKEN - target lost) {pct(lambda r: r['inter']==0):8.1f}%")

print(f"\n[C] by scenario:")
for sc in sorted({r['sc'] for r in rows}):
    rs=[r for r in rows if r['sc']==sc]
    print(f"    {sc:16s} n={len(rs):3d}  solved={pct(lambda r:r['inter']==1,rs):5.1f}%  "
          f"<=10={pct(lambda r:1<=r['inter']<=10,rs):5.1f}%  median={statistics.median(x['inter'] for x in rs):5.0f}")

print(f"\n[D] harvested-constraint count distribution: "
      f"{dict(sorted(collections.Counter(r['n_harvest'] for r in rows).items()))}")

# implied MRR if we rank ties by popularity (assume uniform position within tie)
mrr = 0.0
for r in rows:
    k = r['inter']
    mrr += 1.0 if k==1 else (0.0 if k==0 else sum(1/i for i in range(1,min(k,10)+1))/min(k,10))
print(f"\n[E] implied MRR from intersection alone (ties broken randomly): {mrr/len(rows):.3f}")
