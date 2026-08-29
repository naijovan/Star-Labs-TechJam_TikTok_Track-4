import json, collections, sys
from evaluator.local_evaluator import intent_card, searchable_text, MATERIAL_RE, COLOR_RE

prods, cards = {}, {}
for line in open('data/catalog.jsonl', encoding='utf-8'):
    p = json.loads(line); a = str(p['parent_asin']); prods[a] = p
    cards[a] = intent_card(p)

N = len(prods)
print(f"products: {N}\n")

# --- 1. EMITTED constraint sets: does the full set uniquely identify? ---
emitted = {a: tuple(c['hard_constraints'] + c['soft_preferences']) for a, c in cards.items()}
sig = collections.Counter(emitted.values())
uniq_full = sum(1 for a in prods if sig[emitted[a]] == 1)
print(f"[1] FULL emitted-set uniquely identifies product: {uniq_full/N:6.1%}")

# how many constraints does each product actually emit (deduped)?
ncon = collections.Counter(len(set(emitted[a])) for a in prods)
print(f"    distinct emitted-constraint count: {dict(sorted(ncon.items()))}")

# --- 2. NEGATIVE EVIDENCE: material / color regex presence ---
has_mat = {a: bool(MATERIAL_RE.search(searchable_text(p))) for a, p in prods.items()}
has_col = {a: bool(COLOR_RE.search(searchable_text(p))) for a, p in prods.items()}
m = sum(has_mat.values()); c = sum(has_col.values())
print(f"\n[2] NEGATIVE EVIDENCE (whole-catalog partition)")
print(f"    has material word: {m/N:6.1%}   absent: {1-m/N:6.1%}")
print(f"    has color word:    {c/N:6.1%}   absent: {1-c/N:6.1%}")
quad = collections.Counter((has_mat[a], has_col[a]) for a in prods)
for k, v in sorted(quad.items()):
    print(f"    material={str(k[0]):5s} color={str(k[1]):5s} -> {v:6d}  ({v/N:5.1%})")

# --- 3. POSITIONAL matching: is position informative? ---
pos_index = collections.defaultdict(set)   # (position, string) -> asins
any_index = collections.defaultdict(set)   # string -> asins
for a in prods:
    seq = emitted[a]
    for i, s in enumerate(seq):
        pos_index[(i, s)].add(a); any_index[s].add(a)
pos_uniq = sum(1 for k, v in pos_index.items() if len(v) == 1)
any_uniq = sum(1 for k, v in any_index.items() if len(v) == 1)
print(f"\n[3] POSITIONAL vs ANY-POSITION index")
print(f"    distinct (pos,string) keys: {len(pos_index):7d}  unique: {pos_uniq/len(pos_index):6.1%}")
print(f"    distinct string keys:       {len(any_index):7d}  unique: {any_uniq/len(any_index):6.1%}")
