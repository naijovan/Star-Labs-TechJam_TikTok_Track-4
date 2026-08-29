import json, sqlite3, collections, random
from evaluator.local_evaluator import intent_card, coarse_category
from starter.agent import _terms, _text
prods,cards,cats={},{},{}
for line in open('data/catalog.jsonl',encoding='utf-8'):
    p=json.loads(line); a=str(p['parent_asin']); prods[a]=p
    cards[a]=intent_card(p); cats[a]=coarse_category([str(v) for v in p.get('categories') or []])
bkt=collections.defaultdict(set)
for a,c in cats.items(): bkt[c].add(a)
pop=lambda a: prods[a].get('rating_number') or 0

def build(scope):
    con=sqlite3.connect(":memory:"); cur=con.cursor()
    cur.execute("CREATE VIRTUAL TABLE p USING fts5(asin UNINDEXED, body, tokenize='unicode61 remove_diacritics 2')")
    rows=[]
    for a,pr in prods.items():
        if scope=="emitted":
            body=" ".join(cards[a]['hard_constraints']+cards[a]['soft_preferences'])
        else:
            body=" ".join([_text(pr.get('title')),_text(pr.get('features')),_text(pr.get('details')),
                           _text(pr.get('description')),_text(pr.get('categories')),_text(pr.get('store'))])
        rows.append((a,body))
        if len(rows)>=2000: cur.executemany("INSERT INTO p VALUES (?,?)",rows); rows.clear()
    if rows: cur.executemany("INSERT INTO p VALUES (?,?)",rows)
    con.commit(); return con

samples=[json.loads(l) for l in open('data/public_set.jsonl',encoding='utf-8')]
def reword(s,rate,rng):
    w=s.split()
    if len(w)<3: return s
    k=[x for x in w if rng.random()>rate]
    return " ".join(k) if k else w[0]

def run(con,rate,seed=0):
    rng=random.Random(seed); hit=mrr=0
    for s in samples:
        t=str(s['ground_truth']['parent_asin'])
        cl=[reword(c,rate,rng) for c in cards[t]['hard_constraints']+cards[t]['soft_preferences']]
        base=bkt[cats[t]]
        terms=list(dict.fromkeys(_terms(" ".join(cl))))[:60]
        q=" OR ".join(f'"{x}"' for x in terms)
        lst=[]
        if q:
            rows=con.execute("SELECT asin FROM p WHERE p MATCH ? ORDER BY bm25(p,0.0,1.0) LIMIT 3000",(q,)).fetchall()
            lst=[str(r[0]) for r in rows if str(r[0]) in base]
        if not lst: lst=sorted(base,key=lambda a:-pop(a))
        lst=lst[:10]
        if t in lst: hit+=1; mrr+=1/(lst.index(t)+1)
    return hit/len(samples), mrr/len(samples)

print(f"{'FTS5 BM25, indexed over…':34s} {'drop':>5s} {'hit@10':>7s} {'MRR':>7s}")
print("-"*60)
for scope,label in [("full","FULL product text (starter's choice)"),
                    ("emitted","EMITTED clues only (correct scope)")]:
    con=build(scope)
    for r in (0.0,0.35):
        h,m=run(con,r)
        print(f"{label:34s} {int(r*100):>4d}% {h:7.3f} {m:7.3f}")
