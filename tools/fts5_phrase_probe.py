import json, sqlite3, collections
from evaluator.local_evaluator import evaluate, load_jsonl, catalog_index, coarse_category, intent_card
from starter.agent import _text
FILLER=("Those options are not quite right yet","I don't have an additional preference","I don't have a preference for")
def esc(s): return '"' + s.replace('"','""') + '"'

class FTSAgent:
    def __init__(self, path, backoff="drop_common"):
        self.backoff=backoff
        self.conn=sqlite3.connect(":memory:"); self.hist={}; self.cat={}; self.pop={}
        self.bucket=collections.defaultdict(set); self.df=collections.Counter()
        cur=self.conn.cursor()
        cur.execute("CREATE VIRTUAL TABLE products USING fts5(parent_asin UNINDEXED, title, categories,"
                    " features, details, store, description, tokenize='unicode61 remove_diacritics 2')")
        batch=[]
        for line in open(path,encoding='utf-8'):
            p=json.loads(line); a=str(p['parent_asin'])
            self.pop[a]=p.get('rating_number') or 0
            self.bucket[coarse_category([str(v) for v in p.get('categories') or []])].add(a)
            for s in set(intent_card(p)['hard_constraints']+intent_card(p)['soft_preferences']): self.df[s]+=1
            batch.append((a,_text(p.get('title')),_text(p.get('categories')),_text(p.get('features')),
                          _text(p.get('details')),_text(p.get('store')),_text(p.get('description'))))
            if len(batch)>=1000: cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)",batch); batch.clear()
        if batch: cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)",batch)
        self.conn.commit()
    def reset(self,sid,p): self.hist[sid]=[]; self.cat[sid]=None
    def _clues(self,m):
        if "what matters is:" in m: return [c.strip() for c in m.split("what matters is:",1)[1].rstrip('.').split(';') if c.strip()]
        if "A key requirement is:" in m: return [m.split("A key requirement is:",1)[1].strip().rstrip('.')]
        if "What I need is:" in m: return [m.split("What I need is:",1)[1].strip().rstrip('.')]
        return []
    def respond(self,sid,msg,turn,top_k):
        h=self.hist.setdefault(sid,[])
        if turn==1 and "I'm looking for " in msg:
            c=msg.split("I'm looking for ",1)[1]
            c=c.rsplit(", but I'm still exploring",1)[0].split(". A key requirement is:",1)[0].split(".",1)[0]
            self.cat[sid]=c.strip()
            rest=msg.split(self.cat[sid],1)[1] if self.cat[sid] in msg else ""
            if rest.startswith(". ") and "A key requirement" not in rest:
                v=rest[2:].strip().rstrip('.')
                if v: h.append(v)
        if not any(f in msg for f in FILLER): h.extend(self._clues(msg))
        clues=list(dict.fromkeys(x for x in h if x))
        # RAREST FIRST — drop the most common clue when backing off
        clues.sort(key=lambda c: self.df.get(c, 10**6))
        ids=[]
        for n in range(len(clues),0,-1):
            q=" AND ".join(esc(c) for c in clues[:n])
            try: rows=self.conn.execute("SELECT parent_asin FROM products WHERE products MATCH ? LIMIT 2000",(q,)).fetchall()
            except sqlite3.OperationalError: continue
            if rows: ids=[str(r[0]) for r in rows]; break
        b=self.bucket.get(self.cat.get(sid))
        if b:
            f=[i for i in ids if i in b]
            ids = f if f else (ids or sorted(b,key=lambda a:-self.pop[a]))
        ids=sorted(ids,key=lambda a:-self.pop[a])
        return {"message":"Anything else?","ask_attribute":"other",
                "recommendations":[{"parent_asin":i} for i in ids[:top_k]],
                "usage":{"prompt_tokens":0,"completion_tokens":0}}

samples=load_jsonl('data/public_set.jsonl'); ids,cats,prods=catalog_index('data/catalog.jsonl')
r=evaluate(FTSAgent('data/catalog.jsonl'),samples,ids,cats,prods)
print(f"FTS5 PHRASE-AND, rarest-first back-off + category + popularity")
print(f"  hit@10 {r['hit_rate_at_10']:.3f}  MRR {r['mrr']:.3f}  MTTC {r['mttc']:.2f}  SCORE {r['recommended_technical_score']:.5f}")
print(f"  (previous back-off bug scored 0.87780 | python dict intersection 0.96693)")
