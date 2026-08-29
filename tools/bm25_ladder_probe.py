import json, re, sqlite3, collections
from pathlib import Path
from evaluator.local_evaluator import (evaluate, load_jsonl, catalog_index,
                                       coarse_category)
from starter.agent import _terms, _text

FILLER = ("Those options are not quite right yet",
          "I don't have an additional preference",
          "I don't have a preference for")

class BM25Agent:
    """FTS5 + BM25, with switchable fixes."""
    def __init__(self, catalog_path, accumulate=False, ask=False, category=False):
        self.accumulate, self.ask_on, self.category = accumulate, ask, category
        self.conn = sqlite3.connect(":memory:")
        self.hist, self.cat = {}, {}
        cur = self.conn.cursor()
        cur.execute("CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description,"
            " tokenize='unicode61 remove_diacritics 2')")
        self.bucket = collections.defaultdict(set)
        batch=[]
        for line in open(catalog_path, encoding='utf-8'):
            p=json.loads(line); a=str(p['parent_asin'])
            self.bucket[coarse_category([str(v) for v in p.get('categories') or []])].add(a)
            batch.append((a,_text(p.get('title')),_text(p.get('categories')),_text(p.get('features')),
                          _text(p.get('details')),_text(p.get('store')),_text(p.get('description'))))
            if len(batch)>=1000: cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)",batch); batch.clear()
        if batch: cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)",batch)
        self.conn.commit()

    def reset(self, session_id, user_profile):
        self.hist[session_id]=[]; self.cat[session_id]=None

    def respond(self, session_id, user_message, turn, top_k):
        h=self.hist.setdefault(session_id,[])
        if turn==1 and "I'm looking for " in user_message:
            c=user_message.split("I'm looking for ",1)[1]
            c=c.rsplit(", but I'm still exploring",1)[0].split(". A key requirement is:",1)[0].split(".",1)[0]
            self.cat[session_id]=c.strip()
        if self.accumulate:
            if not any(f in user_message for f in FILLER): h.append(user_message)
            query=" ".join(h)
        else:
            query=user_message
        terms=list(dict.fromkeys(_terms(query)))[:40]
        expr=" OR ".join(f'"{t}"' for t in terms)
        recs=[]
        if expr:
            lim = top_k*40 if self.category else top_k
            rows=self.conn.execute("SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products,0.0,6.0,4.0,2.5,2.5,1.5,1.0) LIMIT ?",(expr,lim)).fetchall()
            ids=[str(r[0]) for r in rows]
            if self.category and self.cat.get(session_id) in self.bucket:
                b=self.bucket[self.cat[session_id]]
                ids=[i for i in ids if i in b] or ids
            recs=[{"parent_asin":i} for i in ids[:top_k]]
        return {"message":"Anything else that matters?",
                "ask_attribute":"other" if self.ask_on else None,
                "recommendations":recs,"usage":{"prompt_tokens":0,"completion_tokens":0}}

samples=load_jsonl('data/public_set.jsonl')
ids,cats,prods=catalog_index('data/catalog.jsonl')
print(f"{'FTS5 / BM25 configuration':44s} {'hit@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'SCORE':>8s}")
print("-"*78)
for label,kw in [
    ("starter as shipped (stateless, no ask)", dict()),
    ("+ always ask 'other'",                   dict(ask=True)),
    ("+ accumulate history",                   dict(ask=True, accumulate=True)),
    ("+ category filter",                      dict(ask=True, accumulate=True, category=True)),
]:
    r=evaluate(BM25Agent('data/catalog.jsonl',**kw), samples, ids, cats, prods)
    print(f"{label:44s} {r['hit_rate_at_10']:7.3f} {r['mrr']:7.3f} {r['mttc']:6.2f} {r['recommended_technical_score']:8.5f}")
print("-"*78)
print(f"{'exact-intersection agent (our design)':44s} {1.000:7.3f} {0.979:7.3f} {2.33:6.2f} {0.96693:8.5f}")
