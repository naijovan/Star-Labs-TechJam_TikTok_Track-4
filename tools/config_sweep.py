import sys, importlib, random
sys.path.insert(0,'.')
import evaluator.local_evaluator as LE
import submission.config as C
from submission.agent import Agent
samples=LE.load_jsonl('data/public_set.jsonl'); ids,cats,prods=LE.catalog_index('data/catalog.jsonl')
agent=Agent('data/catalog.jsonl')
ORIG_CR=LE.customer_reply
def reword(s,r,rng):
    w=s.split()
    if len(w)<3: return s
    k=[x for x in w if rng.random()>r]
    return " ".join(k) if k else w[0]
def patch(drop,seed=0):
    rng=random.Random(seed)
    def cr(sm,ask,d,b):
        m,bb=ORIG_CR(sm,ask,d,b)
        if drop and "what matters is:" in m:
            h,t=m.split("what matters is:",1)
            m=h+"what matters is: "+"; ".join(reword(p.strip(),drop,rng) for p in t.rstrip('.').split(';'))+"."
        return m,bb
    LE.customer_reply=cr
def score(drop=0.0):
    patch(drop); agent.S={}
    r=LE.evaluate(agent,samples,ids,cats,prods); LE.customer_reply=ORIG_CR
    return r['recommended_technical_score'], r['mrr'], r['mttc']
print(f"{'EMIT':>5s} {'HOLD':>5s} | {'clean':>8s} {'MRR':>6s} {'MTTC':>6s} | {'35% drop':>9s}")
print("-"*52)
best=None
for e in (1,2,3,5):
    for h in (2,3,4):
        C.EMIT_THRESHOLD=e; C.HOLD_UNTIL=h
        s,m,t=score(0.0); s2,_,_=score(0.35)
        flag=""
        if best is None or s+s2>best[0]: best=(s+s2,e,h); flag=" <-"
        print(f"{e:>5d} {h:>5d} | {s:8.5f} {m:6.3f} {t:6.2f} | {s2:9.5f}{flag}")
print("-"*52)
print(f"best combined: EMIT_THRESHOLD={best[1]} HOLD_UNTIL={best[2]}")
