"""Adversarial verification. Corruption is a PURE FUNCTION of the message text,
so it is identical regardless of how many turns the agent takes."""
import sys, random, hashlib
sys.path.insert(0,'.')
import evaluator.local_evaluator as LE
from submission.agent import Agent
ORIG_CR, ORIG_IM = LE.customer_reply, LE.initial_message

def reword(s, rate):
    if rate<=0: return s
    w=s.split()
    if len(w)<3: return s
    rng=random.Random(hashlib.md5(s.encode()).hexdigest())   # deterministic per string
    k=[x for x in w if rng.random()>rate]
    return " ".join(k) if k else w[0]

def patch(drop=0.0, cat_noise=False):
    def cr(sample, ask, disclosed, boundary):
        msg,b = ORIG_CR(sample, ask, disclosed, boundary)
        if drop and "what matters is:" in msg:
            h,t = msg.split("what matters is:",1)
            msg = h+"what matters is: "+"; ".join(reword(p.strip(),drop) for p in t.rstrip('.').split(';'))+"."
        return msg,b
    def im(sample, category, disclosed):
        if cat_noise:
            w=category.split()
            category=" ".join(reversed(w)) if len(w)>1 else (category[:-1] if len(category)>4 else category)
        msg = ORIG_IM(sample, category, disclosed)
        if drop and "A key requirement is:" in msg:
            h,t = msg.split("A key requirement is:",1)
            msg = h+"A key requirement is: "+reword(t.strip().rstrip('.'),drop)+"."
        return msg
    LE.customer_reply, LE.initial_message = cr, im

def main():
    samples=LE.load_jsonl('data/public_set.jsonl')
    ids,cats,prods=LE.catalog_index('data/catalog.jsonl')
    agent=Agent('data/catalog.jsonl')
    def go(label, **kw):
        patch(**kw); agent.S={}
        r=LE.evaluate(agent, samples, ids, cats, prods)
        print(f"{label:44s} {r['hit_rate_at_10']:7.3f} {r['mrr']:7.3f} {r['mttc']:6.2f} "
              f"{r['recommended_technical_score']:8.5f}")
        LE.customer_reply, LE.initial_message = ORIG_CR, ORIG_IM
        return r
    print(f"{'scenario':44s} {'hit@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'SCORE':>8s}")
    print("-"*78)
    go("CLEAN (as the organizer ships it)")
    go("clues: 20% of words dropped", drop=0.20)
    go("clues: 35% of words dropped", drop=0.35)
    go("clues: 50% of words dropped", drop=0.50)
    go("CATEGORY words reordered", cat_noise=True)
    go("category reordered + 35% word drop", drop=0.35, cat_noise=True)
    print("-"*78)
    print(f"{'reference: weak BM25 baseline':44s} {0.125:7.3f} {0.068:7.3f} {9.81:6.2f} {0.10671:8.5f}")
    print(f"{'reference: floor probe':44s} {0.980:7.3f} {0.544:7.3f} {2.06:6.2f} {0.83199:8.5f}")

if __name__=="__main__": main()
