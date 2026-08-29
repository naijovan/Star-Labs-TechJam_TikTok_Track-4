import sys, re
sys.path.insert(0,'.')
import evaluator.local_evaluator as LE
from submission.agent import Agent
samples=LE.load_jsonl('data/public_set.jsonl'); ids,cats,prods=LE.catalog_index('data/catalog.jsonl')
agent=Agent('data/catalog.jsonl')
OIM, OCR = LE.initial_message, LE.customer_reply

def rephrase_open(msg):
    m=re.match(r"I'm looking for (.+?), but I'm still exploring\.$", msg)
    if m: return f"I want {m.group(1)}. Just browsing for now."
    m=re.match(r"I'm looking for (.+?)\. A key requirement is: (.+)\.$", msg)
    if m: return f"I need {m.group(1)}. It must have {m.group(2)}."
    m=re.match(r"I'm looking for (.+?)\. (.+)$", msg)
    if m: return f"I need {m.group(1)}. Ideally {m.group(2)}"
    return msg
def rephrase_payout(msg):
    if "what matters is:" in msg:
        h,t=msg.split("what matters is:",1)
        return "What I care about: " + " and ".join(p.strip() for p in t.rstrip('.').split(';')) + "."
    if "Actually, ignore my earlier preference. What I need is:" in msg:
        return "On second thought, forget that. I really want: " + msg.split("What I need is:",1)[1].strip()
    return msg

def patch(open_=False, payout=False):
    def im(s,c,d):
        m=OIM(s,c,d); return rephrase_open(m) if open_ else m
    def cr(s,a,d,b):
        m,bb=OCR(s,a,d,b); return (rephrase_payout(m) if payout else m), bb
    LE.initial_message, LE.customer_reply = im, cr
def go(label, **kw):
    patch(**kw); agent.S={}
    r=LE.evaluate(agent,samples,ids,cats,prods)
    LE.initial_message, LE.customer_reply = OIM, OCR
    print(f"{label:46s} {r['hit_rate_at_10']:7.3f} {r['mrr']:7.3f} {r['mttc']:6.2f} {r['recommended_technical_score']:8.5f}")

print(f"{'simulator wording':46s} {'hit@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'SCORE':>8s}")
print("-"*80)
go("as shipped")
go("OPENING sentence rephrased", open_=True)
go("PAYOUT sentence rephrased", payout=True)
go("BOTH rephrased", open_=True, payout=True)
print("-"*80)
print(f"{'floor probe reference':46s} {0.980:7.3f} {0.544:7.3f} {2.06:6.2f} {0.83199:8.5f}")
