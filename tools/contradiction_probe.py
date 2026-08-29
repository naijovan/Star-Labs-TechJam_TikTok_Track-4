import sys, random, collections
sys.path.insert(0,'.')
import evaluator.local_evaluator as LE
import submission.agent as A
from submission.agent import Agent
samples=LE.load_jsonl('data/public_set.jsonl'); ids,cats,prods=LE.catalog_index('data/catalog.jsonl')

# ---- inject a REAL contradiction: old_value comes from a DIFFERENT product ----
OBF=LE.behavior_for
def poisoned(scenario, card, rng):
    b=OBF(scenario, card, rng)
    if scenario=="intent_override":
        other=rng.choice(ALLCARDS)                     # a different product's clue
        b["override"]["old_value"]=other
        b["override"]["message"]=f"Actually, ignore my earlier preference. What I need is: {b['override']['new_value']}."
    return b
ALLCARDS=[]
for a,p in list(prods.items())[:4000]:
    c=LE.intent_card(p); ALLCARDS += c['soft_preferences']
ALLCARDS=[x for x in ALLCARDS if len(x)>12][:3000]

def run(label, contradiction, detector):
    LE.behavior_for = poisoned if contradiction else OBF
    A.Agent._parse = DETECT if detector else BASE
    ag=Agent('data/catalog.jsonl')
    r=LE.evaluate(ag,samples,ids,cats,prods)
    o=r['scenario_metrics']['intent_override']
    print(f"{label:46s} {r['recommended_technical_score']:8.5f} | ovr hit={o['hit_rate_at_10']:.3f} mrr={o['mrr']:.3f}")
    LE.behavior_for=OBF; A.Agent._parse=BASE

BASE = A.Agent._parse
def DETECT(self, msg, st, turn):
    """On override, test whether the new clue CONTRADICTS what we already hold.
       If keeping both empties the candidate set, the old clues are wrong -> drop them."""
    pre = list(st["clues"])
    BASE(self, msg, st, turn)
    if "ignore my earlier preference" not in msg and "second thought" not in msg.lower():
        return
    new = [c for c in st["clues"] if c not in pre]
    if not new: return
    base = self.bucket.get(st["cat"]) or set(self.pop)
    both = set(base); onlynew = set(base)
    for c in st["clues"]:
        if c in self.clue_to: both &= self.clue_to[c]
    for c in new:
        if c in self.clue_to: onlynew &= self.clue_to[c]
    if not both and onlynew:            # incompatible -> the old clues must go
        st["clues"] = list(new)
        st["free"] = []

print(f"{'setup':46s} {'score':>8s} | intent_override subset")
print("-"*92)
run("shipped simulator, keep everything (current)", False, False)
run("shipped simulator, + contradiction detector", False, True)
run("POISONED simulator, keep everything", True, False)
run("POISONED simulator, + contradiction detector", True, True)
