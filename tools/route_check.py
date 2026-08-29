import sys, json, collections
sys.path.insert(0,'.')
import evaluator.local_evaluator as LE

samples=LE.load_jsonl('data/public_set.jsonl')
ids,cats,prods=LE.catalog_index('data/catalog.jsonl')

def classify(msg):
    """Exactly what submission/agent.py does on turn 1."""
    if "I'm looking for " not in msg: return "UNPARSEABLE"
    rest=msg.split("I'm looking for ",1)[1]
    if ", but I'm still exploring" in rest: return "browsing_or_boundary"
    if ". A key requirement is:" in rest:   return "buying"
    if ". " in rest:                        return "intent_override"
    return "AMBIGUOUS"

conf=collections.Counter(); bad=[]
edge=collections.Counter()
for s in samples:
    tgt=str(s['ground_truth']['parent_asin'])
    card,beh = LE.materialize_hidden_fields(s, prods)
    eff={**s,'intent_card':card,'behavior':beh}
    cat=LE.coarse_category(cats.get(tgt,[]))
    msg=LE.initial_message(eff, cat, set())
    got=classify(msg); truth=s['scenario_type']
    expect = {"buying":"buying","intent_override":"intent_override",
              "browsing":"browsing_or_boundary","boundary":"browsing_or_boundary"}[truth]
    conf[(truth,got)]+=1
    if got!=expect: bad.append((s['sample_id'],truth,got,msg[:70]))
    if not card['hard_constraints']: edge['no hard_constraints']+=1
    if not card['soft_preferences']: edge['no soft_preferences']+=1
    if "." in cat or "," in cat: edge['category contains . or ,']+=1

print("TURN-1 ROUTING ACCURACY over all 200 public sessions")
print("-"*72)
for (truth,got),n in sorted(conf.items()):
    ok = "ok " if got=={"buying":"buying","intent_override":"intent_override",
         "browsing":"browsing_or_boundary","boundary":"browsing_or_boundary"}[truth] else "BAD"
    print(f"  {ok}  true={truth:16s} -> detected={got:22s} n={n}")
print(f"\n  misroutes: {len(bad)}/200")
for b in bad[:5]: print("   ",b)
print(f"\nedge cases in the 200: {dict(edge) or 'none'}")

# how much does losing override detection cost?
from submission.agent import Agent
import submission.agent as A
agent=Agent('data/catalog.jsonl')
r=LE.evaluate(agent,samples,ids,cats,prods)
base=r['recommended_technical_score']
orig=A.Agent._parse
def noflag(self,msg,st,turn):
    orig(self,msg,st,turn); st["is_override"]=False      # pretend we never detect it
A.Agent._parse=noflag; agent.S={}
r2=LE.evaluate(agent,samples,ids,cats,prods)
A.Agent._parse=orig
print(f"\nCOST OF FAILING TO DETECT THE MIND-CHANGE AT TURN 1")
print(f"  with override gate    {base:.5f}")
print(f"  without override gate {r2['recommended_technical_score']:.5f}   ({r2['recommended_technical_score']-base:+.5f})")
print(f"  intent_override subset: with={r['scenario_metrics']['intent_override']['mrr']:.3f}  "
      f"without={r2['scenario_metrics']['intent_override']['mrr']:.3f}")
