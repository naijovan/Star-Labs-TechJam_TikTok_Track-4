"""Contract smoke test. Run BEFORE scoring — catches integration faults in seconds
instead of after a 5-minute evaluation.

    PYTHONPATH=. python3 tools/smoke_test.py peers.marcus.agent
"""
import sys, importlib, traceback, json
sys.path.insert(0,'.')

ALLOWED={"category","material","color","size","style","brand","budget","feature","use_case","other"}

def check(modpath, catalog='data/catalog.jsonl'):
    ok=True
    def bad(m): 
        nonlocal ok; ok=False; print(f"   FAIL  {m}")
    def good(m): print(f"   ok    {m}")
    print(f"\n== {modpath} ==")
    try:
        mod=importlib.import_module(modpath)
    except Exception as e:
        print(f"   FAIL  import: {type(e).__name__}: {e}"); return False
    good("imports")
    if not hasattr(mod,"Agent"): bad("no class named Agent"); return False
    try:
        ag=mod.Agent(catalog)                    # MUST accept a positional path
    except TypeError as e:
        bad(f"Agent(catalog_path) rejected: {e}"); return False
    except Exception as e:
        bad(f"__init__ raised: {type(e).__name__}: {e}"); traceback.print_exc(limit=2); return False
    good("Agent(catalog_path) constructs")
    ids={json.loads(l)['parent_asin'] for l in open(catalog,encoding='utf-8')}
    try: ag.reset("smoke", {"preference_tags":["fit"],"rating_style":"usually positive"})
    except Exception as e: bad(f"reset raised: {e}"); return False
    good("reset accepts a profile")
    msgs=["I'm looking for Basketball Men, but I'm still exploring.",
          "For that, what matters is: polyester; 100% Polyester.",
          "I don't have a preference for other; please use your judgment.",
          "Actually, ignore my earlier preference. What I need is: cotton.",
          ""]
    for i,m in enumerate(msgs,1):
        try: r=ag.respond("smoke", m, i, 10)
        except Exception as e:
            bad(f"respond raised on turn {i}: {type(e).__name__}: {e}"); return False
        if not isinstance(r,dict): bad(f"turn {i}: response is not a dict"); return False
        if not isinstance(r.get("message"),str): bad(f"turn {i}: message is not a str -> WHOLE RESPONSE DISCARDED")
        a=r.get("ask_attribute")
        if a is not None and a not in ALLOWED: bad(f"turn {i}: ask_attribute {a!r} not in the enum")
        recs=r.get("recommendations")
        if not isinstance(recs,list): bad(f"turn {i}: recommendations is not a list")
        else:
            vals=[x.get("parent_asin") if isinstance(x,dict) else x for x in recs]
            if len(vals)!=len(set(vals)): bad(f"turn {i}: duplicate IDs")
            unk=[v for v in vals if v not in ids]
            if unk: bad(f"turn {i}: {len(unk)} IDs not in the catalogue (silently dropped)")
    good("respond survives all 5 message shapes")
    try:
        ag.reset("smoke2", {})
        s=ag.respond("smoke2","I'm looking for Athletic Walking, but I'm still exploring.",1,10)
        ag.reset("smoke", {})
        s2=ag.respond("smoke","I'm looking for Athletic Walking, but I'm still exploring.",1,10)
        if [x for x in s['recommendations']]!=[x for x in s2['recommendations']]:
            bad("reset() does not fully clear state -- one Agent serves all 800 sessions")
        else: good("reset() clears state")
    except Exception as e: bad(f"state check raised: {e}")
    return ok

if __name__=="__main__":
    mods=sys.argv[1:] or ["submission.agent"]
    results={m:check(m) for m in mods}
    print("\n" + "="*50)
    for m,v in results.items(): print(f"  {'PASS' if v else 'FAIL'}  {m}")
    sys.exit(0 if all(results.values()) else 1)
