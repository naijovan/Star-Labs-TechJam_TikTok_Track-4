"""Run several agents through ONE evaluator, on one machine, twice each.

    PYTHONPATH=. python3 tools/bakeoff.py ours=submission.agent marcus=peers.marcus.agent

Each entry is  name=module_path  where the module exposes a class `Agent` whose
__init__ accepts a positional catalogue path. Every agent sees identical sessions,
identical corruption, and is run twice so nondeterminism is caught rather than
mistaken for a difference.
"""
import sys, importlib, traceback
sys.path.insert(0, '.')
import evaluator.local_evaluator as LE
import tools.verify_agent as V

CONDITIONS = [("clean", {}),
              ("35% word drop", dict(drop=0.35)),
              ("category scrambled", dict(cat_noise=True))]

def load(spec):
    name, _, mod = spec.partition("=")
    return name, importlib.import_module(mod).Agent

def score(cls, samples, ids, cats, prods, kw):
    V.patch(**kw)
    try:
        ag = cls('data/catalog.jsonl')
        r = LE.evaluate(ag, samples, ids, cats, prods)
        return r
    finally:
        LE.customer_reply, LE.initial_message = V.ORIG_CR, V.ORIG_IM

def main(specs):
    samples = LE.load_jsonl('data/public_set.jsonl')
    ids, cats, prods = LE.catalog_index('data/catalog.jsonl')
    print(f"one harness · {len(samples)} identical sessions · every agent run twice\n")
    hdr = f"{'agent':16s} {'det?':>5s} " + " ".join(f"{n:>19s}" for n, _ in CONDITIONS)
    print(hdr); print("-" * len(hdr))
    detail = {}
    for spec in specs:
        try:
            name, cls = load(spec)
        except Exception as e:
            print(f"{spec:16s}  FAILED TO IMPORT: {type(e).__name__}: {e}"); continue
        row, det, first = [], True, None
        for label, kw in CONDITIONS:
            try:
                a = score(cls, samples, ids, cats, prods, kw)
                b = score(cls, samples, ids, cats, prods, kw)
            except Exception:
                row.append(None); traceback.print_exc(limit=2); continue
            s1, s2 = a['recommended_technical_score'], b['recommended_technical_score']
            if abs(s1 - s2) > 1e-9: det = False
            row.append(s1)
            if label == "clean": first = a
        detail[name] = first
        cells = " ".join((f"{v:19.5f}" if v is not None else f"{'ERROR':>19s}") for v in row)
        print(f"{name:16s} {('yes' if det else 'NO!'):>5s} {cells}")
    print("\nper-scenario, clean run")
    print(f"{'agent':16s} " + " ".join(f"{k:>17s}" for k in ("buying","browsing","boundary","intent_override")))
    for name, r in detail.items():
        if not r: continue
        m = r['scenario_metrics']
        print(f"{name:16s} " + " ".join(f"{m[k]['mrr']:17.3f}" for k in
              ("buying","browsing","boundary","intent_override")))
    print("\n'det?' = same score twice. A 'NO!' means that agent's number cannot be trusted")
    print("and must be fixed before any comparison is meaningful.")

if __name__ == "__main__":
    args = sys.argv[1:] or ["ours=submission.agent", "starter=starter.agent"]
    main(args)
