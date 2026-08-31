"""1,000-session turn-1 singleton suite — probing the only legitimate MTTC lever.

The shipped agent converts every public session by turn 2 (MTTC 2.00). The one
remaining efficiency headroom is turn-1 conversion, which is only safe when the
turn-1 pool is PROVABLY size 1. Three shapes, censused over the full catalog
(public targets excluded):

  T1_featstring    buying: no MATERIAL_RE/COLOR_RE hit in the target's text, so
                   hard_constraints[0] is a raw feature/details string, AND
                   bucket ∩ emitters(hard[0]) == {target}.
                   Census: 11,563 no-regex buying targets; 5,330 singletons —
                   the "unique feature string" assumption holds only ~46%.
  T2_rarematerial  buying: hard[0] IS a material/color word, but it is rare
                   enough in a small enough bucket that the intersection is
                   still a singleton. Census: 5,711.
  T3_lonecategory  browsing: the leaked leaf category's bucket holds exactly
                   one product. Census: 234 — all included.

All sessions are CLEAN official-shape — no message patching. The adversarial
part is target selection only. Perfect behavior on this suite is
hit 1.000 · MRR 1.0000 · MTTC 1.00 on every shape.

  gen   write the three JSONL files (deterministic, seed 20260831)
  run   evaluate submission.agent; per-shape metrics + %converted-on-turn-1
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
sys.path.insert(0, '.')
import evaluator.local_evaluator as LE

OUT = Path("/private/tmp/claude-501/-Users-naijovan-Projects-techjam-techjam-conversational-search/7e90a565-642c-4c2c-9bef-cda87e0661b9/scratchpad/turn1_suite")
SEED = 20260831
SHAPES = ("T1_featstring", "T2_rarematerial", "T3_lonecategory")


def _census():
    samples = LE.load_jsonl('data/public_set.jsonl')
    public = {str(s['ground_truth']['parent_asin']) for s in samples}
    ids, cats, prods = LE.catalog_index('data/catalog.jsonl')
    bucket, emit, catof, regexfree, hard0 = {}, {}, {}, {}, {}
    for a, p in prods.items():
        cat = LE.coarse_category([str(v) for v in p.get("categories") or []])
        catof[a] = cat
        bucket.setdefault(cat, set()).add(a)
        corpus = LE.searchable_text(p)
        regexfree[a] = not (LE.MATERIAL_RE.search(corpus) or LE.COLOR_RE.search(corpus))
        c = LE.intent_card(p)
        hard0[a] = c["hard_constraints"][0] if c["hard_constraints"] else None
        for s in dict.fromkeys([*c["hard_constraints"], *c["soft_preferences"]]):
            emit.setdefault(s, set()).add(a)
    t1, t2, t3 = [], [], []
    for a in sorted(prods):
        if a in public or hard0[a] is None:
            continue
        singleton = (bucket[catof[a]] & emit.get(hard0[a], set())) == {a}
        if regexfree[a] and singleton:
            t1.append(a)
        elif (not regexfree[a]) and singleton:
            t2.append(a)
        if len(bucket[catof[a]]) == 1:
            t3.append(a)
    return t1, t2, t3, samples


def gen():
    OUT.mkdir(parents=True, exist_ok=True)
    t1, t2, t3, samples = _census()
    profiles = [s['user_profile'] for s in samples]
    rng = random.Random(SEED)
    picks = {
        "T1_featstring":   [(a, "buying")   for a in rng.sample(t1, 383)],
        "T2_rarematerial": [(a, "buying")   for a in rng.sample(t2, 383)],
        "T3_lonecategory": [(a, "browsing") for a in t3],   # all 234
    }
    for shape, rows in picks.items():
        out = [{"sample_id": f"t1s_{shape[:2]}_{i:04d}", "scenario_type": scen,
                "ground_truth": {"parent_asin": a}, "user_profile": rng.choice(profiles)}
               for i, (a, scen) in enumerate(rows)]
        (OUT / f"{shape}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
        print(f"wrote {shape}: {len(out)}")


def run():
    from submission.agent import Agent
    ids, cats, prods = LE.catalog_index('data/catalog.jsonl')
    agent = Agent('data/catalog.jsonl')
    print(f"{'shape':16s} {'n':>5s} {'hit@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'SCORE':>8s} {'%turn-1':>8s}", flush=True)
    tot_n = tot_w = tot_t1 = 0.0
    for shape in SHAPES:
        smp = LE.load_jsonl(OUT / f"{shape}.jsonl")
        agent.S = {}
        r = LE.evaluate(agent, smp, ids, cats, prods)
        n_t1 = sum(1 for s in r["sessions"] if s["first_hit_turn"] == 1)
        print(f"{shape:16s} {len(smp):>5d} {r['hit_rate_at_10']:>7.3f} {r['mrr']:>7.4f} "
              f"{r['mttc']:>6.2f} {r['recommended_technical_score']:>8.5f} {100*n_t1/len(smp):>7.1f}%", flush=True)
        tot_n += len(smp); tot_w += r['recommended_technical_score'] * len(smp); tot_t1 += n_t1
    print(f"{'WEIGHTED-1000':16s} {int(tot_n):>5d} {'':>7s} {'':>7s} {'':>6s} "
          f"{tot_w/tot_n:>8.5f} {100*tot_t1/tot_n:>7.1f}%", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["gen"]: gen()
    elif sys.argv[1:] == ["run"]: run()
    else: print(__doc__)
