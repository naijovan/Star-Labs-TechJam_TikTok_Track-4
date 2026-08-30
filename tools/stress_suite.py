"""500-session private stress suite - four distribution-shift axes.

The public 200 cannot tell us how the agent behaves if the organizer authors
cards, paraphrases messages, or samples colder targets. This suite builds 125
sessions on each of those axes, using the evaluator's OWN materialisation
machinery, and runs the shipped agent plus its dormant variants against them.

  AXIS A  synonym paraphrase   constraints reworded, not just word-dropped
  AXIS B  drifting shoppers    the retracted preference is GENUINELY FALSE
                               (drawn from a different product's card)
  AXIS C  cold targets         uniform + least-popular-half resampling
  AXIS D  authored cards       a deliberately different card generator, so
                               disclosed strings mismatch our replica index

  gen                          write the four JSONL files (deterministic)
  run VARIANT                  evaluate VARIANT on control + all four axes
                               VARIANT in: baseline | timedecay | noproof |
                                           dense | ce

Guardrail: these sessions test OUR GUESSES about the private set. Only effects
that hold across every axis (and cost nothing on the control) are trusted.
"""
from __future__ import annotations

import json, random, re, sys
from pathlib import Path

sys.path.insert(0, '.')
import evaluator.local_evaluator as LE

OUT = Path("/private/tmp/claude-501/-Users-naijovan-Projects-techjam-techjam-conversational-search/7e90a565-642c-4c2c-9bef-cda87e0661b9/scratchpad/stress")
SEED = 20260830
MIX = [("buying", 50), ("browsing", 50), ("intent_override", 19), ("boundary", 6)]

# ---------------------------------------------------------------- generation
def _load():
    samples = LE.load_jsonl('data/public_set.jsonl')
    ids, cats, prods = LE.catalog_index('data/catalog.jsonl')
    public_targets = {str(s['ground_truth']['parent_asin']) for s in samples}
    profiles = [s['user_profile'] for s in samples]
    return samples, ids, cats, prods, public_targets, profiles

def _mixed(rng, pool, n_each=MIX):
    out = []
    for scen, n in n_each:
        for _ in range(n):
            out.append((rng.choice(pool), scen))
    return out

def gen():
    OUT.mkdir(parents=True, exist_ok=True)
    _, ids, cats, prods, public_targets, profiles = _load()
    rng = random.Random(SEED)
    # plausible 5-core-ish pool for axes A/B/D: rating_number >= 5
    core = sorted(a for a, p in prods.items()
                  if (p.get('rating_number') or 0) >= 5 and a not in public_targets)
    everything = sorted(a for a in prods if a not in public_targets)
    cold = sorted(a for a in everything if (prods[a].get('rating_number') or 0) <= 12)

    def base(sid, asin, scen):
        return {"sample_id": sid, "scenario_type": scen,
                "ground_truth": {"parent_asin": asin},
                "user_profile": rng.choice(profiles)}

    # AXIS A - plain sessions; the paraphrase is applied at run time (messages)
    a = [base(f"stress_A_{i:04d}", asin, scen)
         for i, (asin, scen) in enumerate(_mixed(rng, core))]

    # AXIS B - all overrides, old_value drawn from a DIFFERENT product's card
    b = []
    for i in range(125):
        asin = rng.choice(core)
        card = LE.intent_card(prods[asin])
        foreign = rng.choice(core)
        while foreign == asin: foreign = rng.choice(core)
        fcard = LE.intent_card(prods[foreign])
        pool = [*fcard['soft_preferences'], *fcard['hard_constraints']]
        old = rng.choice([v for v in pool if v] or ["I prefer a different style."])
        new = card['hard_constraints'][0] if card['hard_constraints'] else "Please prioritize the target requirements."
        s = base(f"stress_B_{i:04d}", asin, "intent_override")
        s["intent_card"] = card
        s["behavior"] = {"scenario_type": "intent_override",
                         "override": {"turn": rng.choice([3, 4]), "old_value": old,
                                      "new_value": new,
                                      "message": f"Actually, ignore my earlier preference. What I need is: {new}."}}
        b.append(s)

    # AXIS C - cold targets, no corruption: 63 uniform, 62 least-popular half
    c = []
    picks = [rng.choice(everything) for _ in range(63)] + [rng.choice(cold) for _ in range(62)]
    scens = [s for s, n in MIX for _ in range(n)]
    rng.shuffle(scens)
    for i, (asin, scen) in enumerate(zip(picks, scens)):
        c.append(base(f"stress_C_{i:04d}", asin, scen))

    # AXIS D - authored cards from a deliberately different generator
    def authored_card(p, r):
        title = LE._clean_constraint(str(p.get('title') or 'product'), 120)
        cand = [*LE._flatten_values(p.get('features')), *LE._flatten_values(p.get('details'))]
        corpus = LE.searchable_text(p)
        m = LE.MATERIAL_RE.search(corpus); col = LE.COLOR_RE.search(corpus)
        if m: cand.append(m.group(1).lower())            # appended, not inserted
        if col: cand.append(f"color: {col.group(1).lower()}")
        r.shuffle(cand)                                   # different order
        cleaned = list(dict.fromkeys(LE._clean_constraint(x, 120) for x in cand if LE._clean_constraint(x, 120)))
        if not cleaned: cleaned = [title]
        return {"target_category": title,
                "hard_constraints": cleaned[:2],
                "soft_preferences": cleaned[2:4] or cleaned[:1]}
    d = []
    for i, (asin, scen) in enumerate(_mixed(rng, core)):
        r2 = random.Random(f"D{i}")
        card = authored_card(prods[asin], r2)
        s = base(f"stress_D_{i:04d}", asin, scen)
        s["intent_card"] = card
        s["behavior"] = LE.behavior_for(scen, card, r2)
        d.append(s)

    for name, rows in (("A_paraphrase", a), ("B_drift", b), ("C_cold", c), ("D_authored", d)):
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"wrote {path.name}: {len(rows)} sessions")

# ------------------------------------------------------- synonym paraphrase
# Applied to the CONSTRAINT payloads of messages at run time (axis A only).
# Word-boundary regexes; deterministic. Colors kept (people say colors as-is);
# materials, care phrases and closure/feature phrases are reworded - those are
# exactly the strings the exact-match core lives on.
SYN = [
    (r"\b100% Cotton\b",   "pure natural cotton"),
    (r"\b100% Polyester\b","fully synthetic fabric"),
    (r"\b100% Leather\b",  "real genuine hide"),
    (r"\bcotton\b",        "natural cotton fiber"),
    (r"\bpolyester\b",     "synthetic fiber"),
    (r"\bleather\b",       "genuine hide"),
    (r"\bnylon\b",         "polyamide weave"),
    (r"\bwool\b",          "woolen knit"),
    (r"\bspandex\b",       "stretchy elastane"),
    (r"\brayon\b",         "viscose blend"),
    (r"\bMachine Wash\b",  "fine in the washing machine"),
    (r"\bmachine wash(able)?\b", "washing-machine safe"),
    (r"\bHand Wash Only\b","needs washing by hand"),
    (r"\bImported\b",      "made overseas"),
    (r"\bMade in USA\b",   "manufactured in America"),
    (r"\bPull On closure\b", "just slips on, no fasteners"),
    (r"\bZipper closure\b",  "closes with a zip"),
    (r"\bButton closure\b",  "closes with buttons"),
    (r"\bLace-up closure\b", "ties up with laces"),
    (r"\bHook and Loop closure\b", "velcro-style fastening"),
    (r"\bBuckle closure\b",  "fastens with a buckle"),
    (r"\bElastic closure\b", "elasticated fit"),
    (r"\bRubber sole\b",     "grippy rubber bottom"),
    (r"\bSynthetic sole\b",  "man-made outsole"),
    (r"\bDry Clean Only\b",  "dry cleaning required"),
    (r"\bbreathable\b",      "airy"),
    (r"\blightweight\b",     "light and easy to wear"),
    (r"\bwaterproof\b",      "keeps water out"),
    (r"\bmoisture[- ]wicking\b", "sweat-pulling"),
    (r"\bpockets?\b",        "storage pouches"),
    (r"\badjustable\b",      "size-tunable"),
    (r"\bdurable\b",         "built to last"),
]
SYN = [(re.compile(p, re.I), s) for p, s in SYN]

def _synonymize(text):
    for rx, sub in SYN:
        text = rx.sub(sub, text)
    return text

ORIG_CR, ORIG_IM = LE.customer_reply, LE.initial_message
def patch_paraphrase():
    def cr(sample, ask, disclosed, boundary):
        msg, bnd = ORIG_CR(sample, ask, disclosed, boundary)
        if "what matters is:" in msg:
            h, t = msg.split("what matters is:", 1)
            msg = h + "what matters is: " + _synonymize(t.rstrip('.')).strip() + "."
        return msg, bnd
    def im(sample, category, disclosed):
        msg = ORIG_IM(sample, category, disclosed)
        if "A key requirement is:" in msg:
            h, t = msg.split("A key requirement is:", 1)
            msg = h + "A key requirement is: " + _synonymize(t.rstrip('.')).strip() + "."
        return msg
    LE.customer_reply, LE.initial_message = cr, im
def unpatch():
    LE.customer_reply, LE.initial_message = ORIG_CR, ORIG_IM

# -------------------------------------------------------------- variants
def make_agent(variant):
    import submission.config as C
    from submission.agent import Agent, terms
    if variant == "baseline":
        return Agent('data/catalog.jsonl')
    if variant == "noproof":
        C.OVERRIDE_DETECT = False
        ag = Agent('data/catalog.jsonl'); C.OVERRIDE_DETECT = True
        ag._tag = "noproof"; return ag
    if variant == "dense":
        C.USE_DENSE = True
        ag = Agent('data/catalog.jsonl'); C.USE_DENSE = False
        if not ag.dense: raise RuntimeError("dense stack not importable")
        return ag
    if variant == "timedecay":
        DECAY_AGE = 3
        class TimeDecay(Agent):
            """Slots older than DECAY_AGE turns stop constraining retrieval."""
            def _parse(self, msg, st, turn):
                before = set(st["clues"])
                super()._parse(msg, st, turn)
                st.setdefault("_born", {})
                for c in st["clues"]:
                    if c not in before: st["_born"].setdefault(c, turn)
                st["_now"] = turn
            def _retrieve(self, st):
                born, now = st.get("_born", {}), st.get("_now", 1)
                full = st["clues"]
                st["clues"] = [c for c in full if now - born.get(c, now) <= DECAY_AGE]
                try:     return super()._retrieve(st)
                finally: st["clues"] = full
        return TimeDecay('data/catalog.jsonl')
    if variant == "densefull":
        # Full-text embeddings: the shipped dense lane embeds each product's
        # EMITTED CONSTRAINTS (what it would say); this variant embeds the raw
        # searchable text, where "soft / plush / cozy"-style vocabulary lives.
        # Targets the attribute-inference gap, not just the synonym gap.
        import json as _j
        import submission.agent as A
        C.USE_DENSE = True
        ag = Agent('data/catalog.jsonl'); C.USE_DENSE = False
        if not ag.dense: raise RuntimeError("dense stack not importable")
        pairs = []
        for line in open('data/catalog.jsonl', encoding='utf-8'):
            pr = _j.loads(line)
            pairs.append((str(pr["parent_asin"]), A._searchable(pr)[:400]))
        ag.asins = [a for a, _ in pairs]
        ag.E = ag.model.encode([t for _, t in pairs], batch_size=256,
                               convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=False)
        ag.didx = {a: i for i, a in enumerate(ag.asins)}
        return ag
    if variant == "ce2":
        # Top-tier open reranker (bge-reranker-v2-m3), same integration as "ce".
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512)
        class CE2(Agent):
            def __init__(self, path):
                super().__init__(path)
                self.clues_of = {}
                for cstr, asins in self.clue_to.items():
                    for x in asins: self.clues_of.setdefault(x, []).append(cstr)
            def _retrieve(self, st):
                ranked, nc, route = super()._retrieve(st)
                if route in ("bm25", "weak", "floor") and len(ranked) > 1 and st["clues"]:
                    head = ranked[:20]
                    q = " ; ".join(st["clues"])[:400]
                    pairs = [(q, " ; ".join(sorted(self.clues_of.get(x, [""]))[:6])[:400]) for x in head]
                    sc = model.predict(pairs, show_progress_bar=False)
                    order = sorted(zip(head, sc), key=lambda t: (-float(t[1]), -self.pop[t[0]], t[0]))
                    ranked = [x for x, _ in order] + ranked[20:]
                return ranked, nc, route
        return CE2('data/catalog.jsonl')
    if variant == "ce":
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        class CERank(Agent):
            """LLM-semantic-ranking stand-in: rerank the ambiguous head."""
            def __init__(self, path):
                super().__init__(path)
                self.clues_of = {}
                for cstr, asins in self.clue_to.items():
                    for x in asins: self.clues_of.setdefault(x, []).append(cstr)
            def _retrieve(self, st):
                ranked, nc, route = super()._retrieve(st)
                if route in ("bm25", "weak", "floor") and len(ranked) > 1 and st["clues"]:
                    head = ranked[:20]
                    q = " ; ".join(st["clues"])[:400]
                    pairs = [(q, " ; ".join(sorted(self.clues_of.get(x, ["" ]))[:6])[:400]) for x in head]
                    sc = model.predict(pairs, show_progress_bar=False)
                    order = sorted(zip(head, sc), key=lambda t: (-float(t[1]), -self.pop[t[0]], t[0]))
                    ranked = [x for x, _ in order] + ranked[20:]
                return ranked, nc, route
        return CERank('data/catalog.jsonl')
    raise SystemExit(f"unknown variant {variant}")

# ------------------------------------------------------------------ runner
def run(variant):
    samples_pub, ids, cats, prods, _, _ = _load()
    agent = make_agent(variant)
    sets = [("control_200", samples_pub, False)] + [
        (n, LE.load_jsonl(OUT / f"{n}.jsonl"), n == "A_paraphrase")
        for n in ("A_paraphrase", "B_drift", "C_cold", "D_authored")]
    print(f"{'variant':10s} {'dataset':14s} {'n':>4s} {'hit@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'SCORE':>8s}", flush=True)
    for name, smp, para in sets:
        if para: patch_paraphrase()
        agent.S = {}
        r = LE.evaluate(agent, smp, ids, cats, prods)
        if para: unpatch()
        print(f"{variant:10s} {name:14s} {len(smp):>4d} {r['hit_rate_at_10']:>7.3f} "
              f"{r['mrr']:>7.4f} {r['mttc']:>6.2f} {r['recommended_technical_score']:>8.5f}", flush=True)

if __name__ == "__main__":
    if sys.argv[1:] == ["gen"]: gen()
    elif len(sys.argv) == 3 and sys.argv[1] == "run": run(sys.argv[2])
    else: print(__doc__)
