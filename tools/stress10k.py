"""10,000-session generalisation suite - five axes the agent was never built for.

Every transform vocabulary here is DISJOINT from the agent's SYN/CANON tables
and from tools/stress_suite.py's generators: this suite tests the agent, not
our own training data.

  E  realistic paraphrase   held-out rewordings + light word loss
  F  true product switch    turns 1..override describe product A (its category,
                            its facts); the customer then switches to target B.
                            50% cross-category, 50% same-bucket.
  G  vague vibes            constraints replaced by loose attribute talk
  H  moderate chaos         paraphrase + damaged category + vague clues
  I  cold x paraphrase      uniform targets (median 12 reviews) + rewording

  gen         write the five 2,000-session JSONL files (deterministic)
  run V       V in baseline | dense   (dense = set TECHJAM_DENSE before python)
"""
from __future__ import annotations
import hashlib, json, random, re, sys
from pathlib import Path
sys.path.insert(0, '.')
import evaluator.local_evaluator as LE

OUT = Path("/private/tmp/claude-501/-Users-naijovan-Projects-techjam-techjam-conversational-search/7e90a565-642c-4c2c-9bef-cda87e0661b9/scratchpad/stress10k")
SEED = 20260831
MIX = [("buying", 800), ("browsing", 800), ("intent_override", 300), ("boundary", 100)]

# ---------- held-out paraphrase bank (disjoint from agent SYN/CANON) --------
PARA = [(re.compile(p, re.I), r) for p, r in [
    (r"\b100% (\w+)\b",            r"pure \1"),
    (r"\b(\d+)% (\w+)\b",          r"\1 percent \2"),
    (r"\bMachine Wash\b",          "washer friendly"),
    (r"\bHand Wash Only\b",        "wash it by hand please"),
    (r"\bDry Clean Only\b",        "needs professional cleaning"),
    (r"\bImported\b",              "made abroad"),
    (r"\bMade in USA\b",           "American made"),
    (r"\bPull On closure\b",       "no fasteners, just pull them on"),
    (r"\bZipper closure\b",        "it has a zip"),
    (r"\bButton closure\b",        "buttons up the front"),
    (r"\bLace-up closure\b",       "laces you tie"),
    (r"\bHook and Loop closure\b", "the sticky strap kind"),
    (r"\bBuckle closure\b",        "a strap with a buckle"),
    (r"\bElastic closure\b",       "an elasticated band"),
    (r"\bRubber sole\b",           "rubber on the bottom"),
    (r"\bSynthetic sole\b",        "a man-made bottom"),
    (r"\bclosure\b",               "fastening"),
    (r"\bsole\b",                  "underside"),
    (r"\bwaterproof\b",            "keeps the rain out"),
    (r"\bbreathable\b",            "lets air through"),
    (r"\blightweight\b",           "barely weighs anything"),
    (r"\bmoisture[- ]wicking\b",   "pulls sweat away"),
    (r"\badjustable\b",            "you can resize it"),
    (r"\bpockets?\b",              "spots to stash things"),
]]
def _drop(text, rate, key):
    rng = random.Random(hashlib.md5((key + text).encode()).hexdigest())
    words = text.split()
    if len(words) < 4: return text
    kept = [w for w in words if rng.random() > rate]
    return " ".join(kept) if kept else text
def paraphrase(text, key="E"):
    for rx, sub in PARA: text = rx.sub(sub, text)
    return _drop(text, 0.15, key)

# ---------- vague-vibe bank (axis G) ----------------------------------------
VIBE_MAT = {"cotton": "natural feeling and soft on skin",
            "polyester": "man-made quick-dry sort of material",
            "nylon": "that slick sporty material",
            "leather": "that premium animal-skin feel",
            "wool": "warm and cozy for the cold months",
            "spandex": "stretchy, moves with me",
            "silk": "smooth and a bit fancy",
            "rayon": "flowy lightweight stuff",
            "fabric": "nice material"}
SHADE = {"black": "a darker shade", "white": "something light colored",
         "blue": "a cool tone", "red": "a bold warm tone", "pink": "a soft warm tone",
         "green": "an earthy tone", "brown": "an earthy neutral", "gray": "a muted neutral",
         "grey": "a muted neutral", "purple": "a rich deep tone",
         "yellow": "something bright", "orange": "something bright and warm"}
def vibe(constraint, key="G"):
    c = constraint.strip(); low = c.lower()
    if low.startswith("color: "):
        return SHADE.get(low[7:].strip(), "a color that feels right")
    for m, v in VIBE_MAT.items():
        if re.search(rf"\b{m}\b", low): return v
    if re.search(r"wash|clean", low):   return "easy to look after"
    if re.search(r"closure|zip|lace|buckle|strap|pull on", low): return "easy to get on and off"
    if re.search(r"sole|heel|grip", low): return "comfortable to walk around in"
    if re.search(r"budget|\$", low):     return "nothing too pricey"
    return paraphrase(c, key)           # fall back to held-out paraphrase

# ---------- generation -------------------------------------------------------
def _cat_of(prod): return LE.coarse_category([str(v) for v in prod.get("categories") or []])

def gen():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = LE.load_jsonl('data/public_set.jsonl')
    ids, cats, prods = LE.catalog_index('data/catalog.jsonl')
    public = {str(s['ground_truth']['parent_asin']) for s in samples}
    profiles = [s['user_profile'] for s in samples]
    rng = random.Random(SEED)
    core = sorted(a for a, p in prods.items() if (p.get('rating_number') or 0) >= 5 and a not in public)
    everything = sorted(a for a in prods if a not in public)
    by_cat = {}
    for a in core: by_cat.setdefault(_cat_of(prods[a]), []).append(a)

    def base(sid, asin, scen):
        return {"sample_id": sid, "scenario_type": scen,
                "ground_truth": {"parent_asin": asin}, "user_profile": rng.choice(profiles)}
    def mixed(pool):
        out = []
        for scen, n in MIX:
            out += [(rng.choice(pool), scen) for _ in range(n)]
        rng.shuffle(out); return out

    axes = {}
    axes["E_paraphrase"] = [base(f"s10k_E_{i:05d}", a, sc) for i, (a, sc) in enumerate(mixed(core))]
    axes["G_vague"]      = [base(f"s10k_G_{i:05d}", a, sc) for i, (a, sc) in enumerate(mixed(core))]
    axes["H_chaos"]      = [base(f"s10k_H_{i:05d}", a, sc) for i, (a, sc) in enumerate(mixed(core))]
    axes["I_coldpara"]   = [base(f"s10k_I_{i:05d}", a, sc) for i, (a, sc) in enumerate(mixed(everything))]

    # F: true product switches - ALL override, half cross-category
    f = []
    for i in range(2000):
        b = rng.choice(core)                       # the actual target
        bcat = _cat_of(prods[b])
        if i < 1000:                               # cross-category switch
            other = rng.choice([c for c in (rng.choice(list(by_cat)) for _ in range(40))
                                if c != bcat] or [bcat])
            a = rng.choice(by_cat[other])
        else:                                      # same bucket, different product
            pool = [x for x in by_cat.get(bcat, []) if x != b] or core
            a = rng.choice(pool)
        card_b = LE.intent_card(prods[b]); card_a = LE.intent_card(prods[a])
        pool_a = [*card_a["hard_constraints"], *card_a["soft_preferences"]] or ["I prefer a different style."]
        new = card_b["hard_constraints"][0] if card_b["hard_constraints"] else "Please prioritize the target requirements."
        s = base(f"s10k_F_{i:05d}", b, "intent_override")
        s["intent_card"] = card_b
        s["behavior"] = {"scenario_type": "intent_override",
                         "override": {"turn": rng.choice([3, 4]), "old_value": pool_a[0], "new_value": new,
                                      "message": f"Actually, ignore my earlier preference. What I need is: {new}."}}
        s["_precat"] = _cat_of(prods[a])
        s["_precard"] = pool_a[:4]
        f.append(s)
    axes["F_switch"] = f

    for name, rows in axes.items():
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        print(f"wrote {name}: {len(rows)}")

# ---------- run-time message patches -----------------------------------------
ORIG_CR, ORIG_IM = LE.customer_reply, LE.initial_message

def _payload_patch(transform):
    def cr(sample, ask, disclosed, boundary):
        msg, b = ORIG_CR(sample, ask, disclosed, boundary)
        if "what matters is:" in msg:
            h, t = msg.split("what matters is:", 1)
            parts = [p.strip() for p in t.rstrip('.').split(';') if p.strip()]
            msg = h + "what matters is: " + "; ".join(transform(p, sample["sample_id"]) for p in parts) + "."
        return msg, b
    def im(sample, category, disclosed):
        msg = ORIG_IM(sample, category, disclosed)
        if "A key requirement is:" in msg:
            h, t = msg.split("A key requirement is:", 1)
            msg = h + "A key requirement is: " + transform(t.strip().rstrip('.'), sample["sample_id"]) + "."
        return msg
    return cr, im

def _damage_cat(cat, key):
    words = cat.split()
    rng = random.Random(hashlib.md5(("H" + key + cat).encode()).hexdigest())
    if len(words) > 2: words.pop(rng.randrange(len(words)))
    rng.shuffle(words)
    return " ".join(words)

def patch(axis):
    if axis == "E_paraphrase":
        LE.customer_reply, LE.initial_message = _payload_patch(lambda c, k: paraphrase(c, k))
    elif axis == "G_vague":
        LE.customer_reply, LE.initial_message = _payload_patch(lambda c, k: vibe(c, k))
    elif axis == "I_coldpara":
        LE.customer_reply, LE.initial_message = _payload_patch(lambda c, k: paraphrase(c, k))
    elif axis == "H_chaos":
        cr0, im0 = _payload_patch(lambda c, k: vibe(c, k) if int(hashlib.md5(k.encode()).hexdigest(), 16) % 3 == 0 else paraphrase(c, k))
        def im(sample, category, disclosed):
            return im0(sample, _damage_cat(category, sample["sample_id"]), disclosed)
        LE.customer_reply, LE.initial_message = cr0, im
    elif axis == "F_switch":
        def im(sample, category, disclosed):
            # the OPENING is about product A: A's category, A's preference
            old = str(sample["behavior"]["override"]["old_value"])
            return f"I'm looking for {sample['_precat']}. {old}"
        def cr(sample, ask, disclosed, boundary):
            # pre-override asks reveal PRODUCT A facts; afterwards, the target's
            ov_turn = int(sample["behavior"]["override"]["turn"])
            n = sample.setdefault("_precalls", 0)
            if n < ov_turn - 2:                     # replies for turns 2..ov-1
                sample["_precalls"] = n + 1
                pool = [v for v in sample["_precard"] if v not in disclosed]
                if pool:
                    take = pool[:2]; disclosed.update(take)
                    return "For that, what matters is: " + "; ".join(take) + ".", boundary
            return ORIG_CR(sample, ask, disclosed, boundary)
        LE.customer_reply, LE.initial_message = cr, im
def unpatch():
    LE.customer_reply, LE.initial_message = ORIG_CR, ORIG_IM

# ---------- runner ------------------------------------------------------------
def run(tag):
    from submission.agent import Agent
    samples_pub = LE.load_jsonl('data/public_set.jsonl')
    ids, cats, prods = LE.catalog_index('data/catalog.jsonl')
    agent = Agent('data/catalog.jsonl')
    print(f"{'variant':9s} {'axis':13s} {'n':>5s} {'hit@10':>7s} {'MRR':>7s} {'MTTC':>6s} {'SCORE':>8s}", flush=True)
    tot_n = tot_w = 0.0
    for axis in ("E_paraphrase", "F_switch", "G_vague", "H_chaos", "I_coldpara"):
        smp = LE.load_jsonl(OUT / f"{axis}.jsonl")
        for s in smp: s.pop("_precalls", None)
        patch(axis); agent.S = {}
        r = LE.evaluate(agent, smp, ids, cats, prods)
        unpatch()
        print(f"{tag:9s} {axis:13s} {len(smp):>5d} {r['hit_rate_at_10']:>7.3f} "
              f"{r['mrr']:>7.4f} {r['mttc']:>6.2f} {r['recommended_technical_score']:>8.5f}", flush=True)
        tot_n += len(smp); tot_w += r['recommended_technical_score'] * len(smp)
    print(f"{tag:9s} {'WEIGHTED-10K':13s} {int(tot_n):>5d} {'':>7s} {'':>7s} {'':>6s} {tot_w/tot_n:>8.5f}", flush=True)

if __name__ == "__main__":
    if sys.argv[1:] == ["gen"]: gen()
    elif len(sys.argv) == 3 and sys.argv[1] == "run": run(sys.argv[2])
    else: print(__doc__)
