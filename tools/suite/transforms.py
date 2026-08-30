"""Message transformations for F2A (surface), F2B (evidence) and F7 (OOD).

ARCHITECTURE NOTE -- this is what makes requirement 3 mechanical rather than
asserted.  Messages are never string-patched.  They are REBUILT from slots:

    template_variant + slot values  ->  message

* **F2A** varies the template variant and passes constraint slots through
  untouched, so constraint strings are byte-identical *by construction*.
* **F2B** pins the template variant to canonical and substitutes paraphrased
  slot values, so the template is byte-identical *by construction*.

Neither invariant depends on anyone remembering to preserve it.

Slots are `<CAT>`, `<C>`, `<JOINED>`, `<OLD>`, `<ATTR>`.  Substitution is plain
`str.replace`, never `str.format`, because constraint strings contain braces,
percent signs and backslashes drawn from arbitrary product text.

NO LLM IS CALLED ANYWHERE IN THIS MODULE, at generation time or at run time.
Every F2B paraphrase comes from the curated rule table below and is recorded in
the test case so it can be human-reviewed before evaluation.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Canonical templates -- byte-identical to evaluator/local_evaluator.py
# --------------------------------------------------------------------------
SURFACES = ("opening", "payout", "filler", "override", "opening+payout", "all")

CANONICAL = {
    "opening_buying": "I'm looking for <CAT>. A key requirement is: <C>.",
    "opening_browsing": "I'm looking for <CAT>, but I'm still exploring.",
    "opening_override": "I'm looking for <CAT>. <OLD>",
    "payout": "For that, what matters is: <JOINED>.",
    "filler_none": "Those options are not quite right yet. Ask me about one specific attribute.",
    "filler_drained": "I don't have an additional preference for <ATTR>.",
    "filler_boundary": "I don't have a preference for <ATTR>; please use your judgment.",
    "override": "Actually, ignore my earlier preference. What I need is: <C>.",
}

F2A_KINDS = ("lexical", "reorder", "function_word", "shortened", "natural", "prose_drift")
F2A_SEVERITIES = ("S1", "S2", "S3")

# --------------------------------------------------------------------------
# F2A variant table.  variants[template][kind][severity] -> list of templates.
# Every entry keeps its slots intact; nothing else is guaranteed.
# --------------------------------------------------------------------------
_V: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "opening_buying": {
        "lexical": {
            "S1": ["I'm searching for <CAT>. A key requirement is: <C>."],
            "S2": ["I'm after <CAT>. One key requirement is: <C>."],
            "S3": ["I want to buy <CAT>. My must-have is: <C>."],
        },
        "reorder": {
            "S1": ["A key requirement is: <C>. I'm looking for <CAT>."],
            "S2": ["Requirement first: <C>. The item is <CAT>."],
            "S3": ["<C> — that's the requirement. Category: <CAT>."],
        },
        "function_word": {
            "S1": ["I am looking for <CAT>. A key requirement is: <C>."],
            "S2": ["I am looking for some <CAT>. The key requirement is: <C>."],
            "S3": ["It is <CAT> that I am after. The one key requirement is that of: <C>."],
        },
        "shortened": {
            "S1": ["Looking for <CAT>. Key requirement: <C>."],
            "S2": ["<CAT>. Requirement: <C>."],
            "S3": ["<CAT>: <C>."],
        },
        "natural": {
            "S1": ["I'm in the market for <CAT>, and one thing I really need is: <C>."],
            "S2": ["So I need <CAT>, and the thing I can't compromise on is: <C>."],
            "S3": ["Right, here's the thing — I need <CAT>, and it absolutely has to be: <C>."],
        },
        "prose_drift": {
            "S1": ["Hi there. I'm looking for <CAT>. A key requirement is: <C>. Thanks!"],
            "S2": ["Hey, hope you can help. Shopping for <CAT> today. Non-negotiable: <C>. Cheers."],
            "S3": [
                "Good afternoon! My old one finally gave out so I'm replacing it. "
                "The category I need is <CAT>. If it helps, the thing that matters most: <C>. "
                "Appreciate any suggestions."
            ],
        },
    },
    "opening_browsing": {
        "lexical": {
            "S1": ["I'm searching for <CAT>, but I'm still exploring."],
            "S2": ["I'm after <CAT>, though I'm still browsing."],
            "S3": ["I want <CAT>, but I haven't decided yet."],
        },
        "reorder": {
            "S1": ["Still exploring, but I'm looking for <CAT>."],
            "S2": ["Just browsing for now — <CAT> is the category."],
            "S3": ["<CAT> is roughly it, though nothing is settled."],
        },
        "function_word": {
            "S1": ["I am looking for <CAT>, but I am still exploring."],
            "S2": ["I am looking for some <CAT>, although I am still exploring."],
            "S3": ["It is <CAT> that I am looking at, although I am as yet still exploring."],
        },
        "shortened": {
            "S1": ["Looking for <CAT>, still exploring."],
            "S2": ["<CAT>, still looking."],
            "S3": ["<CAT>?"],
        },
        "natural": {
            "S1": ["I'm thinking about <CAT>, but I haven't made my mind up."],
            "S2": ["I could use <CAT>. Not sure exactly what yet, just having a look."],
            "S3": ["Honestly I'm just poking around at <CAT> to see what's out there."],
        },
        "prose_drift": {
            "S1": ["Hi! I'm looking for <CAT>, but I'm still exploring."],
            "S2": ["Hey there, hope you're well. On the hunt for <CAT>. Very much still browsing."],
            "S3": [
                "Afternoon! No rush on this one. Category is <CAT>. "
                "I genuinely have not decided anything yet, so show me whatever you think."
            ],
        },
    },
    "opening_override": {
        "lexical": {
            "S1": ["I'm searching for <CAT>. <OLD>"],
            "S2": ["I'm after <CAT>. <OLD>"],
            "S3": ["I want <CAT>. <OLD>"],
        },
        "reorder": {
            "S1": ["<OLD> I'm looking for <CAT>."],
            "S2": ["<OLD> Category: <CAT>."],
            "S3": ["<OLD> — and it's <CAT>."],
        },
        "function_word": {
            "S1": ["I am looking for <CAT>. <OLD>"],
            "S2": ["I am looking for some <CAT>. <OLD>"],
            "S3": ["It is <CAT> that I am looking for. <OLD>"],
        },
        "shortened": {
            "S1": ["Looking for <CAT>. <OLD>"],
            "S2": ["<CAT>. <OLD>"],
            "S3": ["<CAT> — <OLD>"],
        },
        "natural": {
            "S1": ["I'm in the market for <CAT>. <OLD>"],
            "S2": ["So, <CAT> is what I need. <OLD>"],
            "S3": ["Right — <CAT>, and for what it's worth: <OLD>"],
        },
        "prose_drift": {
            "S1": ["Hi. I'm looking for <CAT>. <OLD>"],
            "S2": ["Hey, hoping you can help me out. Shopping for <CAT>. <OLD>"],
            "S3": ["Good morning! Long story but I need <CAT>. For context: <OLD>"],
        },
    },
    "payout": {
        "lexical": {
            "S1": ["For that, what counts is: <JOINED>."],
            "S2": ["For that, what I care about is: <JOINED>."],
            "S3": ["The things I value are: <JOINED>."],
        },
        "reorder": {
            "S1": ["<JOINED> — that's what matters for that."],
            "S2": ["What matters, for that: <JOINED>."],
            "S3": ["<JOINED>. Those are the things."],
        },
        "function_word": {
            "S1": ["For that, what matters is the following: <JOINED>."],
            "S2": ["For that one, the thing that matters is: <JOINED>."],
            "S3": ["As for that, the things which do matter would be: <JOINED>."],
        },
        "shortened": {
            "S1": ["What matters: <JOINED>."],
            "S2": ["Matters: <JOINED>."],
            "S3": ["<JOINED>."],
        },
        "natural": {
            "S1": ["The things that matter to me there are: <JOINED>."],
            "S2": ["Good question — mainly these: <JOINED>."],
            "S3": ["Honestly the bits I actually care about are these: <JOINED>."],
        },
        "prose_drift": {
            "S1": ["Sure. For that, what matters is: <JOINED>. Hope that helps."],
            "S2": ["Happy to say more. What matters is: <JOINED>. Let me know if that narrows it."],
            "S3": [
                "Thanks for asking, that's a fair question and I've been thinking about it. "
                "The things that genuinely matter to me are these: <JOINED>. "
                "Does that help at all?"
            ],
        },
    },
    "filler_none": {
        "lexical": {
            "S1": ["Those options aren't quite right yet. Ask me about one specific attribute."],
            "S2": ["Those choices aren't right yet. Ask about one specific attribute."],
            "S3": ["Not these. Ask me about a single attribute."],
        },
        "reorder": {
            "S1": ["Ask me about one specific attribute. Those options are not quite right yet."],
            "S2": ["Ask about a specific attribute — these aren't right yet."],
            "S3": ["One attribute at a time, please. These miss."],
        },
        "function_word": {
            "S1": ["Those options are not quite the right ones yet. Ask me about a specific attribute."],
            "S2": ["The options are not quite right as yet. Do ask me about one specific attribute."],
            "S3": ["It is the case that these are not right yet. Ask about but one attribute."],
        },
        "shortened": {"S1": ["Not quite right yet."], "S2": ["Nope."], "S3": ["No."]},
        "natural": {
            "S1": ["Hmm, none of those are quite it. Ask me something specific?"],
            "S2": ["Not really what I had in mind. What do you want to know?"],
            "S3": ["Nah, none of those. Ask me something."],
        },
        "prose_drift": {
            "S1": ["Thanks, but those options are not quite right yet. Ask me about one specific attribute."],
            "S2": ["Appreciate the effort! Not quite there though. Ask me about one thing at a time."],
            "S3": ["I do appreciate you trying. Sadly none of those land. Ask me anything specific."],
        },
    },
    "filler_drained": {
        "lexical": {
            "S1": ["I don't have a further preference for <ATTR>."],
            "S2": ["I have no other preference for <ATTR>."],
            "S3": ["Nothing more on <ATTR>."],
        },
        "reorder": {
            "S1": ["For <ATTR>, I don't have an additional preference."],
            "S2": ["<ATTR> — no additional preference."],
            "S3": ["<ATTR>: nothing else."],
        },
        "function_word": {
            "S1": ["I do not have an additional preference for <ATTR>."],
            "S2": ["I do not have any additional preference as to <ATTR>."],
            "S3": ["There is not any additional preference that I have for <ATTR>."],
        },
        "shortened": {"S1": ["No more on <ATTR>."], "S2": ["<ATTR>: none."], "S3": ["None."]},
        "natural": {
            "S1": ["I think that's everything on <ATTR>."],
            "S2": ["That's all I've got for <ATTR>, sorry."],
            "S3": ["Run out of things to say about <ATTR>."],
        },
        "prose_drift": {
            "S1": ["I don't have an additional preference for <ATTR>, sorry."],
            "S2": ["Sorry, I think I've said everything I have about <ATTR> already."],
            "S3": ["Honestly I've told you all I can about <ATTR> — nothing further comes to mind."],
        },
    },
    "filler_boundary": {
        "lexical": {
            "S1": ["I don't have a preference for <ATTR>; please use your discretion."],
            "S2": ["I have no preference on <ATTR>; your call."],
            "S3": ["No preference on <ATTR>. You pick."],
        },
        "reorder": {
            "S1": ["Please use your judgment; I don't have a preference for <ATTR>."],
            "S2": ["Your judgment on <ATTR> — I have no preference."],
            "S3": ["<ATTR>: you decide."],
        },
        "function_word": {
            "S1": ["I do not have a preference for <ATTR>; please do use your judgment."],
            "S2": ["I do not have any preference as to <ATTR>; please use your own judgment."],
            "S3": ["There is not a preference that I hold for <ATTR>; do use your judgment."],
        },
        "shortened": {"S1": ["No preference on <ATTR>."], "S2": ["<ATTR>: whatever."], "S3": ["Whatever."]},
        "natural": {
            "S1": ["I don't really mind about <ATTR> — you choose."],
            "S2": ["No strong feelings on <ATTR>, go with what you think."],
            "S3": ["Genuinely don't care about <ATTR>, surprise me."],
        },
        "prose_drift": {
            "S1": ["I don't have a preference for <ATTR>; please use your judgment. Thanks!"],
            "S2": ["Good question, but I really don't mind about <ATTR>. I'll trust you on that one."],
            "S3": ["Ha, you've got me — I've never once thought about <ATTR>. Use your judgment."],
        },
    },
    "override": {
        "lexical": {
            "S1": ["Actually, disregard my earlier preference. What I need is: <C>."],
            "S2": ["Actually, forget my earlier preference. What I want is: <C>."],
            "S3": ["Change of plan. What I need is: <C>."],
        },
        "reorder": {
            "S1": ["What I need is: <C>. Actually, ignore my earlier preference."],
            "S2": ["Scrap that — <C> is what I need."],
            "S3": ["<C>. Forget what I said before."],
        },
        "function_word": {
            "S1": ["Actually, please ignore my earlier preference. What I need is: <C>."],
            "S2": ["Actually, do ignore that earlier preference of mine. What I need is: <C>."],
            "S3": ["Actually, it is my earlier preference that should be ignored. What I need is: <C>."],
        },
        "shortened": {"S1": ["Ignore that. Need: <C>."], "S2": ["Scrap that. <C>."], "S3": ["<C> instead."]},
        "natural": {
            "S1": ["Sorry, I've changed my mind. What I really need is: <C>."],
            "S2": ["On second thought, forget that. What I need is: <C>."],
            "S3": ["You know what, ignore all that. It's this: <C>."],
        },
        "prose_drift": {
            "S1": ["Actually, ignore my earlier preference. What I need is: <C>. Sorry for the switch!"],
            "S2": ["Sorry — I've been going back and forth. Ignore the earlier thing. What I need is: <C>."],
            "S3": [
                "Apologies, I've completely changed my mind after talking to my partner about it. "
                "Please ignore everything I said earlier. What I actually need is: <C>."
            ],
        },
    },
}


def template_for(name: str, kind: Optional[str], severity: Optional[str], variant_index: int = 0) -> str:
    """Return the template string for a surface, or the canonical one."""
    if kind is None or severity is None:
        return CANONICAL[name]
    options = _V[name][kind][severity]
    return options[variant_index % len(options)]


def render(template: str, *, cat: str = "", constraint: str = "", joined: str = "",
           old: str = "", attr: str = "") -> str:
    """Slot substitution by replace, never format -- constraint text is arbitrary."""
    out = template
    for token, value in (("<CAT>", cat), ("<C>", constraint), ("<JOINED>", joined),
                         ("<OLD>", old), ("<ATTR>", attr)):
        out = out.replace(token, value)
    return out


# --------------------------------------------------------------------------
# F2B -- evidence paraphrase.  Curated, deterministic, meaning-preserving.
# --------------------------------------------------------------------------
# Three tiers, all shape-aware:
#   t1_pattern  curated CONTENT-WORD rewrites -- strongest retrieval test
#   t2_reorder  structural reordering        -- token-multiset preserving
#   t3_carrier  minimal grammatical wrapping -- original tokens preserved
#
# A constraint is paraphrased only if the result passes `validate_paraphrase`.
# If no safe transformation exists the ORIGINAL IS PRESERVED -- we never force a
# broken rewrite, and the shortfall is recorded as a lower achieved severity.

_BRACKETS = (("(", ")"), ("[", "]"), ("{", "}"), ("\uff08", "\uff09"), ("\u3010", "\u3011"))
_MATERIAL_WORDS = ("cotton", "polyester", "nylon", "leather", "wool",
                   "spandex", "silk", "rayon", "fabric")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_KV_RE = re.compile(r"^([A-Za-z][A-Za-z /&-]{1,30}):\s*(.+)$")
_DIM_RE = re.compile(r"\d+(?:\.\d+)?\s*[x\u00d7]\s*\d", re.I)

SHAPES = ("synthetic_material", "synthetic_color", "synthetic_price", "percentage",
          "key_value", "dimension", "short_fragment", "clause")


def _balanced(text: str) -> bool:
    for left, right in _BRACKETS:
        if text.count(left) != text.count(right):
            return False
    return text.count('"') % 2 == 0


def _numbers(text: str):
    return sorted(_NUM_RE.findall(text))


def _content_tokens(text: str):
    import collections as _c
    return _c.Counter(t.lower() for t in _TOKEN_RE.findall(text))


def _bracket_spans(text: str):
    spans = []
    for left, right in _BRACKETS:
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch == left:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == right and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, i))
    return spans


def classify_shape(text: str) -> str:
    lowered = text.strip().lower()
    if lowered in _MATERIAL_WORDS:
        return "synthetic_material"
    if re.match(r"^color:\s*", text, re.I):
        return "synthetic_color"
    if re.match(r"^budget around \$", text, re.I):
        return "synthetic_price"
    if _PCT_RE.search(text):
        return "percentage"
    if _KV_RE.match(text) and len(text) < 120:
        return "key_value"
    if _DIM_RE.search(text):
        return "dimension"
    if len(text) <= 45 and "." not in text.rstrip("."):
        return "short_fragment"
    return "clause"


def validate_paraphrase(original: str, new: str, tier: str) -> tuple:
    """Mechanical safety gate.  Returns (ok, reason).  NOT a substitute for human
    review -- it catches mechanical breakage only."""
    if not new or not new.strip():
        return False, "empty output"
    if new.strip() == original.strip():
        return False, "unchanged"
    if _balanced(original):
        if not _balanced(new):
            return False, "transformation unbalanced the brackets"
    else:
        for left, right in _BRACKETS:
            if (new.count(left) - new.count(right)) != (original.count(left) - original.count(right)):
                return False, "bracket imbalance changed"
    if _numbers(original) != _numbers(new) and tier != "t1":
        return False, "numeric groups altered"
    if tier == "t2":
        if _content_tokens(original) != _content_tokens(new):
            return False, "token multiset not preserved"
    if tier == "t3":
        original_tokens, new_tokens = _content_tokens(original), _content_tokens(new)
        if any(new_tokens[k] < v for k, v in original_tokens.items()):
            return False, "original tokens lost"
    return True, "ok"


# ---- tier 1: curated content rewrites, by shape --------------------------

def _t1(text: str, shape: str):
    if shape == "synthetic_material":
        return f"made of {text.strip().lower()}", "t1_material"
    if shape == "synthetic_color":
        value = re.sub(r"^color:\s*", "", text, flags=re.I).strip()
        return f"the colour is {value}", "t1_color"
    if shape == "synthetic_price":
        value = re.sub(r"^budget around \$", "", text, flags=re.I).strip()
        return f"I would want to stay near {value} dollars", "t1_budget"
    if shape == "percentage":
        groups = _PCT_RE.findall(text)
        if len(groups) == 1 and groups[0] in ("100", "100.0"):
            rest = _PCT_RE.sub("", text).strip()
            if rest:
                return f"made entirely from {rest}", "t1_pct_full"
        # FIX 2: transform EVERY percentage group, never just the first.
        return _PCT_RE.sub(lambda m: f"{m.group(1)} percent", text), "t1_pct_all"
    if shape == "dimension":
        return f"it measures about {text}", "t1_dimension"
    if shape == "key_value":
        match = _KV_RE.match(text)
        key, value = match.group(1).strip().lower(), match.group(2).strip()
        verb = "are" if key.endswith("s") and not key.endswith("ss") else "is"
        return f"the {key} {verb} {value}", "t1_key_value"
    for pattern, replacement, rule_id in _T1_PHRASE_RULES:
        if pattern.match(text):
            new = pattern.sub(replacement, text).strip()
            if new and new != text:
                return new, rule_id
    return None


_T1_PHRASE_RULES = [
    (re.compile(r"^Pull On closure$", re.I), "it slips on without buttons or a zip", "t1_pullon"),
    (re.compile(r"^(.+) closure$", re.I), r"it fastens with a \1", "t1_closure"),
    (re.compile(r"^Machine Wash(?: .*)?$", re.I), "you can put it through the machine", "t1_wash"),
    (re.compile(r"^Hand Wash(?: .*)?$", re.I), "it needs washing by hand", "t1_handwash"),
    (re.compile(r"^Imported$", re.I), "it is brought in from overseas", "t1_imported"),
    (re.compile(r"^Made in (.+)$", re.I), r"it comes out of \1", "t1_madein"),
    (re.compile(r"^(.+) sole$", re.I), r"the sole is \1", "t1_sole"),
    (re.compile(r"^Lightweight (.+)$", re.I), r"a light \1", "t1_lightweight"),
]


# ---- tier 2: structural reordering, bracket-safe -------------------------

_SPLIT_RE = re.compile(r"\s+(with|and)\s+", re.I)
_T2_MAX_LEN = 160
_T2_MIN_SIDE = 8
_T2_MAX_SIDE = 90


def _t2(text: str):
    """FIX 1: reject any swap that would break brackets, split a parenthetical,
    or produce an unwieldy clause.  Returns None rather than forcing a rewrite."""
    if len(text) > _T2_MAX_LEN or not _balanced(text):
        return None
    spans = _bracket_spans(text)

    def inside_bracket(index: int) -> bool:
        return any(lo <= index <= hi for lo, hi in spans)

    candidates = []
    match = _SPLIT_RE.search(text)
    if match and not inside_bracket(match.start()):
        candidates.append((text[: match.start()].strip(), text[match.end():].strip(), "t2_clause_swap"))
    comma = text.find(",")
    if comma > 0 and not inside_bracket(comma):
        candidates.append((text[:comma].strip(), text[comma + 1:].strip(), "t2_comma_swap"))
    for head, tail, rule_id in candidates:
        if not (_T2_MIN_SIDE <= len(head) <= _T2_MAX_SIDE):
            continue
        if not (_T2_MIN_SIDE <= len(tail) <= _T2_MAX_SIDE):
            continue
        if not (_balanced(head) and _balanced(tail)):
            continue
        joined = f"{tail}, {head[0].lower() + head[1:]}"
        ok, _ = validate_paraphrase(text, joined, "t2")
        if ok:
            return joined, rule_id
    return None


# ---- tier 3: shape-aware grammatical wrapping ----------------------------

def _t3(text: str, shape: str):
    """FIX 3: no single generic carrier.  Each shape gets a natural form that
    adds no meaning the source did not carry."""
    stripped = text.strip()
    if shape == "synthetic_material":
        return f"it is made of {stripped.lower()}", "t3_material"
    if shape == "synthetic_color":
        value = re.sub(r"^color:\s*", "", stripped, flags=re.I).strip()
        return f"the color is {value}", "t3_color"
    if shape == "synthetic_price":
        value = re.sub(r"^budget around \$", "", stripped, flags=re.I).strip()
        return f"the budget is around {value}", "t3_budget"
    if shape == "key_value":
        match = _KV_RE.match(stripped)
        key, value = match.group(1).strip().lower(), match.group(2).strip()
        verb = "are" if key.endswith("s") and not key.endswith("ss") else "is"
        return f"the {key} {verb} {value}", "t3_key_value"
    if shape == "dimension":
        return f"it measures {stripped}", "t3_dimension"
    if shape == "percentage":
        return f"the composition is {stripped}", "t3_percentage"
    if shape == "short_fragment":
        if len(_TOKEN_RE.findall(stripped)) == 1:
            return f"it is {stripped.lower()}", "t3_fragment_single"
        return f"it has {stripped}", "t3_fragment"
    return f"here is what I need: {stripped}", "t3_clause"


F2B_KINDS = ("t1_pattern", "t2_reorder", "t3_carrier", "mixed", "auto", "shape_mixed")
F2B_SEVERITIES = ("E1", "E2", "E3")
_SEVERITY_MIN = {"E1": 1, "E2": 2, "E3": 4}
_TIER_OF = {"t1_pattern": ("t1",), "t2_reorder": ("t2",), "t3_carrier": ("t3",),
            "mixed": ("t1", "t2", "t3"), "auto": ("t1", "t2", "t3"),
            # shape_mixed skips the structural tier, which is the one that cannot
            # apply to every shape.  t1 then t3 are both shape-dispatched, so every
            # constraint has a rewrite and E3 is genuinely attainable.
            "shape_mixed": ("t1", "t3")}


def paraphrase_constraint(text: str, kind: str):
    """Return (paraphrase, rule_id, shape, tier) or None if nothing is safe."""
    shape = classify_shape(text)
    for tier in _TIER_OF[kind]:
        result = {"t1": lambda: _t1(text, shape), "t2": lambda: _t2(text),
                  "t3": lambda: _t3(text, shape)}[tier]()
        if not result:
            continue
        new, rule_id = result
        ok, _reason = validate_paraphrase(text, new, tier)
        if ok:
            return new, rule_id, shape, tier
    return None


def severity_for(n_changed: int, n_total: int) -> str:
    """FIX 4: E3 requires ALL available constraints to be paraphrased."""
    if n_total and n_changed >= n_total and n_changed >= 1:
        return "E3"
    if n_changed >= _SEVERITY_MIN["E2"]:
        return "E2"
    if n_changed >= _SEVERITY_MIN["E1"]:
        return "E1"
    return "E0"


def build_constraint_map(constraints, kind: str, requested: str, seed: int = 0) -> dict:
    """Paraphrase up to the requested budget.  Records every decision."""
    budget = len(constraints) if requested == "E3" else _SEVERITY_MIN[requested]
    constraint_map, records, unparaphrased, rejected = {}, [], [], []
    for text in constraints:
        if len(constraint_map) >= budget:
            unparaphrased.append(text)
            continue
        result = paraphrase_constraint(text, kind)
        if result is None:
            unparaphrased.append(text)
            rejected.append({"constraint": text, "shape": classify_shape(text),
                             "reason": "no safe transformation at this tier"})
            continue
        new, rule_id, shape, tier = result
        constraint_map[text] = new
        records.append({"original_constraint": text, "transformed_constraint": new,
                        "transformation_rule": rule_id, "shape": shape, "tier": tier})
    achieved = severity_for(len(constraint_map), len(constraints))
    for record in records:
        record["requested_severity"] = requested
        record["achieved_severity"] = achieved
    return {"constraint_map": constraint_map, "constraint_records": records,
            "unparaphrased": unparaphrased, "rejected": rejected,
            "requested_severity": requested, "achieved_severity": achieved,
            "coverage": len(constraint_map) / max(len(constraints), 1)}


# --------------------------------------------------------------------------
# F7 -- OOD shopper language.  Scripted turn-1 (and occasionally turn-3) text.
# --------------------------------------------------------------------------
OOD_CATEGORIES = (
    "colour_offpalette", "aesthetic_vocab", "vague_material", "implicit_use_case",
    "slang_register", "typos_asr", "negation", "comparison", "prior_card_reference",
    "verbosity_extremes", "budget_hard_limit", "gift_third_party",
    "occasion_season", "care_practicality", "fit_body_language", "question_to_agent",
)

_OOD_TEMPLATES: Dict[str, List[str]] = {
    "colour_offpalette": [
        "Something in ivory or maybe oat, for <CAT>.",
        "I'm after <CAT> in indigo — not navy, indigo.",
        "<CAT>, ideally teal or a sort of sage.",
    ],
    "aesthetic_vocab": [
        "Looking for <CAT> with a coquette feel.",
        "<CAT>, but make it whimsigoth.",
        "Something Y2K-ish in <CAT>.",
    ],
    "vague_material": [
        "<CAT> that's soft but not clingy, and actually breathes.",
        "I want <CAT> in something substantial — not flimsy.",
        "<CAT>, nothing scratchy please.",
    ],
    "implicit_use_case": [
        "<CAT> for standing on tile floors all day.",
        "I need <CAT> for a job where I'm on my feet from six till three.",
        "Something in <CAT> for commuting by bike.",
    ],
    "slang_register": [
        "need <CAT> thats lowkey fire, nothing corny",
        "<CAT> but make it clean, no yapping",
        "gimme <CAT> that slaps",
    ],
    "typos_asr": [
        "lookign for a blak cotten <CAT>",
        "i need <CAT> pls, sumthing confortable",
        "<CAT> in wite, medim size",
    ],
    "negation": [
        "<CAT>, but not polyester and definitely not black.",
        "I want <CAT> — no logos, nothing shiny.",
        "Anything in <CAT> except pull-on styles.",
    ],
    "comparison": [
        "<CAT> warmer than a hoodie but lighter than a parka.",
        "Like <CAT> but a bit less formal.",
        "<CAT>, thinner than the usual ones.",
    ],
    "prior_card_reference": [
        "The second one you showed, but in blue.",
        "Something like that last one, only cheaper.",
        "Closer to the first suggestion than the others.",
    ],
    "verbosity_extremes": [
        "<CAT>",
        "Good afternoon. My previous one finally wore through after about four years of "
        "steady use, mostly weekends and the occasional trip, and I've been putting off "
        "replacing it. I'd like <CAT>. I'm not especially fussy about brand, though I do "
        "care that it lasts, and I'd rather not spend a fortune on something I'll only "
        "wear now and then. Whatever you'd recommend really.",
        "need <CAT> thx",
    ],
    "budget_hard_limit": [
        "<CAT> under $40, hard limit.",
        "I can go to about fifty dollars for <CAT>, not a cent more.",
        "Cheapest decent <CAT> you have.",
    ],
    "gift_third_party": [
        "<CAT> for my dad — he's 6'4\" and hates logos.",
        "Buying <CAT> as a gift for my sister, she's into minimal stuff.",
        "<CAT> for a colleague, I don't know their size.",
    ],
    "occasion_season": [
        "<CAT> for an outdoor winter wedding — I need to not freeze.",
        "Something in <CAT> for a summer festival.",
        "<CAT> for a job interview next week.",
    ],
    "care_practicality": [
        "<CAT> that's machine washable — I will not hand-wash.",
        "<CAT>, and it can't need ironing.",
        "Something in <CAT> that survives a tumble dryer.",
    ],
    "fit_body_language": [
        "<CAT> — I'm petite with a long torso and always size up in the shoulders.",
        "<CAT> that fits broad shoulders without being tent-like.",
        "I'm between sizes in <CAT>, usually take the larger.",
    ],
    "question_to_agent": [
        "What's the difference between the options you'd show me for <CAT>?",
        "How do I pick a <CAT>? I genuinely don't know what to look for.",
        "Can you explain what makes one <CAT> better than another?",
    ],
}


def ood_message(category: str, variant_index: int, cat: str) -> str:
    options = _OOD_TEMPLATES[category]
    return options[variant_index % len(options)].replace("<CAT>", cat)


def ood_turn(category: str) -> int:
    """Which turn the OOD text is injected at.  Card references need a prior turn."""
    return 3 if category == "prior_card_reference" else 1
