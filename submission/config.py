"""All tunable constants. Sweep these, never edit logic to tune."""
import os
EMIT_THRESHOLD   = 5       # commit when candidate set <= this
HOLD_UNTIL       = 3       # ...or when the turn reaches this, whichever first
BM25_K1          = 1.2
BM25_B           = 0.75
POP_TIEBREAK     = True
JACCARD_MIN      = 0.70    # token-set overlap needed to accept a fuzzy bucket.
                           # Raised 0.50 -> 0.70 (Remediation #4). Measured on 900
                           # matched sessions per generator, paired, 95% bootstrap CI:
                           #   D2 char-deletion   +0.00411 [+0.00109, +0.00804]
                           #   D3 morphological   +0.00589 [+0.00151, +0.01133]
                           #   D1 word-reversal    0.00000 -- STRUCTURALLY INERT:
                           #     reversal preserves the token SET, so Jaccard is 1.00
                           #     on every session and no threshold can bind. D1 is
                           #     tools/verify_agent.py's cat_noise, so this constant
                           #     had never been exercised by our own conditions.
                           # 0.70 is the most conservative value at the left edge of a
                           # plateau replicated across the two LIVE generators; 0.70,
                           # 0.75 and 0.80 are identical on both. NOT proven optimal,
                           # and D2/D3 are synthetic -- their private prevalence is
                           # unknown. Inert on clean input (verified byte-identical).
FUZZY_CATEGORY   = 0.60    # difflib cutoff when the leaked category misses a bucket
WEAK_ABS         = 1.5     # top BM25 score below this => weak evidence
WEAK_RATIO       = 1.15    # top1/top2 below this => flat distribution => weak
# Dense escalation lane — opt-in via environment, never on by default.
# The scoring path must survive a no-network, no-third-party-deps, unknown-timeout
# harness, so the default is the pure-stdlib cascade (identical score on every
# official-shape condition). Setting TECHJAM_DENSE=1 enables the lane IF
# sentence-transformers imports and weights are available; every failure mode
# (missing deps, missing weights, no network) falls back silently to stdlib.
# Measured with the resolution gate: byte-identical on clean/drops/category,
# +0.036 on synonym-paraphrased constraints. Cost: ~75 s startup on MPS (longer
# CPU-only — the reason this is opt-in rather than auto: a harness timeout kill
# is a zeroed run, and their timeout cannot be detected from inside).
USE_DENSE        = os.environ.get("TECHJAM_DENSE", "").strip().lower() not in ("", "0", "false")
DENSE_MODEL      = "BAAI/bge-small-en-v1.5"
CAT_BOOST        = 4.0     # score multiplier when the guessed bucket matches
OVERRIDE_DETECT  = True    # drop prior clues ONLY when they provably contradict the new one
ASK_ATTRIBUTE    = "other"
OVERRIDE_GATE_MAX_TURN = 4  # Finding #1: the override gate is advisory and expires.
                            # behavior_for draws the override turn from
                            # rng.choice([3,4]); a gate still closed after turn 4
                            # was a wrong turn-1 prediction, so stop suppressing.

# Peer-review round 3 — measured on all six conditions before adoption (29 Aug).
GLOBAL_EXACT     = True    # A2 (Germaine): category-free clue-intersection fallback,
                           #     gated on cat_sure. +0.049 cat / +0.019 cat+35%,
                           #     byte-identical everywhere else. Ungated it regressed
                           #     every clue-drop column — the gate is load-bearing.
TURN1_PAGE       = 1       # Finding #2: turn 1 emits at most this many cards.
                           # Turn 1 carries at most one clue and the evaluator locks
                           # the rank of the first hit forever, so a wide speculative
                           # page can only cap a session. Measured on the frozen
                           # suite: F3 +0.01156 paired, clean public +0.00130,
                           # F1a-B +0.00137, F1a-C +0.00394, HitRate unchanged.
                           # No candidate-count threshold: nearby thresholds decay
                           # monotonically (>10 +0.0110, >20 +0.0084, >40 +0.0045,
                           # >100 +0.0019), so any cut-off would be a fitted constant.
NOEVID_PAGE      = 1       # B (Marcus): show 1 card on evidence-free turns. Improves
                           #     ALL six conditions (+0.005 clean); browsing and
                           #     boundary clean MRR both -> 1.0000.
NOMORE_FILTER    = False   # C (Germaine): "no additional preference" bounds card
                           #     size. Measured exactly 0.0 everywhere — inert; off.

# Slot-value scheduler (adopted from Germaine, measured on all six conditions).
# A hit at (turn t, rank r) is worth 0.50 + 0.30/r + 0.02*(11-t). Candidates are
# assigned to the highest-value slots, so the runner-up is held for a rank-1 slot
# next turn instead of being shown at rank 2 now. Supersedes EMIT_THRESHOLD /
# HOLD_UNTIL / HOLD_PAGE, which are kept only for the ablation probes in tools/.
MAX_TURNS = 10
TRACE_PATH        = ""      # set to a file path (e.g. "trace.jsonl") to log one JSONL record per turn
# Pillar III — personalized context distillation (adopted 30 Aug, measured).
# preference_tags + summary words, blended into the EVIDENCE-FREE ordering only:
#   score = log1p(rating_number) + PROFILE_WEIGHT * 10 * tag-affinity
# Improves ALL six conditions (+0.0003..+0.0006) and payout-rephrased (+0.0004);
# wide flat optimum (0.35-0.75 byte-identical); degrades at >=1.0 where profile
# words start outvoting popularity. Deterministic 0.9787 across hash seeds.
PROFILE_WEIGHT    = 0.35
USE_FTS           = False   # P4 full-text FTS5 lane: measured WORSE on 5 of 9
                            # conditions (dilutes good rankings under clue damage);
                            # rejected with numbers, kept only as an ablation switch.

# Proactive clarification (Pillar II). The simulator never reads `message` —
# proven by replacing it with junk text: score byte-identical. These thresholds
# therefore shape only what a human sees in the demo and the report.
OVERGENERAL_AT    = 60      # candidates above this => ask a ruling-out question
CONFIDENT_AT      = 3       # candidates at or below this => present as an answer
STOP_ASKING_WHEN_DRAINED = True   # once the shopper has no preferences left, thank
                                  # them and stop asking rather than re-prompting
SHOW_WHILE_GATED  = True    # intent_override: display the current best guesses on
                            # pre-gate turns. They cannot score (the evaluator
                            # discards hits until the mind-change lands) and are NOT
                            # recorded as seen, so this is score-neutral - measured
                            # byte-identical - and it stops the transcript looking
                            # like the agent has nothing to say.
DENSE_UNRESOLVED_ONLY = True  # dense escalation fires only when NO clue resolved
                              # in the exact index (paraphrase signature)
UNSURE_SEEN_HOLD = True    # when turn 1 matched no template the scenario is unknown,
                           # so do not record shown cards as proven negatives until
                           # past OVERRIDE_GATE_MAX_TURN. Cards are still shown; only
                           # the "already ruled out" bookkeeping waits.
RECOVER_CLUES    = True    # template-free constraint recovery: scan a message that no
                           # template matched for constraint strings that are verbatim
                           # in the index. The simulator copies constraints from the
                           # target's metadata, so paraphrasing the prose AROUND them
                           # leaves them byte-identical and still findable.
RECOVER_MIN_LEN  = 8       # ignore very short constraint strings ("fabric", "Imported"):
                           # they occur incidentally in ordinary prose.
RECOVER_MAX_DF   = 2000    # ...and ignore ones held by more than this many products.
                           # Both gates are about precision, not recall: a false clue
                           # intersects the candidate pool down to the wrong product.
CANONICALIZE     = True    # stdlib query-rewriting: colloquial color/material words
                           # mapped onto the evaluator's vocabulary, consulted only
                           # for clues that resolved nowhere (clean input untouched
                           # by construction).
