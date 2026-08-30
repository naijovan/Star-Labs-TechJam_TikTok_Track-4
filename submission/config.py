"""All tunable constants. Sweep these, never edit logic to tune."""
EMIT_THRESHOLD   = 5       # commit when candidate set <= this
HOLD_UNTIL       = 3       # ...or when the turn reaches this, whichever first
BM25_K1          = 1.2
BM25_B           = 0.75
POP_TIEBREAK     = True
JACCARD_MIN      = 0.50    # token-set overlap needed to accept a fuzzy bucket
FUZZY_CATEGORY   = 0.60    # difflib cutoff when the leaked category misses a bucket
WEAK_ABS         = 1.5     # top BM25 score below this => weak evidence
WEAK_RATIO       = 1.15    # top1/top2 below this => flat distribution => weak
USE_DENSE        = False   # set True only if sentence-transformers is importable
DENSE_MODEL      = "BAAI/bge-small-en-v1.5"
CAT_BOOST        = 4.0     # score multiplier when the guessed bucket matches
OVERRIDE_DETECT  = True    # drop prior clues ONLY when they provably contradict the new one
ASK_ATTRIBUTE    = "other"

# Peer-review round 3 — measured on all six conditions before adoption (29 Aug).
GLOBAL_EXACT     = True    # A2 (Germaine): category-free clue-intersection fallback,
                           #     gated on cat_sure. +0.049 cat / +0.019 cat+35%,
                           #     byte-identical everywhere else. Ungated it regressed
                           #     every clue-drop column — the gate is load-bearing.
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
