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

# Slot-value scheduler (adopted from Germaine, measured on all six conditions).
# A hit at (turn t, rank r) is worth 0.50 + 0.30/r + 0.02*(11-t). Candidates are
# assigned to the highest-value slots, so the runner-up is held for a rank-1 slot
# next turn instead of being shown at rank 2 now. Supersedes EMIT_THRESHOLD /
# HOLD_UNTIL / HOLD_PAGE, which are kept only for the ablation probes in tools/.
MAX_TURNS = 10
