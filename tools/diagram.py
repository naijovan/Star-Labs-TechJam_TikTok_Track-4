"""Vector pipeline diagram for the Track 4 report. reportlab shapes only."""
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon, Circle, Group
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib import colors

TEAL = colors.HexColor("#0F6E56"); INK  = colors.HexColor("#1B1B1B")
MUT  = colors.HexColor("#5F5E5A"); LINE = colors.HexColor("#C9C7BD")
BG   = colors.HexColor("#F1EFE8"); ACC  = colors.HexColor("#993556")
PALE = colors.HexColor("#E4EFEA"); WHITE= colors.white
ROSE = colors.HexColor("#F6EAEE")

F, FB = "Helvetica", "Helvetica-Bold"

def wrap(text, font, size, width):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if stringWidth(t, font, size) <= width or not cur: cur = t
        else: out.append(cur); cur = w
    if cur: out.append(cur)
    return out

def txt(d, x, y, s, size=8, font=F, color=INK, anchor="start"):
    d.add(String(x, y, s, fontName=font, fontSize=size, fillColor=color, textAnchor=anchor))

def para(d, x, y, s, w, size=7.4, font=F, color=MUT, lead=None):
    lead = lead or size + 2.2
    for i, ln in enumerate(wrap(s, font, size, w)):
        txt(d, x, y - i*lead, ln, size, font, color)
    return y - (len(wrap(s, font, size, w)) - 1) * lead

def box(d, x, y, w, h, fill=BG, stroke=LINE, sw=0.7, r=None):
    kw = dict(fillColor=fill, strokeColor=stroke, strokeWidth=sw)
    if r: kw["rx"] = kw["ry"] = r
    d.add(Rect(x, y, w, h, **kw))

def arrow(d, x1, y1, x2, y2, color=TEAL, w=1.1, head=4.4):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w))
    if x1 == x2:
        s = -1 if y2 < y1 else 1
        d.add(Polygon([x2, y2, x2-head, y2-s*head*1.5, x2+head, y2-s*head*1.5],
                      fillColor=color, strokeColor=color))
    else:
        s = -1 if x2 < x1 else 1
        d.add(Polygon([x2, y2, x2-s*head*1.5, y2-head, x2-s*head*1.5, y2+head],
                      fillColor=color, strokeColor=color))

def bar(d, x, y, w, h, label, fill=TEAL, color=WHITE, size=8.4):
    box(d, x, y, w, h, fill=fill, stroke=fill)
    txt(d, x+8, y + h/2 - size*0.36, label, size, FB, color)

def stage(d, x, y, w, h, num, title, body, tint=BG):
    box(d, x, y, w, h, fill=tint, stroke=LINE, sw=0.8)
    d.add(Rect(x, y, 3.2, h, fillColor=TEAL, strokeColor=TEAL))
    cx, cy = x + 18, y + h - 15
    d.add(Circle(cx, cy, 8.2, fillColor=TEAL, strokeColor=TEAL))
    txt(d, cx, cy - 3.1, str(num), 9, FB, WHITE, anchor="middle")
    txt(d, x + 32, y + h - 18.5, title, 9.6, FB, INK)
    para(d, x + 32, y + h - 32, body, w - 42, size=7.5, color=MUT)

def build():
    W, H = 493, 726
    d = Drawing(W, H); d.hAlign = "LEFT"

    # ---------------------------------------------------------------- BAND A
    box(d, 0, 588, W, 138, fill=WHITE, stroke=LINE, sw=0.8)
    bar(d, 0, 700, W, 24, "BUILT ONCE AT STARTUP   —   50,000 products   —   about 4 seconds   —   reused by all 800 sessions", size=8.6)

    box(d, 10, 656, 104, 34, fill=PALE, stroke=TEAL, sw=0.7)
    txt(d, 62, 677, "catalog.jsonl", 8.4, FB, TEAL, "middle")
    txt(d, 62, 666, "50,000 products", 7.2, F, MUT, "middle")
    arrow(d, 116, 673, 132, 673)

    box(d, 134, 656, 208, 34, fill=PALE, stroke=TEAL, sw=0.7)
    txt(d, 238, 679, "replay the organizer's own intent_card()", 8.4, FB, TEAL, "middle")
    txt(d, 238, 668, "on every product, offline", 7.2, F, MUT, "middle")
    arrow(d, 344, 673, 360, 673)

    box(d, 362, 656, 121, 34, fill=PALE, stroke=TEAL, sw=0.7)
    txt(d, 422, 679, "the <=4 sentences", 8.4, FB, TEAL, "middle")
    txt(d, 422, 668, "each product could say", 7.2, F, MUT, "middle")

    txt(d, 10, 643, "which is inverted into five lookup tables — after this, every runtime operation is a dictionary lookup",
        7.6, F, MUT)

    idx = [("category\nbuckets", "name -> products", "50,000 -> ~180"),
           ("clue index", "sentence -> products", "91% point at 1"),
           ("word index", "word -> products", "BM25 backup"),
           ("review counts", "product -> reviews", "tie-break"),
           ("ask map", "attribute -> clues", "question value")]
    bw, gap = 92.6, 7.0
    for i, (nm, mp, note) in enumerate(idx):
        bx = 10 + i * (bw + gap)
        box(d, bx, 594, bw, 46, fill=BG, stroke=LINE, sw=0.7)
        d.add(Rect(bx, 632, bw, 8, fillColor=TEAL, strokeColor=TEAL))
        lines = nm.split("\n")
        y0 = 622 if len(lines) > 1 else 617.5
        for j, ln in enumerate(lines):
            txt(d, bx + bw/2, y0 - j*8.6, ln, 8.1, FB, INK, "middle")
        txt(d, bx + bw/2, 605, mp, 6.7, F, MUT, "middle")
        txt(d, bx + bw/2, 597, note, 6.7, F, ACC, "middle")

    arrow(d, W/2, 586, W/2, 566, w=1.4, head=5.5)

    # ---------------------------------------------------------------- BAND B
    bar(d, 0, 538, W, 24, "ONE TURN   —   repeats up to 10 times   —   the session ends the instant the target appears in the returned IDs",
        fill=ACC, size=8.6)

    SX, SW = 26, 252     # stage column
    AX, AW = 292, 201    # annotation column
    arrow(d, SX + SW/2, 536, SX + SW/2, 520)

    S = [
      (470, 48, 1, "PARSE the message",
       "Match one of 8 fixed sentence shapes. Pull out the category (turn 1) and any new clue sentences."),
      (406, 48, 2, "REMEMBER what was said",
       "Add new clues to the running list. Discard the 3 no-information replies. Mark drained attributes dead."),
      (300, 90, 3, "NARROW the candidates",
       "Five routes tried in order. The first that works wins — certainty short-circuits, it is never blended away."),
      (236, 48, 4, "RANK the survivors",
       "Order by review count, most-reviewed first. Ties broken by product ID so the run is byte-reproducible."),
      (172, 48, 5, "SCHEDULE the answer",
       "Put each candidate in the highest-value (turn, rank) slot still free. Return only the ones that land in THIS turn."),
      (108, 48, 6, "CHOOSE the next question",
       "Pick the attribute that would eliminate the most remaining products. Never re-ask a drained one."),
      (44, 48, 7, "REPLY",
       "Return the ranked IDs plus 1 question, inside a fail-safe wrapper that falls back to the last good answer."),
    ]
    tints = {3: PALE, 5: ROSE}
    for y, h, n, t, b in S:
        stage(d, SX, y, SW, h, n, t, b, tint=tints.get(n, BG))
        if y > 44: arrow(d, SX + SW/2, y - 2, SX + SW/2, y - 14)

    # inline route strip inside stage 3
    txt(d, SX + 32, 330, "tried in this order, first one that works wins:", 7.0, F, MUT)
    chips = ["category", "exact", "BM25", "widen", "popular"]
    cw, cg, cx0 = 38.0, 5.0, SX + 32
    for i, c in enumerate(chips):
        cx = cx0 + i * (cw + cg)
        box(d, cx, 308, cw, 15, fill=TEAL, stroke=TEAL)
        txt(d, cx + cw/2, 312.8, c, 6.2, FB, WHITE, "middle")
        if i < len(chips) - 1:
            d.add(Line(cx + cw + 0.6, 315.5, cx + cw + cg - 0.6, 315.5,
                       strokeColor=MUT, strokeWidth=0.7))

    # ---- annotations -------------------------------------------------
    def note(y, head, body, w=AW):
        txt(d, AX, y, head, 8.1, FB, ACC)
        para(d, AX, y - 11, body, w, size=7.3, color=MUT)

    note(511, "Why no intent classifier",
         "The simulator is templates plus regex, with no LLM. A substring check routes all 200 sessions correctly, so there is nothing here for a model to learn.")

    note(447, "The three replies that carry nothing",
         "\"not quite right yet\" · \"no additional preference for X\" (drained, never ask again) · \"no preference for X; use your judgment\" (a refusal that does NOT consume X, so ask it again).")

    # cascade beside stage 3
    txt(d, AX, 385, "The cascade, and how often each route fires", 8.1, FB, ACC)
    routes = [("R1", "filter to the category bucket", "always"),
              ("R2", "keep products that could say every clue", "131"),
              ("R3", "score the rest with BM25", "158"),
              ("R4", "widen if the category looks wrong", "rare"),
              ("R5", "category sorted by popularity", "96")]
    for i, (r, what, n) in enumerate(routes):
        ry = 360 - i*15.5
        box(d, AX, ry, AW, 13, fill=WHITE if i % 2 else BG, stroke=LINE, sw=0.5)
        d.add(Rect(AX, ry, 17, 13, fillColor=TEAL, strokeColor=TEAL))
        txt(d, AX + 8.5, ry + 3.6, r, 7.2, FB, WHITE, "middle")
        txt(d, AX + 21, ry + 3.8, what, 7.0, F, INK)
        txt(d, AX + AW - 4, ry + 3.8, n, 7.0, FB, ACC, "end")
    txt(d, AX, 272, "385 agent calls over 200 sessions. R5 needs no text", 7.0, F, MUT)
    txt(d, AX, 264, "matching at all, so no rewording can break it.", 7.0, F, MUT)

    note(243, "Why popularity, not a model",
         "Nine rankers were tested, including three embedding models and two rerankers. All lost. Tied candidates have near-identical text, but targets are real purchases: median 6,846 reviews against a catalogue median of 12.")

    note(179, "Why a schedule, not a threshold",
         "A hit at (turn t, rank r) is worth 0.50 + 0.30/r + 0.02(11-t). Rank 1 next turn beats rank 2 now, so the runner-up is held back. The session ends at the FIRST hit, so a wide early slate freezes a bad rank for good. Worth +0.009 to +0.040.")

    note(115, "Asking is free",
         "The question and the 10 IDs travel in the same reply, and the win is checked before the question is read. So the agent asks on every single turn.")

    note(51, "Fail-safe",
         "The evaluator silently swallows exceptions and turns them into an empty turn. Without the wrapper a broken agent looks merely mediocre.")

    # ---- outcome -----------------------------------------------------
    arrow(d, SX + SW/2, 42, SX + SW/2, 30)
    box(d, SX, 0, SW, 28, fill=INK, stroke=INK)
    txt(d, SX + 12, 16, "the evaluator checks the returned IDs", 8.3, FB, WHITE)
    txt(d, SX + 12, 6.5, "hit -> session ends and the rank is locked in", 7.3, F, colors.HexColor("#C9E3D9"))

    # loop-back lane
    LX = 12
    d.add(Line(SX, 14, LX, 14, strokeColor=MUT, strokeWidth=0.9))
    d.add(Line(LX, 14, LX, 494, strokeColor=MUT, strokeWidth=0.9))
    arrow(d, LX, 494, SX - 2, 494, color=MUT, w=0.9, head=3.6)
    g = Group(String(0, 0, "no hit -> next turn", fontName=F, fontSize=6.8, fillColor=MUT))
    g.transform = (0, 1, -1, 0, LX - 2.5, 200)
    d.add(g)
    return d
