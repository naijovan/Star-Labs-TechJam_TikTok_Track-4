# -*- coding: utf-8 -*-
"""Track 4 report: the diagram first, then each stage step by step, then the results."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, KeepTogether)
from tools.diagram import build as build_diagram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "docs-out", "Track4-Agent-Pipeline.pdf")

TEAL = colors.HexColor("#0F6E56"); INK  = colors.HexColor("#1B1B1B")
MUT  = colors.HexColor("#5F5E5A"); LINE = colors.HexColor("#C9C7BD")
BG   = colors.HexColor("#F1EFE8"); ACC  = colors.HexColor("#993556")
PALE = colors.HexColor("#E4EFEA"); ROSE = colors.HexColor("#F6EAEE")
BAND = colors.HexColor("#E8E6DE")
W    = 166*mm

def S(n, **kw):
    b = dict(name=n, fontName="Helvetica", fontSize=9.3, leading=13, textColor=INK)
    b.update(kw); return ParagraphStyle(**b)

TITLE = S("t",  fontName="Helvetica-Bold", fontSize=23, leading=27, spaceAfter=3)
SUB   = S("s",  fontSize=10.5, leading=14.5, textColor=TEAL, spaceAfter=13)
H1    = S("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17.5, textColor=TEAL,
          spaceBefore=13, spaceAfter=5)
BODY  = S("b",  spaceAfter=5)
LEAD  = S("ld", fontSize=10.3, leading=14.8, spaceAfter=7)
SMALL = S("sm", fontSize=8.1, leading=11.4, textColor=MUT, spaceAfter=3)
MONO  = S("m",  fontName="Courier", fontSize=8.6, leading=12)

def cell(t, sz=8.4, color=INK, align=0, font="Helvetica"):
    return Paragraph(t, S("c", fontSize=sz, leading=sz+3.1, textColor=color,
                          alignment=align, fontName=font))
def hdr(t, color=MUT):
    return Paragraph("<b>%s</b>" % t, S("ch", fontSize=7.6, leading=10.5, textColor=color))
def whdr(t):
    return Paragraph("<b>%s</b>" % t, S("wh", fontSize=8.2, leading=11.4, textColor=colors.white))
def mono(t): return Paragraph("<font face='Courier' size='8'>%s</font>" % t, S("mc", fontSize=8, leading=11))

def band(txt_, edge=TEAL, sz=10.5):
    return Paragraph("<b>%s</b>" % txt_, S("bd", fontSize=sz, leading=sz+3.5, textColor=colors.white))

# ---------------------------------------------------------------- builders
def card(title, headers, values, why, widths, tint=BG, edge=TEAL):
    """Title band, column headers, one data row, then a tinted 'why' row."""
    n = len(headers)
    rows = [[band(title, edge)] + [""]*(n-1),
            [hdr(h) for h in headers],
            [cell(v) for v in values],
            [cell(why, sz=8.3)] + [""]*(n-1)]
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("SPAN",(0,0),(-1,0)), ("SPAN",(0,3),(-1,3)),
        ("BACKGROUND",(0,0),(-1,0),edge),
        ("BACKGROUND",(0,1),(-1,1),BAND),
        ("BACKGROUND",(0,3),(-1,3),tint),
        ("BOX",(0,0),(-1,-1),0.6,edge),
        ("LINEBELOW",(0,1),(-1,1),0.4,LINE),
        ("LINEBELOW",(0,2),(-1,2),0.5,edge),
        ("LINEAFTER",(0,1),(-2,2),0.4,LINE),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),3.8),("BOTTOMPADDING",(0,0),(-1,-1),3.8),
        ("TOPPADDING",(0,0),(-1,0),4.5),("BOTTOMPADDING",(0,0),(-1,0),4.5),
    ]))
    return t

def session(tag, share, sid, title, reviews, rows, verdict, tint=BG, edge=TEAL):
    body = [[band("%s   &mdash;   %s" % (tag, share), edge)] + ["", "", ""],
            [cell("<font color='#5F5E5A' size='7.6'>hidden target: %s &mdash; %s reviews "
                  "&nbsp;&middot;&nbsp; %s</font>" % (title, reviews, sid), sz=7.6)] + ["", "", ""],
            [hdr("Turn"), hdr("What the shopper says"), hdr("What the agent does"), hdr("Left")]]
    for t, msg, act, left in rows:
        body.append([cell("<b>%s</b>" % t, align=1), cell(msg, sz=8.2),
                     cell(act, sz=8.2), cell(left, align=1)])
    last = len(body)
    body.append([cell(verdict, sz=8.3)] + ["", "", ""])
    t = Table(body, colWidths=[11*mm, 62*mm, 79*mm, 14*mm])
    st = [("VALIGN",(0,0),(-1,-1),"TOP"),
          ("SPAN",(0,0),(-1,0)), ("SPAN",(0,1),(-1,1)), ("SPAN",(0,last),(-1,last)),
          ("BACKGROUND",(0,0),(-1,0),edge),
          ("BACKGROUND",(0,1),(-1,1),colors.white),
          ("BACKGROUND",(0,2),(-1,2),BAND),
          ("BACKGROUND",(0,last),(-1,last),tint),
          ("BOX",(0,0),(-1,-1),0.6,edge),
          ("LINEBELOW",(0,2),(-1,2),0.4,LINE),
          ("LINEBELOW",(0,last-1),(-1,last-1),0.5,edge),
          ("LINEAFTER",(0,2),(-2,last-1),0.4,LINE),
          ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
          ("TOPPADDING",(0,0),(-1,-1),4.5),("BOTTOMPADDING",(0,0),(-1,-1),4.5)]
    for i in range(3, last-1):
        st.append(("LINEBELOW",(0,i),(-1,i),0.35,LINE))
    t.setStyle(TableStyle(st))
    return t

def table(rows, widths, header=True, zebra=True, extra=None):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN",(0,0),(-1,-1),"TOP"),
          ("GRID",(0,0),(-1,-1),0.4,LINE),
          ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
          ("TOPPADDING",(0,0),(-1,-1),4.2),("BOTTOMPADDING",(0,0),(-1,-1),4.2)]
    if header: st.append(("BACKGROUND",(0,0),(-1,0),TEAL))
    if zebra:
        for i in range(1 + (1 if header else 0), len(rows), 2):
            st.append(("BACKGROUND",(0,i),(-1,i),BG))
    if extra: st += extra
    t.setStyle(TableStyle(st)); return t

def callout(title, body, fill=BG, edge=TEAL, w=W):
    t = Table([[Paragraph("<b>%s</b>" % title, S("bt", fontSize=9.5, leading=13, textColor=edge))],
               [Paragraph(body, S("bb", fontSize=8.8, leading=12.4))]], colWidths=[w])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),fill),
        ("BOX",(0,0),(-1,-1),0.5,LINE), ("LINEBEFORE",(0,0),(0,-1),2.6,edge),
        ("LEFTPADDING",(0,0),(-1,-1),9),("RIGHTPADDING",(0,0),(-1,-1),9),
        ("TOPPADDING",(0,0),(-1,-1),5.5),("BOTTOMPADDING",(0,0),(-1,-1),5.5)]))
    return t

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.4); canvas.setFillColor(MUT)
    canvas.drawString(22*mm, 12*mm, "TikTok TechJam 2026  |  Track 4 Shopping Copilot  |  agent pipeline")
    canvas.drawRightString(188*mm, 12*mm, "%d" % doc.page)
    canvas.setStrokeColor(LINE); canvas.setLineWidth(0.4)
    canvas.line(22*mm, 15.5*mm, 188*mm, 15.5*mm)
    canvas.restoreState()

E = []

# ============================================================ 1  the problem
E += [Paragraph("How the agent works", TITLE),
      Paragraph("TikTok TechJam 2026 &middot; Track 4 &middot; "
                "Shopping Copilot: AI Conversational Search and Recommendations", SUB)]

E += [Paragraph("The problem, in one sentence", H1),
 Paragraph("A simulated shopper has already bought one specific product out of 50,000. The agent never sees "
   "which one. It talks to them for up to 10 turns, and every turn it hands back 10 product IDs. "
   "The moment the right product appears in that list the session ends, and the rank it appeared at is "
   "locked in permanently.", LEAD),
 callout("The one insight the whole design rests on",
   "Every clue the shopper gives is copied <b>word for word</b> out of the real product's own description. "
   "So the agent never asks <i>&quot;which product is most similar to this text?&quot;</i> &mdash; it asks "
   "<b><i>&quot;which products could have said that?&quot;</i></b> and keeps only the ones that survive every "
   "clue. That turns a ranking problem into a set-intersection problem. <b>91% of the 60,670 distinct clue "
   "sentences in the catalogue point at exactly one product.</b>", fill=PALE)]

E += [Paragraph("How it is scored", H1),
 Paragraph("TechnicalScore = 0.50 &times; HitRate@10  +  0.30 &times; MRR  +  0.20 &times; Efficiency", MONO),
 Spacer(1,5),
 table([[whdr("Term"), whdr("Means"), whdr("What it pushes the design towards")],
   [cell("HitRate@10"), cell("how often the product is found at all"),
    cell("never return nothing &mdash; always have a fallback")],
   [cell("MRR"), cell("1 / rank &mdash; rewards first over tenth"),
    cell("do not answer into a large candidate set")],
   [cell("Efficiency"), cell("clip((11 &minus; turns) / 10, 0, 1)"),
    cell("be quick, but a miss counts as 11 turns")]],
  [26*mm, 62*mm, 78*mm]),
 Spacer(1,4),
 Paragraph("Because a miss costs 11 turns, <b>finding the product matters far more than finding it fast</b>. "
   "That single fact is why the agent is built to eliminate rather than to guess early.", BODY)]

E += [Paragraph("The result, up front", H1),
 table([[whdr("Measured on all 200 public sessions, deterministic"), whdr("Result")],
   [cell("Found the right product"), cell("<b>200 of 200 sessions</b> &mdash; hit@10 = 1.000")],
   [cell("Found it at rank 1"), cell("<b>192 of 200 sessions</b>, and never worse than rank 7")],
   [cell("Turns used on average"), cell("<b>1.93</b>, and never more than 5")],
   [cell("TechnicalScore"), cell("<b>0.97359</b>, against 0.10671 for the shipped starter agent")]],
  [78*mm, 88*mm])]

E += [Paragraph("What is in this document", H1),
 table([[whdr("Page"), whdr("")],
   [cell("<b>2</b>", align=1), cell("<b>The whole pipeline on one page</b> &mdash; the diagram")],
   [cell("3", align=1), cell("Reading the diagram: the five tables built at startup")],
   [cell("4&ndash;5", align=1), cell("The seven stages, step by step, with the evidence for each")],
   [cell("6&ndash;7", align=1), cell("Four real sessions, one of each shopper type, traced live")],
   [cell("<b>8&ndash;10</b>", align=1), cell("<b>Results</b> &mdash; headline, ablation, robustness, and the honest limits")]],
  [16*mm, 150*mm])]

# ============================================================ 2  the diagram
E += [PageBreak(), build_diagram()]

# ============================================================ 3  the indexes
E += [PageBreak(), Paragraph("Reading the diagram", TITLE), Spacer(1,3),
 Paragraph("The top half of the diagram happens once. The bottom half happens on every turn.", LEAD),
 Paragraph("The top half &mdash; five tables, built once", H1),
 Paragraph("At startup the agent replays the organizer's own <font face='Courier' size='8'>intent_card()</font> "
   "function over all 50,000 products. That function is what generates the shopper's clues, so replaying it "
   "offline tells the agent <b>exactly which sentences each product is capable of saying</b>. Those sentences "
   "are inverted into the five tables below, in about four seconds.", BODY), Spacer(1,3)]

E += [callout("Why five tables is enough",
  "Between them they answer the only four questions the agent ever asks: <i>what is in this category</i>, "
  "<i>who could have said this sentence</i>, <i>who shares rare words with it</i>, and <i>which of these did a "
  "real person actually buy</i>. A knowledge graph was rejected: the relations it would encode (material, "
  "colour, use case) <i>are</i> the clue strings already.", fill=PALE), Spacer(1,6)]

IW = [56*mm, 76*mm, 34*mm]
for t, mp, ex, used, why in [
 ("1.  Category buckets", "category name -&gt; the products in it",
  "&quot;Basketball Men&quot; -&gt; 13 products", "Stage 3, route R1",
  "<b>Why it matters.</b> The shopper names their category for free in the very first sentence, and this one "
  "lookup cuts 50,000 products down to about 180 &mdash; the biggest single reduction in the pipeline. Category "
  "plus popularity <b>alone</b>, with no clue parsing at all, already scores 0.832."),
 ("2.  Clue index", "one exact sentence -&gt; the products that could say it",
  "&quot;High quality mesh for maximum breathability&quot; -&gt; 1 product", "Stage 3, route R2",
  "<b>Why it matters.</b> The engine of the whole approach, and the highest-value table by a wide margin: removing "
  "it costs 0.209. It is built over what a product <b>would say</b>, not over its raw text &mdash; indexing the raw "
  "text instead measured 0.212 MRR worse."),
 ("3.  Word index (BM25)", "each word -&gt; the products whose clues contain it, plus that word's rarity",
  "&quot;breathability&quot; is rare, so it counts heavily; &quot;cotton&quot; is common, so it barely counts",
  "Stage 3, route R3",
  "<b>Why it matters.</b> The backup for when a clue does not match a table entry exactly. This is the part that "
  "keeps the agent working if the shopper rewords things, which is the one failure mode that matters."),
 ("4.  Review counts", "product -&gt; how many reviews it has",
  "Pro Club shorts -&gt; 3,042 reviews", "Stage 4",
  "<b>Why it matters.</b> Used to break ties. The hidden targets are real purchases drawn by a leave-last-out "
  "split, so they skew heavily popular: the catalogue median is 12 reviews while the target median is 6,846, a "
  "570&times; gap. When several products match the clues equally well, the most-reviewed one is usually the answer."),
 ("5.  Ask map", "attribute -&gt; which clues count as that attribute -&gt; which products",
  "&quot;color&quot; -&gt; [&quot;color: black&quot;, &quot;color: blue&quot;, ...]", "Stage 6",
  "<b>Why it matters.</b> Lets the agent work out which question would rule out the most products, rather than "
  "asking in a fixed order."),
]:
    E += [KeepTogether([card(t, ["What it maps", "Example", "Used by"], [mp, ex, used], why, IW),
                        Spacer(1,3)])]


# ============================================================ 4-5  the stages
STAGES = [
 (1, "PARSE the message", "the raw sentence the shopper just sent",
  "Works out which of 8 fixed sentence shapes this is, then pulls out the useful parts: the leaf category on "
  "turn 1, or the new clue sentences on later turns. Clue sentences are split carefully &mdash; 17.2% of them "
  "contain a semicolon of their own, and naive splitting shattered the target's own clue in 5 of the 200 "
  "sessions. If no template matches at all, the agent falls back to scanning the sentence for any known "
  "category name.",
  "the category, and any new clues",
  "<b>Why it is built this way.</b> The simulator is templates plus regex, with no language model behind it. "
  "Shape detection is a plain substring check and is <b>100% accurate on all 200 sessions</b>, so an intent "
  "classifier here would spend effort on a solved problem. Getting the category right, by contrast, is critical: "
  "a wrong category makes the target unreachable, which is why a fallback scan sits behind the template."),
 (2, "REMEMBER what was said", "the parsed clues",
  "Adds the new clues to everything already known this session, and throws away the three replies that carry no "
  "information. Two of those look almost identical but mean opposite things, and the agent tells them apart: "
  "<i>&quot;no additional preference for X&quot;</i> means that attribute is drained and must never be asked "
  "again, while <i>&quot;no preference for X; please use your judgment&quot;</i> is a one-off refusal that does "
  "<b>not</b> consume the attribute &mdash; so asking it again next turn gets a real answer.",
  "the running clue list, and the drained attributes",
  "<b>Why it is built this way.</b> The starter agent scores 0.107 largely because it forgets everything between "
  "turns and then searches the filler sentences as though they described a product. Telling the two refusals "
  "apart was worth <b>+0.0047</b> and is what took hit@10 from 0.995 to 1.000 &mdash; the single largest gain in "
  "the final round of work."),
 (3, "NARROW the candidates", "the category and every clue so far",
  "Five routes, tried in order, first one that works wins. <b>R1</b> filters to the category bucket. "
  "<b>R2</b> keeps only products that could say every clue &mdash; if exactly one survives, that is the answer. "
  "<b>R3</b> otherwise scores the survivors with BM25 over rare words. <b>R4</b> re-runs the search "
  "catalogue-wide if the category was only guessed and the scores come out flat. <b>R5</b> falls back to the "
  "category sorted by popularity.",
  "a shortlist of candidates",
  "<b>Why it is built this way.</b> This is a cascade, not a fusion. An exact single-product match is "
  "<i>certainty</i>; blending it into a fused ranking would flatten it to 1/61, where two mediocre agreeing "
  "rankers could outvote it. So certainty short-circuits, and fusion applies only to the uncertain path. "
  "R5 is load-bearing precisely because it needs <b>no clue parsing at all</b>: no amount of rewording can break "
  "it, and it floors the agent at 0.832."),
 (4, "RANK the survivors", "the shortlist",
  "Orders them by review count, most-reviewed first, with the product ID as a final tie-break so that two runs "
  "of the same agent produce byte-identical output.",
  "an ordered list of product IDs",
  "<b>Why it is built this way.</b> Nine ranking methods were tested on the hard cases, including bge-small, "
  "bge-base and gte-base embeddings and two cross-encoder rerankers. <b>Review count beat all of them</b>, and "
  "the cross-encoder actively made things worse (0.977 to 0.953). When products survive the same clues their "
  "descriptions are near-identical, so meaning carries no signal &mdash; but popularity carries a 570&times; one. "
  "The tie-break is not cosmetic: without it, Python's set iteration order swung the score by &plusmn;0.03 "
  "between runs of the same code."),
 (5, "SCHEDULE the answer", "the shortlist and the turn number",
  "Works out which of the remaining (turn, rank) slots are worth the most score, and puts each candidate into the "
  "best one still free. A hit at turn <i>t</i>, rank <i>r</i> is worth 0.50 + 0.30/<i>r</i> + 0.02(11 &minus; <i>t</i>), "
  "and a rank-1 slot next turn beats a rank-2 slot now &mdash; so the runner-up is <b>held back</b> rather than "
  "shown. Only the candidates landing in this turn are returned, which early on is often just one or two.",
  "the IDs that belong in this turn",
  "<b>Why it is built this way.</b> The session ends at the first correct guess and that rank is locked in "
  "<b>permanently</b>, so a wide speculative page can hit the target by accident at rank 8 and freeze it there "
  "for good. Scheduling by slot value avoids that by construction rather than by a tuned threshold, and it "
  "improved <b>every one of the nine robustness conditions</b> measured &mdash; by +0.009 on clean data and up to "
  "+0.040 under corruption. It also removed a real defect: the previous rule returned the same ten products every "
  "turn once it committed, so a target at rank 11 was unreachable. Turn-1 wins rose from 17 sessions to 85."),
 (6, "CHOOSE the next question", "the shortlist and the drained attributes",
  "Picks the attribute whose answer would eliminate the most remaining products, skipping anything already "
  "answered or drained.", "one attribute name",
  "<b>Why it is built this way.</b> Asking costs nothing: the question and the 10 IDs travel in the same reply, "
  "and the win is checked <b>before</b> the question is even read. So the agent asks on every single turn "
  "&mdash; there is never a reason not to. Removing the question entirely costs between 0.021 and 0.037."),
 (7, "REPLY", "the ranked list and the chosen question",
  "Returns the ranked IDs plus a question, wrapped so that any internal error falls back to the last good answer "
  "instead of returning nothing.", "the response the evaluator scores",
  "<b>Why it is built this way.</b> The evaluator silently swallows exceptions and converts them into an empty "
  "turn. Without the wrapper a broken agent would look merely mediocre rather than broken, and the bug would "
  "never be found."),
]
SW_ = [33*mm, 100*mm, 33*mm]
E += [PageBreak(), Paragraph("The seven stages, step by step", TITLE), Spacer(1,3),
 Paragraph("Every shopper message runs through these seven stages in order. Each card says what goes in, what "
   "happens, what comes out, and the evidence that it is right.", LEAD), Spacer(1,3)]
for i, (n, name, inp, does, out, why) in enumerate(STAGES):
    if n == 4:
        E += [PageBreak(), Paragraph("The seven stages, continued", TITLE), Spacer(1,7)]
    tint = {3: PALE, 5: ROSE}.get(n, BG)
    E += [KeepTogether([card("%d.  %s" % (n, name), ["Takes in", "What it does", "Passes on"],
                             [inp, does, out], why, SW_, tint=tint), Spacer(1,7)])]

E += [Spacer(1,2), callout("The loop, in one line",
  "Stages 1 to 7 run again on every reply until the target appears in the returned IDs. In practice that is 1.93 "
  "turns on average, and never more than 5 &mdash; on any of the 200 sessions, under any of the "
  "corruption tests on page 9. 85 of the 200 sessions end on turn 1.", fill=PALE)]

# ============================================================ 6-7  sessions
E += [PageBreak(), Paragraph("Four real sessions", TITLE), Spacer(1,3),
 Paragraph("One of each shopper type, traced live through the agent. The messages are exactly what the simulator "
   "sent; the candidate counts are exactly what the agent held at that moment.", LEAD), Spacer(1,4)]

E += [KeepTogether([session("BUYING", "40% of sessions", "public_0001",
  "Celtic Knot Triple Moon Pentagram Pendant Necklace", "490", [
   ("1", "&quot;I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.&quot;",
    "The shape says <b>buying</b>, so a clue arrives free on turn 1. Category and clue together leave just 2 "
    "candidates, and the better of the two takes the rank-1 slot.", "2")],
  "<b>Rank 1 on turn 1.</b> A perfect session, and the reason buying sessions average 1.44 turns.", tint=PALE),
 Spacer(1,9)])]

E += [KeepTogether([session("BROWSING", "40% of sessions", "public_0006",
  "Pro Club Men's Heavyweight Mesh Basketball Shorts", "3,042", [
   ("1", "&quot;I'm looking for Basketball Men, but I'm still exploring.&quot;",
    "&quot;still exploring&quot; means no clue is coming this turn. Only the category is known, so route R5 "
    "fires: the 13 products in the bucket, most-reviewed first. The 3,042-review target is the most reviewed of "
    "the thirteen, so it takes the rank-1 slot.", "13")],
  "<b>Rank 1 on turn 1, before a single clue existed.</b> Popularity alone is enough here because the bucket is "
  "small. The agent still asks a question in the same reply, at no cost, in case it was wrong."),
 Spacer(1,9)])]

E += [KeepTogether([session("BOUNDARY", "5% of sessions", "public_0035",
  "Skechers Men's Go Max Air Mesh Slip-On Walking Shoe", "35,329", [
   ("1", "&quot;I'm looking for Athletic Walking, but I'm still exploring.&quot;",
    "Category only, and 342 products in the bucket. Almost every candidate is scheduled into a later rank-1 "
    "slot rather than shown now. Ask.", "342"),
   ("2", "&quot;I don't have a preference for other; please use your judgment.&quot;",
    "A refusal. The critical detail: this reply does <b>not</b> consume the attribute, so the correct move is to "
    "ask the very same thing again rather than mark it dead.", "342"),
   ("3", "&quot;For that, what matters is: fabric; 100% Textile.&quot;",
    "The re-ask paid off. Two clues cut 342 down to 4, and the target takes the rank-1 slot.", "4")],
  "<b>Rank 1 on turn 3.</b> Marking the attribute dead after turn 2 would have thrown this session away &mdash; "
  "the agent would never have learned the fabric. Boundary sessions now average 2.30 turns at MRR 0.950.", tint=ROSE),
 Spacer(1,9)])]

E += [PageBreak(), Paragraph("Four real sessions, continued", TITLE), Spacer(1,8)]

E += [KeepTogether([session("INTENT OVERRIDE", "15% of sessions", "public_0002",
  "Hide &amp; Drink Rustic Handmade Full Grain Leather Belt", "6,614", [
   ("1", "&quot;I'm looking for Accessories Belts. Buckle closure&quot;",
    "A bare fragment with no template marker &mdash; the signature of an override session. 90 candidates.", "90"),
   ("2", "&quot;For that, what matters is: leather; 100% Leather.&quot;",
    "Three clues now, 16 candidates. The target is already ranked first, but the evaluator will not award the "
    "hit until the mind-change has fired, so the score cannot be banked yet.", "16"),
   ("3", "&quot;Actually, ignore my earlier preference. What I need is: leather.&quot;",
    "The gate opens. The old clues are kept, not discarded, because a proof test shows they still intersect "
    "with the new one &mdash; both were drawn from the same product.", "16")],
  "<b>Rank 1 on turn 3</b>, the earliest this session type can possibly score. Discarding the earlier clues on "
  "&quot;ignore my earlier preference&quot; would have thrown away true information: the simulator draws both "
  "the old and the new value from the same hidden card, so the old one never stopped being true.")])]

E += [Spacer(1,10), KeepTogether([session("THE WORST SESSION OF THE 200", "a browsing session", "public_0167",
  "Champion Women's Absolute Sports Bra", "552", [
   ("1", "&quot;I'm looking for Bras Sports Bras, but I'm still exploring.&quot;",
    "No clue at all, so route R5 fires: the 166 products in the bucket, most-reviewed first. The scheduler "
    "deliberately spreads its turn-1 picks down the ranking rather than taking the top ten in order &mdash; and "
    "the target, 24th by review count, lands in slot 7.", "166")],
  "<b>Rank 7 on turn 1 &mdash; reciprocal rank 0.143, the worst of all 200 sessions.</b> This is an accidental "
  "hit: the agent had heard nothing yet and was covering ground, not answering. Because the evaluator ends the "
  "session at the first hit and keeps that rank for good, a lucky early appearance at a poor rank is worse than "
  "no appearance at all &mdash; one more turn would have brought a clue and almost certainly rank 1. It is the "
  "residual form of exactly the problem the scheduler was adopted to fix, and the reason the remaining headroom "
  "sits in browsing sessions.", tint=ROSE, edge=ACC)])]

# ============================================================ 8  results
NUM = [("ALIGN",(1,1),(-1,-1),"CENTER")]
E += [PageBreak(), Paragraph("Results", TITLE), Spacer(1,3),
 Paragraph("Everything below was produced by running the organizer's own evaluator over all 200 public sessions. "
   "The agent is deterministic: five different hash seeds all return 0.973589 to the last digit.", LEAD)]

E += [Paragraph("Headline", H1),
 table([[whdr("Condition"), whdr("hit@10"), whdr("MRR"), whdr("Turns"), whdr("Score")],
   [cell("<b>Clean, exactly as the organizer ships it</b>"), cell("<b>1.000</b>", align=1),
    cell("<b>0.974</b>", align=1), cell("<b>1.93</b>", align=1), cell("<b>0.97359</b>", align=1)],
   [cell("Reference: category + popularity only, no clue parsing"), cell("0.980", align=1),
    cell("0.544", align=1), cell("2.06", align=1), cell("0.83199", align=1)],
   [cell("Reference: the shipped starter agent"), cell("0.125", align=1),
    cell("0.068", align=1), cell("9.81", align=1), cell("0.10671", align=1)]],
  [80*mm, 20*mm, 20*mm, 20*mm, 26*mm])]

E += [Paragraph("Where the score comes from", H1),
 table([[whdr("Turn the target was first found"), whdr("Sessions"),
         whdr("Rank it was found at"), whdr("Sessions")],
   [cell("turn 1"), cell("<b>85</b>", align=1), cell("rank 1"), cell("<b>192</b>", align=1)],
   [cell("turn 2"), cell("68", align=1), cell("rank 2"), cell("4", align=1)],
   [cell("turn 3"), cell("26", align=1), cell("rank 4"), cell("1", align=1)],
   [cell("turn 4"), cell("19", align=1), cell("rank 6 or 7"), cell("3", align=1)],
   [cell("turn 5"), cell("2", align=1), cell("<b>rank 8 or worse, or never found</b>"), cell("<b>0</b>", align=1)]],
  [52*mm, 22*mm, 52*mm, 22*mm]),
 Spacer(1,6),
 table([[whdr("Shopper type"), whdr("Share"), whdr("Clue on turn 1?"), whdr("hit@10"), whdr("MRR"), whdr("Turns")],
   [cell("buying"), cell("40%", align=1), cell("yes, free"), cell("1.000", align=1),
    cell("0.983", align=1), cell("1.44", align=1)],
   [cell("browsing"), cell("40%", align=1), cell("no"), cell("1.000", align=1),
    cell("0.957", align=1), cell("1.71", align=1)],
   [cell("boundary"), cell("5%", align=1), cell("no, and refuses once"), cell("1.000", align=1),
    cell("0.950", align=1), cell("2.30", align=1)],
   [cell("intent override"), cell("15%", align=1), cell("a bare fragment"), cell("1.000", align=1),
    cell("<b>1.000</b>", align=1), cell("3.67", align=1)]],
  [34*mm, 16*mm, 40*mm, 22*mm, 22*mm, 20*mm]),
 Spacer(1,3),
 Paragraph("Override and boundary sessions take longer by construction, not by weakness: neither can be scored "
   "before the mind-change or the refusal has arrived.", SMALL)]

E += [Paragraph("What each part is actually worth", H1),
 Paragraph("Each row is the same pipeline with one thing removed, on the same 200 sessions.", BODY),
 table([[whdr("Remove this"), whdr("Cost"), whdr("Why it matters that much")],
   [cell("the clue index, route R2"), cell("<b>&minus;0.209</b>", align=1),
    cell("Set intersection is the primary path; scoring cannot tell near-identical products apart.")],
   [cell("the popularity tie-break"), cell("&minus;0.048", align=1),
    cell("Targets are real purchases, so they are 570&times; more reviewed than the catalogue median.")],
   [cell("the category from turn 1"), cell("&minus;0.040", align=1),
    cell("Free information, and the largest single reduction in the candidate set.")],
   [cell("the slot-value scheduler"), cell("&minus;0.009 to &minus;0.040", align=1),
    cell("Showing ten candidates while unsure hits the target by accident at a poor rank, which the evaluator "
         "then keeps forever. Worth least on clean data and most under corruption.")],
   [cell("asking a question at all"), cell("&minus;0.021 to &minus;0.037", align=1),
    cell("Asking is free, so never asking is pure loss.")]],
  [40*mm, 26*mm, 100*mm])]

# ============================================================ 10  robustness
E += [PageBreak(), Paragraph("Robustness", TITLE), Spacer(1,3),
 Paragraph("The public simulator is fixed, but the private 800 sessions are not visible. So the agent was re-run "
   "against deliberately corrupted messages. Corruption is a pure function of the message text, which means it "
   "is identical no matter how many turns the agent takes &mdash; the comparison is fair.", LEAD),
 table([[whdr("What was broken"), whdr("hit@10"), whdr("MRR"), whdr("Turns"), whdr("Score")],
   [cell("nothing &mdash; clean"), cell("1.000", align=1), cell("0.974", align=1),
    cell("1.93", align=1), cell("<b>0.97359</b>", align=1)],
   [cell("20% of the words in every clue deleted"), cell("1.000", align=1), cell("0.966", align=1),
    cell("2.06", align=1), cell("0.96869", align=1)],
   [cell("35% of the words in every clue deleted"), cell("1.000", align=1), cell("0.970", align=1),
    cell("2.12", align=1), cell("0.96869", align=1)],
   [cell("50% of the words in every clue deleted"), cell("1.000", align=1), cell("0.963", align=1),
    cell("2.10", align=1), cell("0.96686", align=1)],
   [cell("category words scrambled into a different order"), cell("0.950", align=1), cell("0.924", align=1),
    cell("2.40", align=1), cell("0.92409", align=1)],
   [cell("category scrambled <b>and</b> 35% of clue words deleted"), cell("0.950", align=1),
    cell("0.920", align=1), cell("2.58", align=1), cell("0.91949", align=1)],
   [cell("<b>both message templates rewritten from scratch</b>"), cell("0.825", align=1),
    cell("0.723", align=1), cell("4.16", align=1), cell("<b>0.76605</b>", align=1)]],
  [72*mm, 20*mm, 20*mm, 20*mm, 34*mm]),
 Spacer(1,6),
 callout("What this says",
   "Half the clue words can be destroyed and the agent still finds <b>every single target</b> (hit@10 1.000) at "
   "0.967 &mdash; because whatever survives is still a verbatim fragment, and BM25 over rare words finds it. "
   "Scrambling the category hurts far more than damaging the clues, which is exactly why a token-set fallback "
   "sits behind the category match. And even with <b>both templates rewritten entirely</b>, so that no clue "
   "string can ever match again, the agent holds 0.766 &mdash; seven times the shipped baseline &mdash; because "
   "route R5 needs no text matching at all.",
   fill=PALE)]

E += [Paragraph("Things that were tried and rejected", H1),
 Paragraph("Each was implemented behind a switch and measured before being accepted or dropped. None survived.", BODY),
 table([[whdr("Idea"), whdr("MRR on the hard cases"), whdr("Why it lost")],
   [cell("bge-small / bge-base / gte-base embeddings"), cell("0.922 vs 0.977", align=1),
    cell("Tested fairly, with the correct query prefix. Products that survive the same clues have "
         "near-identical text, so there is nothing for a semantic model to read.")],
   [cell("cross-encoder reranking"), cell("0.953 vs 0.977", align=1),
    cell("Actively made things worse. A perfect ranker on the residual hard cases would be worth 0.006.")],
   [cell("crossing off products already shown"), cell("&minus;0.0015 <font size='6.5'>score</font>", align=1),
    cell("Logically sound, but the agent converges at 2.40 turns with hit@10 1.000 &mdash; there is almost no "
         "session left to eliminate over.")],
   [cell("a per-turn expected-value scheduler"), cell("&plusmn;0.0015 <font size='6.5'>score</font>", align=1),
    cell("Every answer-timing policy tested lands within 0.0015 of the simple rule. The gain it claims is the "
         "gain over answering greedily, which is already banked.")],
   [cell("<font face='Courier' size='8'>difflib</font> for a missed category"), cell("<b>0.298</b> <font size='6.5'>score</font>", align=1),
    cell("Catastrophic, and caught only by testing it. Scrambling word order preserves the word set but "
         "destroys the character sequence, and difflib matches sequences. Replaced with token-set overlap: "
         "0.298 to 0.899.")]],
  [43*mm, 28*mm, 95*mm])]

# ============================================================ 10  limits
E += [PageBreak(), Paragraph("The honest limits", TITLE), Spacer(1,3),
 callout("What this agent cannot do",
   "About 15% of sessions cannot be scored before turn 3 no matter what the agent does &mdash; the shopper simply "
   "has not spoken yet &mdash; and six targets can never be ranked first, because a more popular product survives "
   "every clue they do. Our oracle probe puts the bound at <b>0.97578</b>, and at <b>0.97359</b> the agent is "
   "within <b>0.002</b> of it. That probe assumes the oracle answers on a single optimal turn, though, and the "
   "scheduler wins 85 sessions on turn 1 against the oracle's 77 &mdash; so treat 0.976 as a model-specific "
   "bound rather than a true ceiling.",
   fill=ROSE, edge=ACC),
 Spacer(1,7),
 table([[whdr("Risk"), whdr("What was done about it"), whdr("Residual")],
   [cell("Overfitting to the public 200"),
    cell("Constants were tuned on one half and tested on the other half."),
    cell("Gap of <b>0.001</b>. Only one constant is load-bearing, and turn 3 is principled: it is when the clue "
         "pool runs dry.")],
   [cell("The private simulator may paraphrase"),
    cell("Re-measured under six corruption regimes, up to and including rewriting both templates."),
    cell("Worst measured case <b>0.766</b>, still 7&times; the baseline. Route R5 is the reason.")],
   [cell("Targets may be less popularity-skewed in private"),
    cell("Measured the agent with the popularity prior removed entirely."),
    cell("Floor of <b>0.899</b>, and it is the mechanism behind the worst session on page 7. Only 274 catalogue "
         "products clear the public median target popularity, so the private 800 cannot be skewed as hard.")],
   [cell("Non-reproducible scoring"),
    cell("Added a product-ID tie-break to every sort and removed set-iteration dependence."),
    cell("Byte-identical <b>0.973589</b> across five hash seeds. Before the fix, run-to-run noise was "
         "&plusmn;0.03 &mdash; enough to swamp every effect being measured.")]],
  [42*mm, 56*mm, 68*mm])]

E += [Paragraph("Every library used", H1),
 Paragraph("All six ship with Python. There is no machine learning framework, no model weights, no downloads and "
   "no network access. The requirements file for this submission is empty, and the agent was verified to run "
   "under <font face='Courier' size='8'>python3 -S</font> with no third-party module loaded.", BODY),
 table([[whdr("Library"), whdr("Standard?"), whdr("What it does here")],
   [mono("json"), cell("yes", align=1), cell("reads the 50,000-product catalogue, one product per line")],
   [mono("re"), cell("yes", align=1), cell("splits text into words, and matches the 8 sentence shapes")],
   [mono("collections"), cell("yes", align=1),
    cell("Counter holds the BM25 scores; defaultdict holds the inverted indexes")],
   [mono("math"), cell("yes", align=1), cell("the logarithm inside BM25 that makes rare words count more")],
   [mono("difflib"), cell("yes", align=1), cell("secondary fuzzy pass for a genuinely mistyped category name")],
   [mono("pathlib"), cell("yes", align=1), cell("file paths")]],
  [32*mm, 24*mm, 110*mm])]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=22*mm, rightMargin=22*mm,
                        topMargin=16*mm, bottomMargin=20*mm,
                        title="Track 4 Shopping Copilot - agent pipeline", author="Jovan Nai")
doc.build(E, onFirstPage=footer, onLaterPages=footer)
print("WROTE", OUT)
