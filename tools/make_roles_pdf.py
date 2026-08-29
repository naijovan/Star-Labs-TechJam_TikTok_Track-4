from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                KeepTogether, PageBreak)

OUT="/Users/naijovan/Projects/techjam/techjam-conversational-search/docs-out/Track4-Team-Roles.pdf"
TEAL=colors.HexColor("#0F6E56"); INK=colors.HexColor("#1B1B1B")
MUT=colors.HexColor("#5F5E5A"); LINE=colors.HexColor("#D3D1C7")
BG=colors.HexColor("#F1EFE8"); RED=colors.HexColor("#993556")

def S(n,**kw):
    b=dict(name=n,fontName="Helvetica",fontSize=9.5,leading=13.5,textColor=INK); b.update(kw)
    return ParagraphStyle(**b)
TITLE=S("t",fontName="Helvetica-Bold",fontSize=21,leading=25,spaceAfter=3)
SUB=S("s",fontSize=10.5,leading=14,textColor=TEAL,spaceAfter=12)
H1=S("h1",fontName="Helvetica-Bold",fontSize=13.5,leading=17,textColor=TEAL,spaceBefore=14,spaceAfter=6)
H2=S("h2",fontName="Helvetica-Bold",fontSize=9.8,leading=13,spaceBefore=8,spaceAfter=3)
BODY=S("b",spaceAfter=5)
BUL=S("bu",fontSize=9.3,leading=13,leftIndent=11,bulletIndent=2,spaceAfter=2.5)
SMALL=S("sm",fontSize=8.5,leading=11.5,textColor=MUT)

def bullets(items): return [Paragraph(t,BUL,bulletText="•") for t in items]
def cell(t,sz=8.8): return Paragraph(t,S("c",fontSize=sz,leading=12))
def hd(t): return Paragraph(f"<b>{t}</b>",S("ch",fontSize=8.8,leading=12,textColor=colors.white))
def tbl(rows,w,head=True):
    t=Table(rows,colWidths=w,repeatRows=1 if head else 0)
    st=[("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),0.4,LINE),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]
    if head: st.append(("BACKGROUND",(0,0),(-1,0),TEAL))
    t.setStyle(TableStyle(st)); return t
def box(title,body,col=BG):
    t=Table([[Paragraph(f"<b>{title}</b>",S("x",fontSize=9.5,leading=13))],
             [Paragraph(body,S("y",fontSize=9,leading=12.5))]],colWidths=[166*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),col),("BOX",(0,0),(-1,-1),0.4,LINE),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    return t
def foot(c,d):
    c.saveState(); c.setFont("Helvetica",7.5); c.setFillColor(MUT)
    c.drawString(22*mm,12*mm,"TikTok TechJam 2026 - Track 4 - team roles")
    c.drawRightString(188*mm,12*mm,f"page {d.page}")
    c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(22*mm,15*mm,188*mm,15*mm); c.restoreState()

def role(tag,name,mission,why,tasks,owns,done):
    out=[KeepTogether([Paragraph(f"{tag} &mdash; {name}",H1),
         Paragraph(f"<b>{mission}</b>",BODY),
         Paragraph(why,BODY),
         Paragraph("Tasks",H2)])]
    out+=bullets(tasks)
    out+=[Spacer(1,4),
      tbl([[hd("Owns these files"),hd("Done when")],
           [cell(owns),cell(done)]],[80*mm,86*mm]),
      Spacer(1,6)]
    return [KeepTogether(out)]

E=[Paragraph("Splitting the work across four engineers",TITLE),
   Paragraph("TikTok TechJam 2026 &middot; Track 4 &middot; Shopping Copilot",SUB),
   box("Where the project stands",
     "The agent scores <b>0.951</b> against the organizer's own evaluator, versus 0.107 for the shipped starter. "
     "It is 290 lines of pure standard library with an empty requirements file.<br/><br/>"
     "The achievable ceiling is <b>0.976</b>, not 1.000 &mdash; 15% of sessions cannot convert before turn 3, and six "
     "targets can never be ranked first. <b>So the real headroom is 0.0247, and reaching even that needs oracle "
     "knowledge no agent can have.</b><br/><br/>"
     "<b>The coding left is under one engineer-day. The evidence and presentation work is larger and carries more of "
     "the grade</b> &mdash; Technical Execution is 35%, but Innovation, Impact, Feasibility and Presentation together "
     "are 65%. Weight the team accordingly."),
   Spacer(1,8),
   Paragraph("The four roles at a glance",H1),
   tbl([[hd("#"),hd("Role"),hd("Mission in one line"),hd("Load")],
     [cell("<b>A</b>"),cell("Adversarial tester"),cell("Break the agent before the organizer does"),cell("full")],
     [cell("<b>B</b>"),cell("Pipeline improver"),cell("Close the last 0.02, and only that"),cell("paired with C")],
     [cell("<b>C</b>"),cell("Measurement owner"),cell("Make sure every number quoted is real"),cell("paired with B")],
     [cell("<b>D</b>"),cell("Instrumentation and submission"),cell("Produce the evidence, then the submission"),cell("full, plus help")],
    ],[10*mm,48*mm,72*mm,36*mm]),
   Spacer(1,6),
   box("On B and C being paired",
     "Neither is 72 hours of work on its own. The agent is 290 lines and the measured headroom is under 0.02, so B "
     "would run dry; and C's gate is minutes per merge, so C would idle. <b>Give both jobs to two engineers and let "
     "them swap</b> &mdash; whoever is not editing the agent owns the harness that checks it. That also means the "
     "person measuring a change is never the person who wrote it."),
   Spacer(1,8),
   box("The one rule that prevents most of the pain",
     "<b>Only Person B writes </b><font face='Courier' size='9'>submission/agent.py</font><b>.</b> "
     "Everyone else proposes changes as a failing test or a measured result. Four engineers editing a 290-line file "
     "over 72 hours is where hackathon repos go wrong, and the agent is small enough that one owner is plenty."),
   Spacer(1,4)]

E+=role("Person A","Adversarial tester",
 "Break the agent before the organizer does.",
 "All 200 public sessions come from one generator, and the private 800 come from the same one. So generating more "
 "<i>sessions</i> mostly tests the same thing repeatedly. What actually differs between public and private is the "
 "simulator's <b>wording</b>. Rewriting the two sentence templates once dropped the agent from 0.951 to 0.127 before "
 "hardening &mdash; the only intervention that ever produced a catastrophic result.",
 ["Build an adversarial <b>simulator</b>, not just more sessions: rewrite <font face='Courier' size='8'>customer_reply</font> "
  "and <font face='Courier' size='8'>initial_message</font> in a dozen voices &mdash; terse, chatty, synonym-swapped, "
  "reordered, typo'd, LLM-paraphrased.",
  "<b>Validate the generator first</b> by reproducing the shipped 200 exactly. If it cannot, the variants are not "
  "measuring what you think.",
  "Report a score table per variant, and flag anything that drops below the 0.832 floor.",
  "Generate volume where it helps: many targets per category bucket, targets with no unique clue, targets with empty "
  "feature lists (10.4% of the catalogue).",
  "Hand failures to Person B as a reproducible failing case, never as a suggestion.",
  "Own the regression corpus so fixes stay fixed."],
 "<font face='Courier' size='8'>tools/adversarial_*.py</font><br/><font face='Courier' size='8'>tests/</font>",
 "Every simulator variant has a measured score, and no variant is an unexplained surprise.")

E+=role("Person B","Pipeline improver  (paired with C)",
 "Close the last 0.02, and only that.",
 "Headroom is tightly bounded and already located. The oracle ceiling is <b>0.976</b> and the agent is at 0.951, so "
 "<b>0.0247 is the absolute maximum</b> &mdash; and reaching it would require knowing, per session, the optimal turn to "
 "answer, which no agent can know. 182 of 200 sessions already return the right product at rank 1. The 18 that do not "
 "are targets whose clues are all shared with other products (median minimum document frequency 21, versus 1 for the "
 "sessions that succeed).",
 ["Start from <font face='Courier' size='8'>tools/diagnose18.py</font>, not from general ideas. The 18 named sessions "
  "are worth 0.017; a single miss is worth 0.003; turn efficiency is worth 0.008.",
  "Accept that most of the 0.0247 is unreachable. Do not burn the weekend chasing it.",
  "Fix failures handed over by Person A, as reproducible cases.",
  "Every change must be measured on the harness before it is proposed.",
  "<b>Sole writer of </b><font face='Courier' size='8'>submission/agent.py</font>. Everyone else proposes changes as "
  "a failing test or a measured result.",
  "Keep it dependency-free. A stdlib agent cannot fail an install, hit a version conflict, or break when the organizer "
  "disables the network.",
  "Do not re-test retrieval models &mdash; see the do-not-do list."],
 "<font face='Courier' size='8'>submission/agent.py</font>",
 "Every merged change is backed by a measured, reproducible gain on the harness.")

E+=role("Person C","Measurement owner  (paired with B)",
 "Make sure every number the team quotes is real.",
 "This is not about git conflicts &mdash; those are cheap to resolve. It is about <b>semantic regressions git cannot "
 "see</b> (two branches merge cleanly and the agent is worse) and about <b>bad measurement</b>. During the design work "
 "on this agent, four separate conclusions turned out to be false: a harness whose corruption depended on RNG "
 "consumption order, a test where forgotten arguments made four conditions run identically, two reported effects "
 "(+0.029 and -0.04) that were pure noise, and an agent that was <b>nondeterministic by plus or minus 0.03 between "
 "identical runs</b>. Every one would have shipped as a finding without someone whose job is to distrust the result.",
 ["Own the merge gate. <b>Every branch must pass before merging.</b>",
  "<b>Run the harness twice and require identical numbers.</b> The agent was nondeterministic at one point "
  "(plus or minus 0.03 between identical runs) and it silently invalidated a full day of comparisons.",
  "Require no regression on any of the six adversarial conditions.",
  "Require contract compliance: message is always a string, at most 10 unique in-catalogue IDs, valid "
  "<font face='Courier' size='8'>ask_attribute</font>, no exception escapes.",
  "Own <font face='Courier' size='8'>config.py</font> and run the constant sweeps. Nobody else edits constants.",
  "Keep <font face='Courier' size='8'>main</font> always submittable: at any hour, the current main must be a valid entry.",
  "<b>Never measure your own change.</b> Swap with B so the person checking a result did not write it.",
  "When not gating, take the improver seat and let B take the harness."],
 "<font face='Courier' size='8'>submission/config.py</font><br/><font face='Courier' size='8'>tools/verify_agent.py</font><br/>the merge gate",
 "No number reaches the report unless the harness produced it twice, identically.")

E+=role("Person D","Instrumentation and submission owner",
 "Produce the evidence, then produce the submission.",
 "This role covers the 65% of the grade that is not TechnicalScore, and it is the part teams leave to the final hour. "
 "The front half is technical, the back half is the deliverables.",
 ["<b>First: build the trace log and self-monitor</b> (designed in <font face='Courier' size='8'>ARCHITECTURE.md</font>, "
  "not yet written). About 20 lines.",
  "One JSONL line per turn: session, turn, scenario, route used, clues, candidate count, decision, top result.",
  "Track early-exit rate, average turns and route histogram. The agent never sees whether it was right, but a session "
  "that ends before turn 10 was a hit &mdash; that is a live self-evaluation signal.",
  "This is the only runtime evidence you will have about the private run, whatever the score turns out to be.",
  "<b>Then: the deliverables.</b> README with one-command setup; the short report on architecture, model choice, "
  "cost and limitations; team member contributions.",
  "Map every module to the four judged pillars <b>by name</b> so judges can see the mapping without hunting.",
  "Disclose latency, token usage, estimated cost and network requirements &mdash; the model policy requires it.",
  "Record and edit the <b>3-minute YouTube demo</b>, public, linked from Devpost.",
  "Own the Devpost submission and the deadline. Note the post-deadline bio request: <b>failure to supply within 48 "
  "hours can disqualify the team</b>."],
 "<font face='Courier' size='8'>submission/trace.py</font><br/><font face='Courier' size='8'>README.md</font>, report, video, Devpost",
 "Submission is complete and linked 6 hours before the deadline, not 6 minutes.")

E+=[Paragraph("File ownership &mdash; keeps merges clean",H1),
 tbl([[hd("Owner"),hd("Files"),hd("Rule")],
   [cell("B &mdash; improver"),cell("<font face='Courier' size='8'>submission/agent.py</font>"),cell("Sole writer <i>while holding the seat</i>. Others propose changes as failing tests or measured results.")],
   [cell("A &mdash; tester"),cell("<font face='Courier' size='8'>tools/adversarial_*.py, tests/</font>"),cell("Adds cases; never edits the agent.")],
   [cell("C &mdash; integration"),cell("<font face='Courier' size='8'>config.py, verify_agent.py</font>"),cell("Sole editor of constants. Runs the gate.")],
   [cell("D &mdash; submission"),cell("<font face='Courier' size='8'>trace.py</font>, README, report"),cell("Read-only on the agent.")],
  ],[32*mm,60*mm,74*mm]),
 Spacer(1,6),
 Paragraph("Branch protocol",H1),
 tbl([[hd("Step"),hd("Action")],
   [cell("1"),cell("Branch from the latest <font face='Courier' size='8'>main</font>, never from another feature branch.")],
   [cell("2"),cell("Make the change. Run the harness locally.")],
   [cell("3"),cell("Open a PR quoting the six condition scores, before and after.")],
   [cell("4"),cell("Person C runs the gate: harness twice, identical numbers, no regression, contract compliant.")],
   [cell("5"),cell("Merge. Everyone rebases onto the new main immediately.")],
  ],[16*mm,150*mm]),
 Spacer(1,6),
 KeepTogether([Paragraph("Two things nobody should spend time on",H1),
 box("",
   "<b>1. Re-testing retrieval models.</b> Nine approaches were tested across two document representations &mdash; "
   "bge-small, bge-base, gte-base embeddings and two cross-encoder rerankers. All lost to a plain review count. On the "
   "hard cases the candidate products have byte-identical descriptions, so no model can separate them. A perfect ranker "
   "there would be worth 0.006.<br/><br/>"
   "<b>2. Tuning constants by hand.</b> The response is flat &mdash; tuning on half the sessions and testing on the "
   "other half moved the score by 0.001. Hand-tuning mostly produces noise that looks like progress. Sweeps belong to "
   "Person C, on the harness.", colors.HexColor("#FBEAF0"))]),
 Spacer(1,6),
 KeepTogether([Paragraph("How to spend spare capacity",H1),
 Paragraph("B and C will finish before A and D. When they do, move them to <b>D's half of the job</b> &mdash; the "
   "README, the report, the pillar mapping, the demo video &mdash; or to <b>A's</b>, writing more simulator variants. "
   "Do not move them back onto the agent. The measured ceiling is 0.976 and the agent is at 0.951; the last two "
   "hundredths are the most expensive and least certain points on the board, while the submission deliverables are "
   "worth 65% of the grade and are entirely within your control.",BODY)])]

doc=SimpleDocTemplate(OUT,pagesize=A4,leftMargin=22*mm,rightMargin=22*mm,
                      topMargin=18*mm,bottomMargin=20*mm,
                      title="Track 4 - team roles",author="Jovan Nai")
doc.build(E,onFirstPage=foot,onLaterPages=foot)
print("WROTE",OUT)
