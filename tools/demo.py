"""Read-only demo viewer: runs the real agent on a real session, paced for screen capture.

    python3 tools/demo.py buying
    python3 tools/demo.py browsing | override | boundary | all
    python3 tools/demo.py buying --speed 0.5     # slower (default 1.0; 0 = instant)

Uses the organizer's unmodified evaluator and the shipped agent. Prints only —
it cannot influence the scored path.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

C = dict(t="\033[38;5;30m", b="\033[1m", d="\033[2m", g="\033[38;5;28m",
         a="\033[38;5;130m", r="\033[0m", w="\033[38;5;250m")

# one representative session per scenario
PICK = {"buying": "public_0009", "browsing": "public_0011",
        "override": "public_0002", "boundary": "public_0041"}
SPEED = 1.0


def pause(x=0.35):
    if SPEED: time.sleep(x * SPEED)


def type_out(text, indent, colour="", delay=0.006):
    sys.stdout.write(indent + colour)
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush()
        if SPEED and delay: time.sleep(delay * SPEED)
    sys.stdout.write(C["r"] + "\n")


def wrap(text, width=64):
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line: out.append(line)
    return out


def run(scenario, samples, ids, cats, prods, Agent, LE, trace_path):
    sid = PICK[scenario]
    smp = [s for s in samples if s["sample_id"] == sid]
    if not smp:
        print(f"session {sid} not found"); return
    target = str(smp[0]["ground_truth"]["parent_asin"])

    Path(trace_path).write_text("")
    agent = Agent("data/catalog.jsonl")
    res = LE.evaluate(agent, smp, ids, cats, prods)
    turns = [json.loads(l) for l in open(trace_path)]
    sess = res["sessions"][0]

    print(f"\n{C['t']}{C['b']}{'═'*72}")
    print(f" {scenario.upper()}   ·   session {sid}   ·   catalog: 50,000 products")
    print(f"{'═'*72}{C['r']}")
    pause(0.8)

    for t in turns:
        print(f"\n{C['b']} TURN {t['turn']}{C['r']}")
        print(f"{C['d']} {'─'*70}{C['r']}")
        pause(0.3)
        for i, line in enumerate(wrap(t["msg"])):
            type_out(line, "  SHOPPER   " if i == 0 else "            ", C["w"])
        pause(0.5)

        cat = t["cat"] or "—"
        sure = "locked" if t["cat_sure"] else "guessed"
        clues = ", ".join(t["new_clues"]) or "—"
        print(f"{C['d']}   · parse      {C['r']}{scenario} · category {sure}: {C['t']}{cat}{C['r']}"); pause(0.25)
        print(f"{C['d']}   · remember   {C['r']}new: {C['t']}{clues}{C['r']}"); pause(0.25)
        print(f"{C['d']}   · narrow     {C['r']}candidates left: {C['t']}{C['b']}{t['cand']}{C['r']}"
              f"   {C['d']}[route: {t['route']}]{C['r']}"); pause(0.25)
        gate = f"   {C['a']}(mind-change not sent yet — cannot score){C['r']}" if t["gated"] else ""
        print(f"{C['d']}   · schedule   {C['r']}showing {C['t']}{len(t['emitted'])}{C['r']} card(s){gate}"); pause(0.4)

        for i, line in enumerate(wrap(t["said"])):
            type_out(line, "  AGENT     " if i == 0 else "            ", C["g"])
        print(f"{C['d']}            asks: {t['ask']}{C['r']}")

        if target in t["emitted"] and not t["gated"]:
            rank = t["emitted"].index(target) + 1
            pause(0.5)
            print(f"\n  {C['g']}{C['b']}  ✓  TARGET FOUND — turn {t['turn']}, rank {rank}{C['r']}")
        elif target in t["emitted"]:
            print(f"{C['a']}            target is on screen, but this turn cannot score:{C['r']}")
            print(f"{C['a']}            the shopper has not changed their mind yet{C['r']}")
        else:
            print(f"{C['d']}            target not in this list{C['r']}")
        pause(0.6)

    print(f"\n{C['t']}{'─'*72}{C['r']}")
    print(f"  hit: {C['b']}{sess['hit']}{C['r']}   turn: {C['b']}{sess['first_hit_turn']}{C['r']}"
          f"   reciprocal rank: {C['b']}{sess['reciprocal_rank']:.2f}{C['r']}"
          f"   tokens used: {C['b']}0{C['r']}")
    print(f"{C['t']}{'─'*72}{C['r']}\n")
    pause(1.2)


def main():
    global SPEED
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--speed" in sys.argv:
        SPEED = float(sys.argv[sys.argv.index("--speed") + 1])
    which = args[0] if args else "all"

    import submission.config as CFG
    trace_path = "/tmp/techjam_demo_trace.jsonl"
    CFG.TRACE_PATH = trace_path
    import evaluator.local_evaluator as LE
    from submission.agent import Agent

    print(f"\n{C['d']}  loading the organizer's catalog and building indexes…{C['r']}")
    samples = LE.load_jsonl("data/public_set.jsonl")
    ids, cats, prods = LE.catalog_index("data/catalog.jsonl")
    print(f"{C['d']}  ready.{C['r']}")

    todo = list(PICK) if which == "all" else [which]
    for s in todo:
        if s not in PICK:
            print(f"unknown scenario '{s}'. choose: {', '.join(PICK)} | all"); return
        run(s, samples, ids, cats, prods, Agent, LE, trace_path)


if __name__ == "__main__":
    main()
