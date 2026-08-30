"""Environment/contract assertions for the shipped agent (F8, non-session).

    PYTHONHASHSEED=0 python3 -m tools.suite.env_check

Read-only with respect to production code.  Each construction test that depends
on the working directory runs in a SUBPROCESS with an explicit cwd, because
`os.chdir` in-process would leak into anything else running in the same run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CATALOG_REL = "data/catalog.jsonl"
CATALOG_ABS = str(REPO / CATALOG_REL)


def _construct_in(cwd: str, expr: str, timeout: int = 120) -> tuple:
    """Build an Agent in a subprocess with a given cwd; return (ok, note)."""
    code = (
        f"import sys; sys.path.insert(0, {str(REPO)!r});"
        f"from submission.agent import Agent;"
        f"a = {expr};"
        f"print('OK', len(a.pop))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    tail = (proc.stderr.strip().splitlines() or ["?"])[-1]
    return False, tail[:88]


def run() -> list:
    checks = []

    def add(name, fn):
        try:
            ok, note = fn()
        except Exception as exc:  # noqa: BLE001 - a raising check is a failing check
            ok, note = False, f"{type(exc).__name__}: {exc}"[:88]
        checks.append((name, ok, note))

    home = os.path.expanduser("~")
    sibling = tempfile.mkdtemp(prefix="sibling_")

    add("Agent('data/catalog.jsonl') from repo root",
        lambda: _construct_in(str(REPO), "Agent('data/catalog.jsonl')"))
    add("Agent(absolute_path)",
        lambda: _construct_in(str(REPO), f"Agent({CATALOG_ABS!r})"))
    add("Agent() from repo root",
        lambda: _construct_in(str(REPO), "Agent()"))
    add("Agent() from /tmp",
        lambda: _construct_in(tempfile.gettempdir(), "Agent()"))
    add("Agent() from $HOME",
        lambda: _construct_in(home, "Agent()"))
    add("Agent() from a sibling directory",
        lambda: _construct_in(sibling, "Agent()"))
    add("Agent(absolute_path) from /tmp",
        lambda: _construct_in(tempfile.gettempdir(), f"Agent({CATALOG_ABS!r})"))
    add("explicit relative path still honoured verbatim from repo root",
        lambda: _construct_in(str(REPO), "Agent('data/catalog.jsonl')"))

    def explicit_bad_path_still_raises():
        ok, note = _construct_in(str(REPO), "Agent('data/definitely_missing.jsonl')")
        return (not ok) and "FileNotFoundError" in note, \
               "explicit bad path raises (inference must not mask it)"
    add("explicit bad path is NOT silently repaired", explicit_bad_path_still_raises)

    add("import submission.agent",
        lambda: _construct_in(str(REPO),
                              "__import__('submission.agent', fromlist=['Agent']).Agent(%r)" % CATALOG_ABS))
    add("import starter.agent (evaluator entry point)",
        lambda: _construct_in(str(REPO),
                              "__import__('starter.agent', fromlist=['Agent']).Agent(%r)" % CATALOG_ABS))

    def two_instances():
        code = (f"import sys; sys.path.insert(0,{str(REPO)!r});"
                "from submission.agent import Agent;"
                f"a=Agent({CATALOG_ABS!r}); b=Agent({CATALOG_ABS!r});"
                "print('OK', len(a.pop), len(b.pop))")
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                              capture_output=True, text=True, timeout=240)
        return proc.returncode == 0, proc.stdout.strip() or proc.stderr.strip()[-88:]
    add("two Agent instances in one process", two_instances)

    def standalone_submission():
        """Copy submission/ + data/ to a clean tree and construct from inside it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            (root / "submission").mkdir(parents=True)
            for name in ("agent.py", "config.py", "tracelog.py", "__init__.py"):
                src = REPO / "submission" / name
                if src.exists():
                    (root / "submission" / name).write_bytes(src.read_bytes())
            (root / "data").mkdir()
            os.symlink(REPO / CATALOG_REL, root / "data" / "catalog.jsonl")
            code = (f"import sys; sys.path.insert(0,{str(root)!r});"
                    "from submission.agent import Agent; a=Agent(); print('OK', len(a.pop))")
            proc = subprocess.run([sys.executable, "-c", code], cwd=tempfile.gettempdir(),
                                  capture_output=True, text=True, timeout=240)
            return proc.returncode == 0, (proc.stdout.strip()
                                          or (proc.stderr.strip().splitlines() or ['?'])[-1][:88])
    add("standalone submission/ tree, Agent() from elsewhere", standalone_submission)

    return checks


def main() -> int:
    start = time.time()
    checks = run()
    width = max(len(n) for n, _, _ in checks) + 2
    print(f"{'ENVIRONMENT / CONTRACT ASSERTIONS':{width}s}")
    print("-" * (width + 46))
    for name, ok, note in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:{width}s} {note}")
    failed = [n for n, ok, _ in checks if not ok]
    print("-" * (width + 46))
    print(f"  {len(checks)-len(failed)}/{len(checks)} passed in {time.time()-start:.0f}s")
    if failed:
        print("  FAILING: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
