# The page's notes CSV helpers, run under Node. Skipped where Node is not installed.

import json
import re
import shutil
import subprocess

import pytest

from maybemap.page import WEB_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

NAMES = ("FORMULA_LEAD", "guardFormula", "unguardFormula", "escapeCSV", "csvParse")


def _helpers() -> str:
    # Each helper is a top-level declaration: a one-liner, or a block that closes
    # with "}" at column 0
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    parts = []
    for name in NAMES:
        match = re.search(
            rf"^(?:const {name} = .*?;$|function {name}\(.*?(?:\}}$(?=\n(?!\s))))",
            source,
            re.M | re.S,
        )
        assert match, f"{name} not found in app.js"
        parts.append(match.group(0))
    return "\n".join(parts)


def _run(expr: str):
    script = _helpers() + f"\nprocess.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


NOTES = [
    "=SUM(A1)",
    "-2 trucks",
    "+1 contact",
    "@ the gate",
    "'=already quoted",
    "''-two",
    "it's fine",
    "plain, with comma",
    'say "hi"\nline two',
]


def test_formula_cells_are_neutralized():
    cells = _run(f"{json.dumps(NOTES[:4])}.map(escapeCSV)")
    assert all(c.startswith("'") for c in cells)


def test_every_note_survives_export_then_import():
    back = _run(
        f"{json.dumps(NOTES)}.map(n => unguardFormula(csvParse('h\\r\\n' + escapeCSV(n))[1][0]))"
    )
    assert back == NOTES
