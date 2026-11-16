# The page's pure helpers (notes CSV, drive-to search), run under Node. Skipped where Node is not installed.

import json
import re
import shutil
import subprocess

import pytest

from maybemap.page import WEB_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

NAMES = ("FORMULA_LEAD", "guardFormula", "unguardFormula", "escapeCSV", "csvParse", "matchPlaces")


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


PLACES = [
    {"name": "Zz Alpha", "state": "ZZ"},
    {"name": "Alphaville", "state": "ZZ"},
    {"name": "Beta Alpha", "state": "YY"},
    {"name": "Gamma <b>x</b>", "state": "ZZ"},
    {"name": "Delta", "state": "YY"},
]


def _match(q, exclude=-1, *extra):
    args = ", ".join(json.dumps(a) for a in (PLACES, q, exclude, *extra))
    return _run(f"matchPlaces({args}).map(j => {json.dumps([p['name'] for p in PLACES])}[j])")


def test_match_places_is_case_insensitive_and_ranks_prefixes_first():
    # "alpha" is a prefix of Alphaville and a substring of the other two
    assert _match("  ALPHA ") == ["Alphaville", "Beta Alpha", "Zz Alpha"]


def test_match_places_matches_the_state_too():
    assert _match("delta, yy") == ["Delta"]


def test_match_places_excludes_the_start_town():
    assert _match("alpha", 1) == ["Beta Alpha", "Zz Alpha"]


def test_match_places_respects_the_limit():
    assert _match("a", -1, 2) == ["Alphaville", "Beta Alpha"]
    assert len(_match("a")) == 5


def test_match_places_gives_nothing_for_an_empty_query():
    assert _match("") == []
    assert _match("   ") == []


def test_match_places_keeps_markup_in_a_name_as_data():
    assert _match("<b>") == ["Gamma <b>x</b>"]
