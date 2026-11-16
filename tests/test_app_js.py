# The page's pure helpers (notes CSV, drive-to search, closest techs, orange pins), run under Node.
# Skipped without Node.

import json
import re
import shutil
import subprocess

import pytest

from maybemap.page import WEB_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

NAMES = (
    "FORMULA_LEAD",
    "guardFormula",
    "unguardFormula",
    "escapeCSV",
    "csvParse",
    "matchPlaces",
    "closestTechs",
    "outsideService",
    "namedTechs",
    "techOn",
    "activeTechs",
    "pinIsOrange",
)


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


def _run(expr: str, prelude: str = ""):
    # prelude declares the page globals a helper reads (techs, state)
    script = prelude + "\n" + _helpers() + f"\nprocess.stdout.write(JSON.stringify({expr}));"
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


def _closest(rows, *extra):
    args = ", ".join(json.dumps(a) for a in (rows, *extra))
    return _run(f"closestTechs({args}).map(r => r.tech.name)")


def _row(tech, s, m=0, here=False):
    return {"tech": {"name": tech}, "s": s, "m": m, "here": here}


def test_closest_techs_sorts_by_time_then_distance_then_name():
    rows = [
        _row("d", 900, 5),
        _row("c", 600, 9),
        _row("b", 600, 7),
        _row("a2", 600, 7),
        _row("a1", 600, 7),
    ]
    assert _closest(rows) == ["a1", "a2", "b"]


def test_closest_techs_limits_to_n_and_defaults_to_three():
    rows = [_row(c, i) for i, c in reversed(list(enumerate("abcde")))]
    assert _closest(rows) == ["a", "b", "c"]
    assert _closest(rows, 1) == ["a"]
    assert _closest(rows, 10) == ["a", "b", "c", "d", "e"]


def test_closest_techs_handles_fewer_than_n_and_none():
    assert _closest([]) == []
    assert _closest([_row("only", 5)]) == ["only"]


def test_closest_techs_puts_a_tech_based_here_first_and_keeps_two_in_one_town():
    rows = [_row("far", 3000, 40), _row("z", 0, 0, True), _row("a", 0, 0, True)]
    assert _closest(rows) == ["a", "z", "far"]


def test_closest_techs_does_not_change_its_input():
    rows = [_row("b", 2), _row("a", 1)]
    names = "r.map(x => x.tech.name)"
    got = _run(f"(() => {{ const r = {json.dumps(rows)}; closestTechs(r); return {names}; }})()")
    assert got == ["b", "a"]


# A tech town with no row in the city export (city is null) is outside the service area
IN_EXPORT = {"name": "Zz Inside", "city": 7}
OUTSIDE = {"name": "Zz Outside", "city": None}


def test_outside_service_needs_no_export_row_and_an_active_tech():
    assert _run(f"outsideService({json.dumps(OUTSIDE)}, 1)") is True
    assert _run(f"outsideService({json.dumps(OUTSIDE)}, 3)") is True


def test_outside_service_is_false_with_no_active_tech():
    # The department filter can hide every tech of a town
    assert _run(f"outsideService({json.dumps(OUTSIDE)}, 0)") is False


def test_outside_service_is_false_for_a_town_in_the_export():
    assert _run(f"outsideService({json.dumps(IN_EXPORT)}, 2)") is False
    # City id 0 is a real export row, not "no row"
    assert _run('outsideService({"city": 0}, 2)') is False


def test_orange_pins_name_no_techs_and_teal_pins_keep_theirs():
    techs = [{"name": "Zz One"}, {"name": "Zz Two"}]
    assert _run(f"namedTechs({json.dumps(OUTSIDE)}, {json.dumps(techs)})") == []
    assert _run(f"namedTechs({json.dumps(IN_EXPORT)}, {json.dumps(techs)})") == techs
    assert _run(f"namedTechs({json.dumps(OUTSIDE)}, [])") == []


# The orange pin rule has one source of truth: pinIsOrange. Labels and search swatches use it
ZZ_TECHS = [
    {"name": "Zz One", "departments": ["Zz Dept"]},
    {"name": "Zz Two", "departments": ["Zz Other"]},
]
PIN_PRELUDE = (
    f"const techs = {json.dumps(ZZ_TECHS)};"
    " const state = { depts: { 'Zz Dept': true, 'Zz Other': false } };"
)


def _orange(place: dict) -> bool:
    return _run(f"pinIsOrange({json.dumps(place)})", PIN_PRELUDE)


def test_in_service_tech_town_is_not_an_orange_pin():
    assert _orange({**IN_EXPORT, "techs": [0]}) is False


def test_outside_town_with_an_active_tech_is_an_orange_pin():
    assert _orange({**OUTSIDE, "techs": [0]}) is True


def test_outside_town_with_no_active_tech_is_not_an_orange_pin():
    # No techs at all, and a tech whose only department is switched off
    assert _orange({**OUTSIDE, "techs": []}) is False
    assert _orange({**OUTSIDE, "techs": [1]}) is False
