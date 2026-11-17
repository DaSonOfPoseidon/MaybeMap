# The contrast helpers that keep label text at AAA (7:1)

import re

import pytest

from maybemap import colors


def test_contrast_matches_the_wcag_extremes():
    assert colors.contrast("#000000", "#ffffff") == pytest.approx(21)
    assert colors.contrast("#777777", "#777777") == pytest.approx(1)


@pytest.mark.parametrize("bg", ["#bf8000", "#0054bf", "#63007e", "#ffffff", "#000000", "#777777"])
def test_readable_reaches_the_target_and_moves_the_color_only_as_needed(bg):
    out, ink = colors.readable(bg)
    assert ink in (colors.BLACK, colors.WHITE)
    assert colors.contrast(ink, out) >= 7
    if colors.contrast(ink, bg) >= 7:
        assert out == bg


def test_readable_keeps_a_requested_ink():
    out, ink = colors.readable("#ffcc00", ink=colors.WHITE)
    assert ink == colors.WHITE and colors.contrast(ink, out) >= 7


def test_lift_lightness_matches_the_values_the_dark_theme_resolved_in_a_browser():
    assert colors.lift_lightness("#20245C", 0.85) == "#bbc9ff"
    assert colors.lift_lightness("#352B79", 0.78) == "#adacff"


@pytest.mark.parametrize("bg", ["#ffffff", "#ffcc00", "#BBC9FF"])
def test_lift_lightness_leaves_a_color_already_above_the_floor_alone(bg):
    assert colors.lift_lightness(bg, 0.78) == bg.lower()


def test_lift_lightness_turns_black_into_a_neutral_gray_at_the_floor():
    out = colors.lift_lightness("#000000", 0.85)
    r, g, b = colors._rgb(out)
    assert r == g == b and out != "#000000"
    assert colors._oklch(out)[0] == pytest.approx(0.85, abs=0.01)


@pytest.mark.parametrize("bg", ["#0000ff", "#ff0000", "#352B79", "#1f2a44"])
def test_lift_lightness_always_gives_a_plain_hex_color(bg):
    # An out-of-gamut lift is clamped the way a browser does it, never an error
    out = colors.lift_lightness(bg, 0.85)
    assert re.fullmatch(r"#[0-9a-f]{6}", out)
    assert colors.luminance(out) > colors.luminance(bg)


@pytest.mark.parametrize("bg", ["#1f2a44", "#2f5d8a", "#352B79", "#20245C"])
def test_lift_lightness_reaches_the_floor_when_the_lift_stays_in_gamut(bg):
    assert colors._oklch(colors.lift_lightness(bg, 0.78))[0] >= 0.78 - 0.01
