# The contrast helpers that keep label text at AAA (7:1)

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
