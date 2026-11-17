# Branding comes from the environment: neutral defaults, strict colors, and a set
# path that is missing or of the wrong type stops the build.

import base64
import re

import pytest

from maybemap import branding, colors

ALL_COLORS = [var for var, _ in branding.COLORS.values()]


def test_defaults_when_nothing_is_set():
    b = branding.load_branding()
    assert b == branding.Branding()
    assert b.title == branding.DEFAULT_TITLE
    assert b.secondary == b.primary
    assert (b.logo, b.heading_font, b.body_font) == ("", "", "")


def test_empty_values_count_as_unset(monkeypatch):
    for var in ["MAP_TITLE", "BRAND_LOGO", "BRAND_FONT_HEADING", *ALL_COLORS]:
        monkeypatch.setenv(var, "  ")
    assert branding.load_branding() == branding.Branding()


def test_every_variable_is_read(monkeypatch, tmp_path):
    logo = tmp_path / "logo.svg"
    logo.write_bytes(b"<svg/>")
    font = tmp_path / "Head.woff2"
    font.write_bytes(b"wOF2head")
    body = tmp_path / "Body.WOFF2"
    body.write_bytes(b"wOF2body")
    monkeypatch.setenv("MAP_TITLE", "Acme Coverage")
    monkeypatch.setenv("BRAND_COLOR_DARK", "#20245C")
    monkeypatch.setenv("BRAND_COLOR_PRIMARY", "#352b79")
    monkeypatch.setenv("BRAND_COLOR_SECONDARY", "#883F98")
    monkeypatch.setenv("BRAND_COLOR_ACCENT", "#FDB912")
    monkeypatch.setenv("BRAND_LOGO", str(logo))
    monkeypatch.setenv("BRAND_FONT_HEADING", str(font))
    monkeypatch.setenv("BRAND_FONT_BODY", str(body))
    b = branding.load_branding()
    assert b.title == "Acme Coverage"
    assert (b.dark, b.primary, b.secondary, b.accent) == (
        "#20245C",
        "#352b79",
        "#883F98",
        "#FDB912",
    )
    assert b.logo == "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode()
    assert b.heading_font == "data:font/woff2;base64," + base64.b64encode(b"wOF2head").decode()
    assert b.body_font.startswith("data:font/woff2;base64,")


def test_secondary_follows_a_set_primary(monkeypatch):
    monkeypatch.setenv("BRAND_COLOR_PRIMARY", "#123456")
    assert branding.load_branding().secondary == "#123456"


def test_a_relative_path_is_read_from_the_project_root(monkeypatch, tmp_path):
    (tmp_path / "brand").mkdir()
    (tmp_path / "brand" / "logo.png").write_bytes(b"png")
    monkeypatch.setattr(branding, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("BRAND_LOGO", "brand/logo.png")
    assert branding.load_branding().logo.startswith("data:image/png;base64,")


@pytest.mark.parametrize(
    "value", ["red", "#12345", "#1234567", "#12345g", "#fff;}</style><script>", "1f2a44"]
)
@pytest.mark.parametrize("var", ALL_COLORS)
def test_a_color_that_is_not_plain_hex_stops_the_build(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=var):
        branding.load_branding()


@pytest.mark.parametrize("var", ["BRAND_LOGO", "BRAND_FONT_HEADING", "BRAND_FONT_BODY"])
def test_a_missing_file_stops_the_build(monkeypatch, tmp_path, var):
    suffix = ".png" if var == "BRAND_LOGO" else ".woff2"
    monkeypatch.setenv(var, str(tmp_path / ("nope" + suffix)))
    with pytest.raises(ValueError, match=var):
        branding.load_branding()


@pytest.mark.parametrize(
    "var, name",
    [("BRAND_LOGO", "logo.gif"), ("BRAND_FONT_HEADING", "f.ttf"), ("BRAND_FONT_BODY", "f.png")],
)
def test_a_file_of_the_wrong_type_stops_the_build(monkeypatch, tmp_path, var, name):
    (tmp_path / name).write_bytes(b"x")
    monkeypatch.setenv(var, str(tmp_path / name))
    with pytest.raises(ValueError, match=var):
        branding.load_branding()


def test_css_sets_the_variables_and_only_the_fonts_given():
    css = branding.css(branding.Branding(heading_font="data:font/woff2;base64,AA=="))
    assert (
        ":root{--brand-dark:#1f2a44;--brand:#274e74;--brand-2:#274e74;"
        "--brand-2-deep:#274e74;--accent:#e3ae3c;"
        "--heading-dark:#becef0;--brand-ui-dark:#91bce8}"
    ) in css
    assert "font-family:MapHeading;src:url(data:font/woff2;base64,AA==)" in css
    assert "MapBody" not in css
    assert "@font-face" not in branding.css(branding.Branding())


@pytest.mark.parametrize("secondary", ["#883F98", "#2f5d8a", "#ffcc00", "#ffffff", "#000000"])
def test_the_gradient_end_under_the_presets_keeps_white_text_at_aaa(secondary):
    # The preset buttons' white text sits on the secondary end of the header gradient
    css = branding.css(branding.Branding(secondary=secondary))
    deep = re.search(r"--brand-2-deep:(#[0-9a-f]{6})", css).group(1)
    assert colors.contrast("#ffffff", deep) >= 7
