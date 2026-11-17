# Company branding, read from the environment (.env) at build time. The repo ships
# neutral defaults; a company's title, colors, logo and fonts are set in .env and its
# asset files live in the gitignored data/ folder, so nothing company-specific is tracked.

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from maybemap.colors import WHITE, readable
from maybemap.config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_TITLE = "Service Coverage Map"
# (variable, default). A secondary color left unset follows the primary
COLORS = {
    "dark": ("BRAND_COLOR_DARK", "#1f2a44"),
    "primary": ("BRAND_COLOR_PRIMARY", "#2f5d8a"),
    "secondary": ("BRAND_COLOR_SECONDARY", None),
    "accent": ("BRAND_COLOR_ACCENT", "#e0a526"),
}
LOGO_TYPES = {".png": "image/png", ".svg": "image/svg+xml", ".webp": "image/webp"}
FONT_TYPES = {".woff2": "font/woff2"}
HEX = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class Branding:
    title: str = DEFAULT_TITLE
    dark: str = COLORS["dark"][1]
    primary: str = COLORS["primary"][1]
    secondary: str = COLORS["primary"][1]
    accent: str = COLORS["accent"][1]
    # data: URIs, or "" when unset
    logo: str = ""
    heading_font: str = ""
    body_font: str = ""


def _env(name: str) -> str:
    # An empty variable means unset
    return os.getenv(name, "").strip()


def _color(name: str, default: str | None) -> str | None:
    value = _env(name)
    if not value:
        return default
    # The value is written into the page's CSS, so only a plain hex color gets through
    if not HEX.fullmatch(value):
        raise ValueError(f"{name} must be a hex color like #1f2a44, got {value!r}")
    return value


def _data_uri(name: str, types: dict[str, str]) -> str:
    value = _env(name)
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    kind = types.get(path.suffix.lower())
    if kind is None:
        raise ValueError(f"{name} must be one of {', '.join(sorted(types))}: {path}")
    if not path.is_file():
        raise ValueError(f"{name} points at a file that does not exist: {path}")
    return f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def load_branding() -> Branding:
    colors = {key: _color(var, default) for key, (var, default) in COLORS.items()}
    return Branding(
        title=_env("MAP_TITLE") or DEFAULT_TITLE,
        dark=colors["dark"],
        primary=colors["primary"],
        secondary=colors["secondary"] or colors["primary"],
        accent=colors["accent"],
        logo=_data_uri("BRAND_LOGO", LOGO_TYPES),
        heading_font=_data_uri("BRAND_FONT_HEADING", FONT_TYPES),
        body_font=_data_uri("BRAND_FONT_BODY", FONT_TYPES),
    )


def css(b: Branding) -> str:
    # The brand variables and font faces, prepended to the page's stylesheet
    # The preset buttons' white text sits on the gradient's secondary end, so that end is
    # darkened as far as AAA contrast needs (a dark enough secondary passes unchanged)
    deep, _ = readable(b.secondary, ink=WHITE)
    rules = [
        f":root{{--brand-dark:{b.dark};--brand:{b.primary};"
        f"--brand-2:{b.secondary};--brand-2-deep:{deep};--accent:{b.accent}}}"
    ]
    for family, uri in (("MapHeading", b.heading_font), ("MapBody", b.body_font)):
        if uri:
            rules.append(
                f"@font-face{{font-family:{family};src:url({uri}) format('woff2');"
                "font-weight:100 900;font-display:swap}"
            )
    return "\n".join(rules)
