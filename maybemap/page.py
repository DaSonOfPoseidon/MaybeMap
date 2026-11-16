# Writes the map as one self-contained HTML file: the template with its stylesheet,
# script and data inlined. Only Leaflet and the basemap tiles load from the network.

import json
import re
from html import escape
from pathlib import Path

from maybemap import branding as brand
from maybemap.config import OUTPUT_PATH
from maybemap.files import write_text_atomic

WEB_DIR = Path(__file__).resolve().parent / "web"


def _inline_json(payload: dict) -> str:
    # Every "<" is written as the JSON escape \u003c, so nothing in the data can
    # close the script block early or open a comment that hides the real closing
    # tag ("<!--<script"). NaN is refused outright: Python writes it, but
    # JSON.parse rejects the whole page.
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return text.replace("<", "\\u003c")


def render(payload: dict, branding: brand.Branding | None = None) -> str:
    branding = branding or brand.load_branding()
    template = (WEB_DIR / "template.html").read_text(encoding="utf-8")
    # The logo stays hidden when no BRAND_LOGO is set
    logo = '<img class="brand-logo" src="__LOGO__" alt="" hidden>'
    if branding.logo:
        template = template.replace(logo, logo.replace(" hidden", ""))
    replacements = {
        "__TITLE__": escape(payload["title"]),
        "/*__BRAND__*/": brand.css(branding),
        "__LOGO__": branding.logo,
        "/*__CSS__*/": (WEB_DIR / "app.css").read_text(encoding="utf-8"),
        "/*__JS__*/": (WEB_DIR / "app.js").read_text(encoding="utf-8"),
        "__PAYLOAD__": _inline_json(payload),
    }
    for marker in replacements:
        if marker not in template:
            raise ValueError(f"template.html is missing its {marker} marker")
    # One pass, so a marker inside an inserted value (a title naming __LOGO__) stays literal
    pattern = re.compile("|".join(re.escape(m) for m in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)], template)


def write(payload: dict, path: Path | str = OUTPUT_PATH) -> Path:
    path = write_text_atomic(path, render(payload))
    print(f"Wrote {path.name} ({path.stat().st_size / 1024:.0f} kB)")
    return path
