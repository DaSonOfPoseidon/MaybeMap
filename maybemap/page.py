# Writes the map as one self-contained HTML file: the template with its stylesheet,
# script and data inlined. Only Leaflet and the basemap tiles load from the network.

import json
from html import escape
from pathlib import Path

from maybemap.config import OUTPUT_PATH

WEB_DIR = Path(__file__).resolve().parent / "web"


def _inline_json(payload: dict) -> str:
    # Inside <script type="application/json"> the only thing that can end the block
    # early is "</", so it is written as the equivalent JSON escape "<\/". NaN is
    # refused outright: Python writes it, but JSON.parse rejects the whole page.
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return text.replace("</", "<\\/")


def render(payload: dict) -> str:
    template = (WEB_DIR / "template.html").read_text()
    replacements = {
        "__TITLE__": escape(payload["title"]),
        "/*__CSS__*/": (WEB_DIR / "app.css").read_text(),
        "/*__JS__*/": (WEB_DIR / "app.js").read_text(),
        "__PAYLOAD__": _inline_json(payload),
    }
    for marker, value in replacements.items():
        if marker not in template:
            raise ValueError(f"template.html is missing its {marker} marker")
        template = template.replace(marker, value)
    return template


def write(payload: dict, path: Path | str = OUTPUT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload))
    print(f"Wrote {path.name} ({path.stat().st_size / 1024:.0f} kB)")
    return path
