# The single-file HTML: everything inlined, and data that cannot break out of its
# script block.

import json
import re

import pytest

from maybemap import page

PAYLOAD = {
    "title": "Test & Map",
    "note": "</script><script>alert(1)</script>",
    # An HTML comment opener followed by <script puts the parser into the
    # double-escaped state, where the real closing tag no longer ends the block
    "region": "<!--<script>",
}


def _data(html):
    match = re.search(r'<script type="application/json" id="map-data">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


def test_payload_round_trips_and_cannot_close_its_script():
    html = page.render(PAYLOAD)
    assert _data(html) == PAYLOAD
    assert "</script><script>alert" not in html
    data_block = html.split('id="map-data">', 1)[1].split("</script>", 1)[0]
    assert "<" not in data_block


def test_assets_are_inlined_and_markers_consumed():
    html = page.render(PAYLOAD)
    for marker in ("__TITLE__", "/*__CSS__*/", "/*__JS__*/", "__PAYLOAD__"):
        assert marker not in html
    assert "<title>Test &amp; Map</title>" in html
    assert "function renderZones" in html and ".layer-head" in html


def test_every_layer_section_is_present():
    html = page.render(PAYLOAD)
    for layer in ("zones", "outlines", "dots", "dispatch"):
        assert f'data-layer="{layer}"' in html
        assert f'id="layer-{layer}"' in html


def test_nan_is_refused():
    with pytest.raises(ValueError):
        page.render({"title": "x", "bad": float("nan")})


def test_write_creates_the_output_folder(tmp_path):
    out = page.write(PAYLOAD, tmp_path / "nested" / "map.html")
    assert out.exists()


def test_a_failed_write_keeps_the_previous_map(tmp_path, monkeypatch):
    # A lone surrogate cannot be encoded, so the write fails partway through --
    # after a plain write_text has already truncated the old map
    out = tmp_path / "map.html"
    out.write_text("previous complete map")
    monkeypatch.setattr(page, "render", lambda payload: "<html>" + "\udcff")
    with pytest.raises(UnicodeEncodeError):
        page.write(PAYLOAD, out)
    assert out.read_text() == "previous complete map"
    assert [p.name for p in tmp_path.iterdir()] == ["map.html"]


def test_assets_are_read_and_written_as_utf8(tmp_path, monkeypatch):
    # The page declares utf-8 and app.js carries non-ASCII, so no file access may
    # fall back to the locale's encoding (cp1252 on a stock Windows install)
    import pathlib

    reads = []
    original = pathlib.Path.read_text

    def checked(self, *args, **kwargs):
        reads.append((self.name, kwargs.get("encoding")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", checked)
    text = "Sainte-Geneviève ✓"
    out = page.write({**PAYLOAD, "town": text}, tmp_path / "map.html")

    # Each of the three assets was read, and every read named utf-8
    assert {name for name, _ in reads} == {"template.html", "app.css", "app.js"}
    assert all(enc == "utf-8" for _, enc in reads), reads
    # Write side: page.write goes through os.fdopen, not Path.write_text, so the
    # spy above never sees it. The payload is dumped without ASCII escapes, so the
    # text reaches the file raw and its UTF-8 bytes must be there.
    assert text.encode("utf-8") in out.read_bytes()


def test_atomic_writes_cover_the_caches_too(tmp_path, monkeypatch):
    # The boundary cache is the slowest thing to rebuild; a truncated one used to
    # read back as empty
    from maybemap import files, geocoding

    target = tmp_path / "boundaries.json"
    target.write_text('{"KEPT": null}')
    monkeypatch.setattr(geocoding, "BOUNDARY_CACHE_PATH", target)
    monkeypatch.setattr(geocoding, "BOUNDARY_WKT", {"NEW": None})

    def interrupted(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(files.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        geocoding._save_boundary_cache()
    assert target.read_text() == '{"KEPT": null}'
    assert [p.name for p in tmp_path.iterdir()] == ["boundaries.json"]


def test_atomic_write_creates_folders(tmp_path):
    from maybemap import files

    assert files.write_text_atomic(tmp_path / "x" / "y.txt", "ok").read_text() == "ok"


def test_department_changes_refresh_the_open_card():
    # The town card lists nearest technicians, which the department filter decides,
    # but update() leaves the card alone, so every handler that flips a department
    # has to re-render it as well
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    writes = [
        line for line in source.splitlines() if re.search(r"state\.depts\[[^\]]+\]\s*=", line)
    ]
    assert writes, "no department handler found in app.js"
    for line in writes:
        assert "renderCard()" in line, line
