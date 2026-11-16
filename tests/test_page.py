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
    # The town card lists its closest techs, which the department filter decides,
    # but update() leaves the card alone, so every handler that flips a department
    # has to re-render it as well
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    writes = [
        line for line in source.splitlines() if re.search(r"state\.depts\[[^\]]+\]\s*=", line)
    ]
    assert writes, "no department handler found in app.js"
    for line in writes:
        assert "renderCard()" in line, line
        # The search swatches follow the same filter (orange pins only)
        assert "renderDirectory()" in line, line


def test_label_and_search_colors_follow_the_orange_pin_rule():
    # count[] is the label placement priority, and a teal pin with an active tech has count > 0,
    # so keying the label class on it painted in-service towns orange. The class and the search
    # swatch both come from pinIsOrange, the rule the pin icon uses
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    render = re.search(r"^function renderPins\(\) \{.*?^\}$", source, re.M | re.S)
    assert render
    class_line = next(line for line in render.group(0).splitlines() if "className:" in line)
    assert "count[" not in class_line
    assert "outside[i]" in class_line
    assert "pinIsOrange" in render.group(0)
    directory = re.search(r"^function renderDirectory\(\) \{.*?^\}$", source, re.M | re.S)
    assert directory and "pinIsOrange(" in directory.group(0)
    assert "techs.length ? ' tech'" not in directory.group(0)


def test_drive_time_destination_is_built_once_outside_the_card():
    # The destination sits under the town search, not in the card. Its input is built once, so a
    # department change (which re-renders the card) cannot wipe a half-typed search, and the
    # chosen destination lives in a top-level variable that renderCard() only reads
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    assert "Drive time to…" in html
    assert re.search(r"^let driveToKey\b", source, re.M)
    card = re.search(r"^function renderCard\(\) \{.*?^\}$", source, re.M | re.S)
    assert card
    assert "renderDriveTo(" in card.group(0)
    assert "driveToSection(" not in card.group(0)
    assert "driveToSearch" not in card.group(0)
    draw = re.search(r"^function renderDriveTo\(\) \{.*?^\}$", source, re.M | re.S)
    assert draw and "driveToSection(" not in draw.group(0)
    # Selecting a town refreshes it through renderCard, clearing hides it
    clear = re.search(r"^function clearSelection\(\) \{.*?^\}$", source, re.M | re.S)
    assert clear and "renderDriveTo()" in clear.group(0)
    # Its pick and clear handlers redraw only the destination, not the whole card
    builder = re.search(r"^function driveToSection\(\) \{.*?^\}$", source, re.M | re.S)
    assert builder and "renderCard()" not in builder.group(0)
    assert "renderDriveTo()" in builder.group(0)
    assert draw and "renderCard()" not in draw.group(0)


def test_search_box_follows_how_a_town_was_opened():
    # Opening a town from anywhere but its own search result (a destination row, a closest-town or
    # tech row, a pin or a region) empties the search box and its results. Clicking a search result
    # keeps the query, so other results can be tried, and redraws the list so the picked row is
    # marked active
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    pick = re.search(r"^function select\(key, fly, fromSearch\) \{.*?^\}$", source, re.M | re.S)
    assert pick
    body = pick.group(0)
    assert re.search(r"if \(!fromSearch\) \$\('#townFilter'\)\.value = '';", body)
    # The list is redrawn after the selection is set, so the picked row shows .active
    assert body.index("selectedKey = key") < body.index("renderDirectory()")
    # Only the main search's result buttons pass fromSearch
    directory = re.search(r"^function renderDirectory\(\) \{.*?^\}$", source, re.M | re.S)
    assert directory and "select(s.key, true, true)" in directory.group(0)
    # ... and so does Enter in the main search, which opens the first result
    assert "select(first.key, true, true)" in _town_filter_keydown(source)
    call = r"(?<![\w.])select\((?:keyOf\([^()]*\)|[\w.]+), (?:true|false)(?:, true)?\)"
    calls = re.findall(call, source)
    assert len(calls) == 7
    assert sum(c.endswith(", true, true)") for c in calls) == 2


def _town_filter_keydown(source):
    handler = re.search(
        r"^\$\('#townFilter'\)\.addEventListener\('keydown', .*?^\}\);$", source, re.M | re.S
    )
    assert handler, "#townFilter has no keydown handler"
    return handler.group(0)


def test_search_results_show_the_state():
    # Towns share names across states, so every result row reads "Name, ST" (through el(), so
    # textContent), the same as the drive-time suggestions
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    directory = re.search(r"^function renderDirectory\(\) \{.*?^\}$", source, re.M | re.S)
    assert directory
    assert "el('span', '', s.t.name + ', ' + s.t.state + (getNote(s.key) ? ' 📝' : ''))" in (
        directory.group(0)
    )
    assert ".innerHTML" not in directory.group(0)


def test_town_search_has_enter_and_escape():
    # Like the drive-time search: Enter opens the first result shown (as a search-result
    # selection, so the query stays), Escape clears the box and the list
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    body = _town_filter_keydown(source)
    assert "'Enter'" in body and "'Escape'" in body
    assert "directoryHits(" in body and "preventDefault()" in body
    assert "select(first.key, true, true)" in body
    escape = body[body.index("'Escape'") :]
    assert ".value = ''" in escape and "renderDirectory()" in escape
    # The result list and Enter share one hit list, so Enter opens exactly the first row shown
    directory = re.search(r"^function renderDirectory\(\) \{.*?^\}$", source, re.M | re.S)
    assert directory and "directoryHits(" in directory.group(0)


def test_sidebar_order_is_search_then_destination_then_card_then_layers():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    offsets = [
        html.index(marker)
        for marker in (
            'id="stats"',
            'id="searchSection"',
            'id="driveToSection"',
            'id="selectedSection"',
            'data-layer="zones"',
        )
    ]
    assert offsets == sorted(offsets)
    assert html.count('id="driveToSection"') == 1
    # The destination starts hidden, until a dispatch town is selected
    assert re.search(r'<[^>]*id="driveToSection"[^>]*\bhidden\b', html)


def test_search_results_are_labeled_by_town_only():
    # A tech's name in a result label made towns look like people; the index still holds the
    # tech names, so searching one finds their town
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    directory = re.search(r"^function renderDirectory\(\) \{.*?^\}$", source, re.M | re.S)
    assert directory and "techList(" not in directory.group(0)
    index = re.search(r"^const searchIndex = .*?^\}\);$", source, re.M | re.S)
    assert index and "x.name" in index.group(0)


def test_clicked_regions_draw_no_focus_box():
    # Clicking a region drew a box around its path. Region paths can't take keyboard focus, so
    # their focus ring goes (plain :focus). The map container keeps its ring for keyboard panning,
    # so only its mouse focus is removed (:focus-visible), never a plain :focus
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    assert "path.leaflet-interactive:focus{outline:none}" in css
    assert ".leaflet-container:focus:not(:focus-visible){outline:none}" in css
    assert ".leaflet-container:focus{outline:none}" not in css


def test_card_shows_the_closest_techs_for_every_town():
    # The section is built outside the dispatch-town branch, or a town with no region gets none
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    card = re.search(r"^function renderCard\(\) \{.*?^\}$", source, re.M | re.S)
    assert card and "closestTechsSection(t)" in card.group(0)
    assert "closestTechsSection" not in re.search(
        r"if \(t\.pi >= 0\) \{.*?\n  \}", card.group(0), re.S
    ).group(0)


def test_basemap_tiles_come_only_from_esri():
    # OSM's tile server 403s browser requests with no Referer (every file:// map), so
    # no basemap may use it. Every tile layer must be built from the ESRI prefix, and
    # the regex must find them, or the check passes on nothing.
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "tile.openstreetmap.org" not in source
    assert "ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/'" in source
    urls = re.findall(r"L\.tileLayer\(\s*([^,]+?),", source)
    assert len(urls) >= 5
    for url in urls:
        assert url.startswith("ESRI + '"), url
    assert "ESRI + 'World_Street_Map/MapServer/tile/{z}/{y}/{x}'" in source


def test_unassigned_region_defaults_off_in_source():
    # The gray Unassigned dots start hidden: DEFAULTS.regionsOn is built from the payload's
    # unassigned label, not from allOn(regionNames), which turned every region on
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "regionsOn: allOn(regionNames)" not in source
    default = "regionsOn: Object.fromEntries(regionNames.map(n => [n, n !== D.unassignedLabel]))"
    assert default + "," in source
