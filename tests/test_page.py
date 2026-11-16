# The single-file HTML: everything inlined, and data that cannot break out of its
# script block.

import json
import re

import pytest

from maybemap import branding, page

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
    for marker in (
        "__TITLE__",
        "/*__BRAND__*/",
        "__LOGO__",
        "/*__CSS__*/",
        "/*__JS__*/",
        "__PAYLOAD__",
    ):
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
    monkeypatch.setattr(page, "render", lambda payload, branding=None: "<html>" + "\udcff")
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
    assert "Drive time from" in html
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


def test_page_order_is_header_then_search_then_destination_then_card_then_layers():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    offsets = [
        html.index(marker)
        for marker in (
            'class="appbar"',
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


COLLAPSIBLE = {
    "zonesSection": "zonesBody",
    "outlinesSection": "outlinesBody",
    "dotsSection": "dotsBody",
    "filterSection": "filterBody",
    "dispatchSection": "dispatchBody",
    "stagingSection": "stagingBody",
    "basemapSection": "basemapBody",
    "notesSection": "notesBody",
}


def _section(html, section_id):
    match = re.search(rf'<section[^>]*id="{section_id}".*?^  </section>', html, re.M | re.S)
    assert match, section_id
    return match.group(0)


def test_every_collapsible_section_has_a_toggle_for_its_body():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    for section_id, body_id in COLLAPSIBLE.items():
        section = _section(html, section_id)
        buttons = re.findall(r"<button[^>]*\bcollapse-btn\b[^>]*>", section)
        assert len(buttons) == 1, section_id
        button = buttons[0]
        assert 'type="button"' in button
        assert 'aria-expanded="true"' in button
        assert f'aria-controls="{body_id}"' in button
        assert f'id="{body_id}"' in section
        assert "aria-label=" in button and "data-name=" in button


def test_the_toggle_is_never_inside_a_layer_switch_label():
    # Clicking a layer's header label flips its switch, so the toggle sits beside it
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    assert html.count('class="collapse-btn"') == len(COLLAPSIBLE)
    for label in re.findall(r"<label\b.*?</label>", html, re.S):
        assert "collapse-btn" not in label


def test_search_destination_and_selected_sections_have_no_toggle():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    for section_id in ("searchSection", "driveToSection", "selectedSection"):
        match = re.search(rf'<div[^>]*id="{section_id}".*?\n  </div>\n', html, re.S)
        assert match, section_id
        assert "collapse-btn" not in match.group(0), section_id


def test_collapsed_set_is_stored_apart_from_the_map_state():
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "const COLLAPSE_KEY = 'maybemap_collapsed_v1';" in source
    assert "const STATE_KEY = 'maybemap_state_v1';" in source


def test_collapsing_only_changes_the_sidebar_not_the_map():
    # Collapse is not a switch: its handler never redraws the map or touches state
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    for name in ("setCollapsed", "toggleCollapsed"):
        handler = re.search(rf"^function {name}\(.*?^\}}$", source, re.M | re.S)
        assert handler, f"{name} not found"
        body = handler.group(0)
        assert "update(" not in body and "state." not in body and "persist(" not in body, name
    toggle = re.search(r"^function toggleCollapsed\(.*?^\}$", source, re.M | re.S).group(0)
    assert "writeStore(COLLAPSE_KEY" in toggle
    # The click handler goes straight to toggleCollapsed, and nothing else calls update()
    assert "addEventListener('click', () => toggleCollapsed(section))" in source


def test_collapsed_sections_hide_only_their_body():
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    assert ".section.collapsed .layer-body,.section.collapsed .section-body{display:none}" in css


BRANDED = branding.Branding(
    title="unused",
    dark="#20245C",
    primary="#352B79",
    secondary="#883F98",
    accent="#FDB912",
    logo="data:image/png;base64,TE9HTw==",
    heading_font="data:font/woff2;base64,SEVBRA==",
    body_font="data:font/woff2;base64,Qk9EWQ==",
)


def test_branding_is_inlined_and_the_logo_shown():
    html = page.render(PAYLOAD, BRANDED)
    assert ":root{--brand-dark:#20245C;--brand:#352B79;--brand-2:#883F98;--accent:#FDB912}" in html
    assert "src:url(data:font/woff2;base64,SEVBRA==)" in html
    assert "src:url(data:font/woff2;base64,Qk9EWQ==)" in html
    assert '<img class="brand-logo" src="data:image/png;base64,TE9HTw==" alt="">' in html
    # The brand block comes before the stylesheet that uses it
    assert html.index("--brand-dark:") < html.index(".appbar{")


def test_without_branding_the_page_is_neutral_and_the_logo_hidden():
    html = page.render(PAYLOAD)
    assert "@font-face" not in html
    assert '<img class="brand-logo" src="" alt="" hidden>' in html
    assert f"--brand-dark:{branding.Branding().dark}" in html


def test_a_marker_inside_the_title_stays_literal():
    # Markers are replaced in one pass, so a title can't pull the logo or the script into <title>
    html = page.render({**PAYLOAD, "title": "__LOGO__ /*__JS__*/"}, BRANDED)
    assert "<title>__LOGO__ /*__JS__*/</title>" in html


def test_the_page_loads_nothing_but_leaflet_and_esri_tiles():
    html = page.render(PAYLOAD, BRANDED)
    for url in re.findall(r"https?://[^\s'\"<>)]+", html):
        assert url.startswith(
            ("https://unpkg.com/leaflet@1.9.4/", "https://server.arcgisonline.com/")
        ) or url.startswith(("https://www.openstreetmap.org", "https://www.google.com/maps")), url


def test_no_company_branding_is_committed_in_the_source():
    # Company names, colors and assets come from .env; the tracked page and config stay neutral
    root = page.WEB_DIR.parent
    for path in [*page.WEB_DIR.iterdir(), root / "config.py", root / "branding.py"]:
        text = path.read_text(encoding="utf-8").lower()
        assert "socket" not in text, path
        for color in ("#20245c", "#352b79", "#883f98", "#fdb912"):
            assert color not in text, (path, color)


def test_the_old_stats_strip_and_filler_copy_are_gone():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    for gone in (
        'id="stats"',
        "panel-head",
        "eyebrow",
        "Independent switches",
        "Town notes · Backup",
    ):
        assert gone not in html, gone
    assert "$('#updated').textContent = 'Updated ' + D.builtAt;" in source


def test_initial_fit_padding_leaves_room_on_a_landscape_phone():
    # The header sits above #map, not over it, so the first fit needs no room for it.
    # A 375px-tall phone in landscape leaves #map about 130px (57vh minus the
    # 84px header); padding that eats more than that gives Leaflet a negative
    # fit height and the page stops before the layers are built
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    fits = re.findall(r"map\.fitBounds\(startBounds,\s*\{([^}]*)\}", source)
    assert fits, "no initial fitBounds found in app.js"
    for options in fits:
        vertical = 0
        for key, y in re.findall(r"(padding\w*)\s*:\s*\[\s*\d+\s*,\s*(\d+)\s*\]", options):
            vertical += int(y) * (2 if key == "padding" else 1)
        assert vertical <= 60, options


def _js_function(source, name):
    # The body of a top-level `function name() { ... }`, up to the next top-level line
    start = source.index("function " + name + "(")
    end = source.index("\n}\n", start)
    return source[start:end]


def test_the_legend_and_update_stamp_are_the_sidebar_footer():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    side, side_end = html.index('<aside class="sidebar"'), html.index("</aside>")
    foot = html.index('<footer class="sidebar-foot"')
    # The legend, then the small update stamp under it, after every section
    order = [
        html.index('id="notesSection"'),
        foot,
        html.index('id="legend"'),
        html.index('id="updated"'),
    ]
    assert side < order[0] and order == sorted(order) and order[-1] < side_end
    # The header keeps only the title
    bar = html[html.index('<header class="appbar"') : html.index("</header>")]
    assert "Updated" not in bar and 'id="updated"' not in bar and 'id="subtitle"' not in html
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    # The footer stays in view at the bottom of the sidebar while the sections scroll
    rule = re.search(r"\.sidebar-foot\{([^}]*)\}", css).group(1)
    assert "position:sticky" in rule and "bottom:0" in rule


def test_the_sidebar_sits_on_the_right():
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    assert "#map{position:absolute;inset:var(--bar) var(--side) 0 0;" in css
    rule = re.search(r"\.sidebar\{([^}]*)\}", css).group(1)
    assert "right:0" in rule and "left:0" not in rule


def test_the_title_is_centered_between_logo_and_presets():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    bar = html[html.index('<header class="appbar"') : html.index("</header>")]
    assert bar.index('id="resetMap"') < bar.index('id="title"') < bar.index('class="presets"')
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    rule = re.search(r"\.appbar\{([^}]*)\}", css).group(1)
    assert "grid-template-columns:1fr auto 1fr" in rule
    assert re.search(r"\.brand-text\{[^}]*text-align:center", css)


def test_the_logo_is_the_reset_button():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    button = re.search(r'<button[^>]*id="resetMap"[^>]*>(.*?)</button>', html, re.S)
    assert button and 'class="brand-logo"' in button.group(1)
    assert 'aria-label="Reset the map"' in button.group(0)
    # Without a logo the button still has something to click
    assert "home-icon" in button.group(1)
    # A branded build shows the logo inside it
    branded = page.render(PAYLOAD, BRANDED)
    assert '<img class="brand-logo" src="data:image/png;base64,TE9HTw==" alt="">' in branded


def test_reset_restores_defaults_and_keeps_the_viewers_own_data():
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    body = _js_function(source, "resetMap")
    # Every saved setting goes back to its default, then the map redraws and saves that
    assert "for (const k of Object.keys(DEFAULTS)) state[k] = structuredClone(DEFAULTS[k]);" in body
    assert "update();" in body
    # The open town, the destination and both searches close
    for step in ("clearSelection();", "driveToKey = null;", "$('#townFilter').value = ''"):
        assert step in body, step
    assert "renderDirectory();" in body
    assert "map.fitBounds(startBounds" in body
    # Notes and collapsed sections belong to the viewer, not the map's settings
    for key in ("NOTES_KEY", "COLLAPSE_KEY", "notes", "setCollapsed"):
        assert key not in body, key
    assert "$('#resetMap').addEventListener('click', resetMap);" in source


def test_dots_are_sized_by_customers_and_the_legend_lists_the_tiers():
    # The old four-step bucket radius is gone; the radius comes from dotRadius, the selection ring
    # clears the town's own dot, and the legend lists the size tiers, built with a text node
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "DOT_RADIUS" not in source
    dots = re.search(r"^const dotLayers = .*?^\}\);$", source, re.M | re.S)
    assert dots and "dotRadius(c.customers" in dots.group(0)
    select = re.search(r"^function select\(.*?^\}$", source, re.M | re.S)
    assert select and "highlight.setRadius(" in select.group(0)
    assert "dotRadius(" in select.group(0)
    legend = re.search(r"^function renderLegend\(\) \{.*?^\}$", source, re.M | re.S)
    assert legend and "DOT_TIERS.forEach" in legend.group(0)
    assert "Customers" in legend.group(0)
    assert "createTextNode(text)" in legend.group(0)


def _css_rule(css, selector):
    # The declarations of the rule that starts a line with exactly this selector
    match = re.search(r"^" + re.escape(selector) + r"\{([^}]*)\}", css, re.M)
    assert match, selector
    return match.group(1)


SURFACE_RULES = (
    "html,body",
    "#map",
    ".sidebar",
    ".select",
    ".seg button",
    ".seg button.active",
    "textarea",
    ".action.secondary",
    ".sidebar-foot",
    ".leaflet-tooltip.techlabel",
    ".leaflet-tooltip.marketlabel",
    ".leaflet-tooltip.edge-label",
    ".leaflet-tooltip.hover-tip",
    ".legend .sw",
)


def test_surfaces_take_their_colors_from_tokens_not_light_literals():
    # A surface that hardcodes white (or the map gray) stays light in dark mode. The data colors
    # (pins, routes, zone and region palettes) are not in this list on purpose
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    for selector in SURFACE_RULES:
        body = _css_rule(css, selector)
        for value in re.findall(r"background(?:-color)?:([^;]*)", body):
            assert not re.search(r"#fff(?:fff)?(?![0-9a-f])|\bwhite\b|#dfe3e8", value, re.I), (
                selector,
                value,
            )
    # The map gray lives in one token, and the tooltips and scrollbar in theirs
    assert len(re.findall(r"#dfe3e8", css, re.I)) == 1
    root = _css_rule(css, ":root")
    for token in ("--surface", "--pop", "--map-bg", "--heading", "--brand-ui", "--tech-ink"):
        assert token + ":" in root, token
    assert "#c3c8d2" not in re.sub(r"--scroll:#c3c8d2|--switch-off:#c3c8d2", "", css)


def _dark_blocks(css):
    # The system-following block (in a media query) and the forced one
    auto = re.search(
        r"@media \(prefers-color-scheme: dark\)\{\s*"
        r':root:not\(\[data-theme="light"\]\)\{([^}]*)\}\s*\}',
        css,
    )
    forced = re.search(r':root\[data-theme="dark"\]\{([^}]*)\}', css)
    assert auto and forced
    return auto.group(1), forced.group(1)


def test_both_dark_blocks_define_the_same_tokens_with_one_body():
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    auto, forced = _dark_blocks(css)
    assert auto == forced
    names = re.findall(r"(--?[a-z0-9-]+):", auto)
    assert len(names) > 20 and len(names) == len(set(names))
    assert "color-scheme" in re.findall(r"([a-z-]+):", auto)
    # A dark token must exist in :root too, or the light theme would lack the property
    root = _css_rule(css, ":root")
    for name in names:
        assert name + ":" in root, name


def test_dark_brand_colors_are_derived_never_hardcoded():
    css = (page.WEB_DIR / "app.css").read_text(encoding="utf-8")
    auto, _ = _dark_blocks(css)
    brand_ui = re.search(r"--brand-ui:([^;]*)", auto).group(1)
    heading = re.search(r"--heading:([^;]*)", auto).group(1)
    # Same hue and chroma as the brand variable, only the lightness is floored
    assert re.fullmatch(r"oklch\(from var\(--brand\) max\(l,\.\d+\) c h\)", brand_ui), brand_ui
    assert re.fullmatch(r"oklch\(from var\(--brand-dark\) max\(l,\.\d+\) c h\)", heading), heading
    # The brand tokens here name no hex color, and the blocks never redefine the brand itself
    assert "#" not in brand_ui + heading
    assert "--brand:" not in auto and "--brand-dark:" not in auto


def test_the_theme_toggle_lives_in_the_sidebar_footer_and_the_head_applies_it_early():
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    foot = html[html.index('<footer class="sidebar-foot"') : html.index("</footer>")]
    assert '<button type="button" class="theme-toggle" id="themeToggle"' in foot
    assert foot.index('id="updated"') < foot.index('id="themeToggle"')
    # The icon button has a name and a title, and no visible text
    button = re.search(r'<button[^>]*id="themeToggle"[^>]*>(.*?)</button>', foot, re.S)
    assert 'aria-label="Switch to dark mode"' in button.group(0)
    assert 'title="Switch to dark mode"' in button.group(0)
    # Only a light and a dark icon: there is no visible auto state
    assert "i-auto" not in html
    assert not re.sub(r"<svg.*?</svg>", "", button.group(1), flags=re.S).strip()
    # An inline script in the head, before the stylesheet, sets data-theme from the same storage key
    head = html[: html.index("</head>")]
    early = re.search(
        r"<script>(?:(?!</script>).)*maybemap_theme_v1(?:(?!</script>).)*</script>", head, re.S
    )
    assert early and head.index(early.group(0)) < head.index("<style>")
    assert "documentElement.dataset.theme" in early.group(0) and "catch" in early.group(0)
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "const THEME_KEY = 'maybemap_theme_v1';" in source
    # The click flips light and dark, and the system's own switch redraws the page and the icon
    assert "themeMode = toggleTheme(themeMode, systemDark.matches);" in source
    assert "systemDark.addEventListener('change', applyTheme)" in source


def test_reset_keeps_the_theme_and_the_basemap_follows_it():
    source = (page.WEB_DIR / "app.js").read_text(encoding="utf-8")
    body = _js_function(source, "resetMap")
    assert "THEME_KEY" not in body and "theme" not in body.lower()
    defaults = re.search(r"^const DEFAULTS = \{.*?^\};$", source, re.M | re.S).group(0)
    assert "theme" not in defaults.lower()
    # The dark gray tiles are Esri's too; the toggle and the system switch both redraw the basemap
    assert "ESRI + 'Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'" in source
    assert "ESRI + 'Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}'" in source
    assert "renderBasemap();" in _js_function(source, "applyTheme")
    # A stored theme is only accepted from the three values
    assert "themeFrom(readStore(THEME_KEY))" in source
    html = (page.WEB_DIR / "template.html").read_text(encoding="utf-8")
    assert 'data-value="gray">Gray<' in html
