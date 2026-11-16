# Pitfalls

Classes of bug that reviews have caught here. Check every change against them, and add an entry when a review
finds a new class.

## A failed lookup cached as "nothing there"

A geocoder, boundary or routing call that never got an answer (network error, rate limit,
5xx, retries exhausted) was returned as `None`, and the caller cached `None` as a real
"not found". One bad minute of network then hides a town or its boundary for good.

**Check:** every external lookup separates "answered, empty" (cache it) from "no answer"
(raise and cache nothing). Look at every `return None` inside an `except` or retry loop,
and at every cache write that follows it.

The quiet variant: an empty answer that might really be an outage (an unbroken run of
misses) is cached before the outage check runs. When the check then fires, the evidence
is already saved as permanent "nothing there".

**Where:** the pre-push gate on the combined-map branch (`geoapify_forward`, the Overpass
fallback in `fetch_osm_city_polygon`, and the miss caching in `load_city_boundaries`).

## Data inlined into a `<script>` block

Escaping only `</` is not enough. `<!--` followed by `<script` inside the block puts the
HTML parser in the double-escaped state, where the real `</script>` no longer closes it.
The same goes for any value written into an HTML attribute (colors read from a KMZ).

**Check:** inlined JSON escapes every `<` (`<`), and values that reach attributes or
`innerHTML` are validated (hex colors) or escaped.

**Where:** the pre-push gate on the combined-map branch (`page._inline_json`,
`kml_color_to_hex`).

## A cache that reads "known empty" as "not cached"

A cache that stores `None` for "looked it up, nothing there" must not hand back that same
`None` for a key it has never seen. If it does, every known gap gets re-fetched on every
build (slow, and it hits rate limits), and anything that counts misses counts the known
gaps too.

**Check:** a cache read returns a distinct sentinel for a missing key (`NOT_CACHED`), and
callers branch on that sentinel, not on truthiness.

**Where:** the pre-push gate on the combined-map branch (`get_cached_osm_polygon`, and
the consecutive-miss abort in `load_city_boundaries`).

## A filter that leaves the open card stale

`update()` redraws the map layers and controls, but not the selected town's card. A
control whose state feeds the card (the department filter decides its nearest
technicians and greys out tech rows) left the card showing the old answer until the
town was picked again.

**Check:** for every new piece of `state`, list what reads it. If `renderCard()` or
another panel outside `update()` reads it, the handler that changes it re-renders that
panel too.

**Where:** Codex on PR #11 (the department buttons).

## Real roster data in a test fixture

A fixture copied from the real tech export put technicians' names and Tech IDs into
the public repo. A first name and an initial looks made up, but it is still a real
person.

**Check:** fixtures use invented names and IDs (`zz…` IDs). Before pushing a test that
contains people, grep the gitignored `data/` exports for each name and ID it uses.

**Where:** Codex on PR #11 (`tests/test_dispatch_sources.py`).

## A secret quoted back in an error message

`requests` puts the full request URL in its exception text ("... for url: ...&apiKey=...").
Folding that text into a build error printed the Geoapify key to the terminal and any log
that captured it.

**Check:** any message built from a request exception for a keyed service has the key
replaced, and it is not chained with `from exc` (the traceback would print the original).

The key is quoted in the URL's encoded form too: a `+`, `/`, `=` or space in it becomes `%2B`,
`%2F`, `%3D` or `+`, and a plain `replace(key, ...)` misses that. Redact every form
(`_redact`: raw, `quote`, `quote_plus`, `urlencode`), and test with a key that encoding changes.

**Where:** the pre-push gate, filed as #5 (`geoapify_forward`), and #16 for the encoded forms.

## A spy test that never sees the path it claims to cover

`test_assets_are_read_and_written_as_utf8` spied on `Path.read_text` and `Path.write_text`
and asserted every recorded call named utf-8. The page is written through `os.fdopen`,
so the write spy recorded nothing, and `seen and all(...)` still passed with only the
reads in it. A write in the wrong encoding would have gone unnoticed.

**Check:** a test that records calls also asserts the calls it is about were recorded
(by name, not just "something was"), and a path the spy cannot see is checked by its
result instead (the bytes on disk).

**Where:** issue #8, found by the pre-push gate.

## Aggregating paired columns independently

`groupby(...).agg(lat="first", lon="first")` takes each column's first non-null value on its
own. A town on several export rows, one with only a latitude and another with only a
longitude, got a point built from two different rows, and it was never geocoded.

**Check:** values that only mean something together (latitude and longitude) are taken from
one row: filter to the rows that have all of them, then take the first. When no row has
the whole set, all of them are NA.

**Where:** the pre-push gate, filed as #15 (`load_city_points`).

## A style key taken from a different variable

`renderPins` had two arrays: `count` (techs a pin names, used to order labels) and `outside` (is the
pin orange). The label's CSS class was keyed on `count`, so every teal pin with an active tech got
the orange label box while its dot was teal. The search list had the same mix-up with `techs.length`.

**Check:** a color or style comes from the same flag as the thing it must match (`pinIsOrange`), never
from a value that was computed for another job (ordering, priority, counts). When a new reader of that
flag depends on a filter, the filter's handler redraws it too.

**Where:** issue #29 (label and search-swatch colors after #28).

## An override row that matches nothing is silently dropped

A row in an operator's file (`region_hubs.csv`) names a region by its text. A typo in the Region
matched no region in the export, so the row did nothing and the build went on with the default hub,
with no sign that the override was never applied.

**Check:** every row of an operator-edited file is either applied or reported. After matching rows to
what they refer to, the leftovers stop the build, naming each one and the file. The match uses the
same comparison as everywhere else for that key (exact, edges stripped). The same goes for an error
about a bad value: it names the file the value came from (a hub's town is in `region_hubs.csv`, a
tech's own town is in `tech_info.csv`), not the one the code happened to be reading.

**Where:** issues #37 and #38, found by the pre-push gate on PR #39.

## An import that guesses or overwrites instead of skipping

The notes import took the first same-name town when a row had no State, and wrote a blank Notes
cell over the note already saved for that town (counting it as imported). An empty file also
crashed on the header read instead of saying what was expected.

**Check:** an import row is applied only when it names exactly one target and carries a value. A
blank or missing value never overwrites saved data, and a name that matches several targets is
skipped and listed in the result, not guessed. Empty input gets the same message as a wrong
header. Text from the file reaches the user only as plain alert text.

**Where:** issues #46, #47 and #48 (`importNotes`).

## A layout change that leaves old offsets behind

Moving UI off the map (a floating topbar became a header above `#map`) left the JS
padding that reserved room for it. On a short landscape phone that padding was taller
than the map itself, Leaflet got a negative fit height, and the page stopped before the
layers were built.

**Check:** when a panel moves, grep the JS for the pixel offsets that made room for it
(`fitBounds` padding, popup and legend offsets), and check them against the smallest
map the CSS allows (landscape phones included).

**Where:** Codex on PR #62 (the initial `fitBounds` in `app.js`).
