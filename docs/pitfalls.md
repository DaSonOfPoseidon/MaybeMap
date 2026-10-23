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

**Where:** the pre-push gate on the combined-map branch (`geoapify_forward`, and the
Overpass fallback in `fetch_osm_city_polygon`).

## Data inlined into a `<script>` block

Escaping only `</` is not enough. `<!--` followed by `<script` inside the block puts the
HTML parser in the double-escaped state, where the real `</script>` no longer closes it.
The same goes for any value written into an HTML attribute (colors read from a KMZ).

**Check:** inlined JSON escapes every `<` (`<`), and values that reach attributes or
`innerHTML` are validated (hex colors) or escaped.

**Where:** the pre-push gate on the combined-map branch (`page._inline_json`,
`kml_color_to_hex`).
