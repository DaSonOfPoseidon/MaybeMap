# AGENTS.md

Instructions for AI agents (Codex, Claude) working on this repository. Claude also reads `CLAUDE.md`.

## Code Review Rules

Codex's PR review and the pre-push review gate both follow this section. Rate every finding on this scale:

- **P0**: data loss, a security hole, or a change that breaks the main path for everyone. Never merge.
- **P1**: wrong behaviour on a real path, untrusted input that can be misparsed, a broken project invariant
  (below), or a doc that would lead a user to do something harmful.
- **P2**: an edge case that breaks (0, max, max+1, empty, error paths), a doc or comment promise the code
  doesn't keep, a test that can pass while the code is wrong, or CI/script behaviour that differs from its
  description.
- **P3**: a minor issue a strict reviewer might still flag. P3s are filed as follow-up issues, not fixed in
  the PR that found them.

Every finding needs a concrete failure scenario (input or state → wrong result). Don't flag style, naming,
or anything CI already checks (formatting, lints, tests).

### Project invariants

- **This repo is public. Business data never gets committed.** That covers the city export,
  KMZ, dispatch rosters (tech names), task exports, any `.csv`/`.xlsx`/`.kmz`, lookup caches,
  and built maps (the page inlines customer counts). They live in the gitignored `data/`,
  `cache/` and `output/`. `.env` (API keys) never gets committed either.
- **A failed lookup is not an empty answer.** A network failure from Geoapify, Nominatim,
  Overpass or OSRM must never be cached, or drawn, as "no boundary", "no route" or "no
  coordinates". Retry, then fail the build loudly.
- **The built map is complete or not written.** The drive-time matrix has no holes, and the
  inlined payload is strict JSON (no `NaN`, `</` escaped). The page makes no routing or data
  requests at view time; only Leaflet (pinned version and SRI) and basemap tiles load from
  the network.
- **Untrusted text reaches the page.** That includes export fields (town, county, region
  names), KMZ descriptions, the roster CSVs, imported notes CSVs and localStorage. It goes
  into the DOM through `textContent` or `esc()`, never raw into `innerHTML` or a
  tooltip/popup string.
- **Layers are independent.** Every layer has its own switch and stacks with the others.
  Presets only flip those switches; they are never separate views.
- **Tests never touch the network or the real caches.** Stub geocoding, boundaries and OSRM.
- Python 3.14 with `uv` (not pip). Checks: `uv run ruff check . && uv run black --check . && uv run pytest`.

### Known pitfalls

Classes of bug past reviews caught here are in [`docs/pitfalls.md`](docs/pitfalls.md); check every change
against them.
