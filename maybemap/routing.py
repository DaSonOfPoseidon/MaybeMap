# Drive distance and time between every pair of dispatch places, from OSRM's table
# service. Fetched at build time and cached per pair. Only rows and columns that
# still hold a missing pair are requested, so an unchanged town list makes no
# requests at all.

import json
import time

import requests

from maybemap.config import (
    CACHE_DIR,
    OSRM_BASE_URL,
    OSRM_CHUNK,
    OSRM_RETRIES,
    OSRM_RETRY_PAUSE_S,
)
from maybemap.files import write_text_atomic

MATRIX_CACHE_PATH = CACHE_DIR / "osrm_matrix.json"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MaybeMap/1.0"})


def _point_key(lat: float, lon: float) -> str:
    # Five decimals is about a metre, finer than any town-center geocode moves
    return f"{lat:.5f},{lon:.5f}"


def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}"


def load_cache(path=MATRIX_CACHE_PATH) -> dict[str, list[float]]:
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except FileNotFoundError, json.JSONDecodeError:
        return {}


def save_cache(cache: dict, path=MATRIX_CACHE_PATH):
    write_text_atomic(path, json.dumps(cache, indent=0, sort_keys=True))


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _table(coords: list[tuple[float, float]], sources: list[int], destinations: list[int]):
    # One /table request. coords are (lat, lon); OSRM wants lon,lat.
    path = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)
    url = f"{OSRM_BASE_URL}/table/v1/driving/{path}"
    params = {
        "annotations": "duration,distance",
        "sources": ";".join(map(str, sources)),
        "destinations": ";".join(map(str, destinations)),
    }
    for attempt in range(1, OSRM_RETRIES + 1):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != "Ok":
                raise RuntimeError(f"OSRM answered {data.get('code')}: {data.get('message')}")
            return data["distances"], data["durations"]
        except (requests.RequestException, RuntimeError, KeyError, ValueError) as exc:
            if attempt == OSRM_RETRIES:
                raise RuntimeError(
                    f"OSRM table request failed after {OSRM_RETRIES} attempts "
                    f"({type(exc).__name__}: {exc}). Pairs fetched so far are saved in "
                    f"{MATRIX_CACHE_PATH.name}, so rerunning resumes from here."
                ) from exc
            pause = OSRM_RETRY_PAUSE_S * attempt
            print(f"   OSRM request failed ({type(exc).__name__}), retrying in {pause:.0f}s")
            time.sleep(pause)
    raise AssertionError("unreachable")  # the loop always returns or raises


def drive_matrix(
    points: list[tuple[float, float]], *, cache: dict | None = None, persist: bool = True
) -> tuple[list[list[int]], list[list[int]]]:
    # Returns (meters, seconds), each points x points, rounded to whole units.
    # Raises rather than leave a hole: a dispatch map quietly missing drive times
    # is worse than no map.
    if cache is None:
        cache = load_cache()
    keys = [_point_key(lat, lon) for lat, lon in points]
    n = len(points)

    missing = {
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and _pair_key(keys[i], keys[j]) not in cache
    }
    if missing:
        rows = sorted({i for i, _ in missing})
        cols = sorted({j for _, j in missing})
        print(f"Fetching {len(missing)} drive-time pairs from OSRM ({n} places)...")
        for src in _chunks(rows, OSRM_CHUNK):
            for dst in _chunks(cols, OSRM_CHUNK):
                if not any((i, j) in missing for i in src for j in dst):
                    continue
                # A place in both lists is sent once and indexed from both
                members = sorted(set(src) | set(dst))
                index = {p: k for k, p in enumerate(members)}
                distances, durations = _table(
                    [points[p] for p in members],
                    [index[i] for i in src],
                    [index[j] for j in dst],
                )
                for a, i in enumerate(src):
                    for b, j in enumerate(dst):
                        if i == j:
                            continue
                        meters, seconds = distances[a][b], durations[a][b]
                        if meters is None or seconds is None:
                            raise RuntimeError(
                                f"OSRM found no road route between points {points[i]} "
                                f"and {points[j]}. Check that town's coordinates."
                            )
                        cache[_pair_key(keys[i], keys[j])] = [meters, seconds]
                if persist:
                    save_cache(cache)
    else:
        print(f"All {n * (n - 1)} drive-time pairs answered from the cache")

    meters = [[0] * n for _ in range(n)]
    seconds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m, s = cache[_pair_key(keys[i], keys[j])]
                meters[i][j], seconds[i][j] = round(m), round(s)
    return meters, seconds
