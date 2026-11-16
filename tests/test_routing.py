# Drive-time matrix: per-pair caching, chunked OSRM requests, and failing loudly.

import pytest

from maybemap import routing

POINTS = [(38.76, -93.74), (38.71, -93.23), (37.84, -94.36)]


class FakeOSRM:
    # Answers like /table: distance and duration grow with the index gap, so any
    # mix-up between sources and destinations shows in the result
    def __init__(self, fail=False, unroutable=None):
        self.calls = []
        self.fail = fail
        self.unroutable = unroutable

    def __call__(self, coords, sources, destinations):
        self.calls.append((coords, sources, destinations))
        if self.fail:
            raise RuntimeError("OSRM table request failed")
        index = [POINTS.index(c) for c in coords]

        def cell(a, b):
            i, j = index[a], index[b]
            if self.unroutable == (i, j):
                return None
            return 1000.0 * (abs(i - j) + 1)

        distances = [[cell(a, b) for b in destinations] for a in sources]
        durations = [[None if d is None else d / 10 for d in row] for row in distances]
        return distances, durations


@pytest.fixture
def fake(monkeypatch):
    f = FakeOSRM()
    monkeypatch.setattr(routing, "_table", f)
    return f


def test_matrix_is_square_with_a_zero_diagonal(fake):
    meters, seconds = routing.drive_matrix(POINTS, cache={}, persist=False)
    assert len(meters) == len(seconds) == 3
    assert [meters[i][i] for i in range(3)] == [0, 0, 0]
    assert meters[0][2] == 3000 and seconds[0][2] == 300


def test_a_warm_cache_makes_no_requests(fake):
    cache = {}
    routing.drive_matrix(POINTS, cache=cache, persist=False)
    fake.calls.clear()
    routing.drive_matrix(POINTS, cache=cache, persist=False)
    assert fake.calls == []


def test_adding_a_place_reuses_the_cached_pairs(fake):
    cache = {}
    routing.drive_matrix(POINTS[:2], cache=cache, persist=False)
    fake.calls.clear()
    meters, _ = routing.drive_matrix(POINTS, cache=cache, persist=False)
    # One request covers the new place's row and column; the old pair is not why
    assert len(fake.calls) == 1
    assert len(cache) == 6
    assert meters[2][0] == 3000 and meters[0][1] == 2000


def test_requests_are_chunked(fake, monkeypatch):
    monkeypatch.setattr(routing, "OSRM_CHUNK", 2)
    routing.drive_matrix(POINTS, cache={}, persist=False)
    assert all(len(srcs) <= 2 and len(dsts) <= 2 for _, srcs, dsts in fake.calls)
    # Three blocks: the bottom-right one holds only the diagonal, so it is never sent
    assert len(fake.calls) == 3


def test_a_failed_request_raises(monkeypatch):
    monkeypatch.setattr(routing, "_table", FakeOSRM(fail=True))
    with pytest.raises(RuntimeError, match="failed"):
        routing.drive_matrix(POINTS, cache={}, persist=False)


def test_an_unroutable_pair_raises_rather_than_leaving_a_hole(monkeypatch):
    monkeypatch.setattr(routing, "_table", FakeOSRM(unroutable=(0, 1)))
    with pytest.raises(RuntimeError, match="no road route"):
        routing.drive_matrix(POINTS, cache={}, persist=False)


def test_table_is_rectangular_origins_by_destinations(fake):
    meters, seconds = routing.drive_table(
        [POINTS[0], POINTS[1]], [POINTS[2]], cache={}, persist=False
    )
    assert len(meters) == len(seconds) == 2
    assert all(len(row) == 1 for row in meters + seconds)
    assert meters[0][0] == 3000 and seconds[0][0] == 300
    assert meters[1][0] == 2000 and seconds[1][0] == 200


def test_table_pairs_are_directional(fake):
    # The fake answers by index gap, so both directions look alike; check the keys
    cache = {}
    routing.drive_table([POINTS[0]], [POINTS[2]], cache=cache, persist=False)
    assert list(cache) == [
        routing._pair_key(routing._point_key(*POINTS[0]), routing._point_key(*POINTS[2]))
    ]
    fake.calls.clear()
    # The reverse direction is a different pair, so it is fetched
    routing.drive_table([POINTS[2]], [POINTS[0]], cache=cache, persist=False)
    assert len(fake.calls) == 1 and len(cache) == 2


def test_table_and_matrix_share_the_cache(fake):
    cache = {}
    routing.drive_matrix(POINTS, cache=cache, persist=False)
    fake.calls.clear()
    # A pair the matrix already fetched is not fetched again by a table
    meters, _ = routing.drive_table([POINTS[1]], [POINTS[0], POINTS[2]], cache=cache, persist=False)
    assert fake.calls == [] and meters == [[2000, 2000]]
    # ...and the other way round
    cache = {}
    routing.drive_table(POINTS, POINTS, cache=cache, persist=False)
    fake.calls.clear()
    routing.drive_matrix(POINTS, cache=cache, persist=False)
    assert fake.calls == []


def test_table_fetches_only_the_missing_pairs(fake):
    cache = {}
    routing.drive_table([POINTS[0]], [POINTS[1]], cache=cache, persist=False)
    fake.calls.clear()
    meters, _ = routing.drive_table([POINTS[0]], [POINTS[1], POINTS[2]], cache=cache, persist=False)
    assert len(fake.calls) == 1
    coords, _, destinations = fake.calls[0]
    # Only the column that holds the missing pair is asked for
    assert [coords[d] for d in destinations] == [POINTS[2]]
    assert meters == [[2000, 3000]]


def test_table_sends_a_point_in_both_lists_once(fake):
    routing.drive_table([POINTS[0], POINTS[1]], [POINTS[1], POINTS[2]], cache={}, persist=False)
    coords = fake.calls[0][0]
    assert len(coords) == len(set(coords)) == 3


def test_table_pair_with_the_same_point_is_zero_without_a_request(fake):
    cache = {}
    meters, seconds = routing.drive_table([POINTS[0]], [POINTS[0]], cache=cache, persist=False)
    assert meters == [[0]] and seconds == [[0]]
    assert fake.calls == [] and cache == {}


def test_table_with_nothing_to_route_makes_no_request(fake):
    assert routing.drive_table([], POINTS, cache={}, persist=False) == ([], [])
    assert routing.drive_table(POINTS[:2], [], cache={}, persist=False) == ([[], []], [[], []])
    assert fake.calls == []


def test_table_is_chunked(fake, monkeypatch):
    monkeypatch.setattr(routing, "OSRM_CHUNK", 1)
    routing.drive_table(POINTS[:2], POINTS, cache={}, persist=False)
    assert all(len(srcs) == 1 and len(dsts) == 1 for _, srcs, dsts in fake.calls)
    # 2 x 3 cells, minus the 2 that pair a point with itself
    assert len(fake.calls) == 4


def test_table_saves_the_cache_after_every_chunk(fake, monkeypatch):
    monkeypatch.setattr(routing, "OSRM_CHUNK", 1)
    saved = []
    monkeypatch.setattr(routing, "save_cache", lambda cache: saved.append(len(cache)))
    routing.drive_table(POINTS[:1], POINTS, cache={})
    assert saved == [1, 2]


def test_table_failure_caches_nothing_for_that_chunk(monkeypatch):
    monkeypatch.setattr(routing, "_table", FakeOSRM(fail=True))
    cache = {}
    with pytest.raises(RuntimeError, match="failed"):
        routing.drive_table(POINTS[:1], POINTS[1:], cache=cache, persist=False)
    assert cache == {}


def test_table_unroutable_pair_raises_naming_the_coordinates(monkeypatch):
    monkeypatch.setattr(routing, "_table", FakeOSRM(unroutable=(0, 2)))
    with pytest.raises(RuntimeError, match=r"no road route.*37\.84"):
        routing.drive_table(POINTS[:2], POINTS[2:], cache={}, persist=False)
