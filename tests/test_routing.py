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
