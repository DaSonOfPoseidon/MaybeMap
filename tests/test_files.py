# The atomic writer: the mode a written file ends up with.

import os
import stat

import pytest

from maybemap import files

posix_only = pytest.mark.skipif(os.name == "nt", reason="file modes are not POSIX here")


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    yield
    os.umask(old)


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_a_new_file_gets_the_umask_mode(tmp_path, umask_022):
    # mkstemp creates 0600, which would leave a built map unreadable to the web
    # server's group once the temp file is swapped in
    out = files.write_text_atomic(tmp_path / "map.html", "x")
    assert _mode(out) == 0o644


@posix_only
def test_a_replaced_file_keeps_its_mode(tmp_path, umask_022):
    target = tmp_path / "map.html"
    target.write_text("old")
    target.chmod(0o640)
    files.write_text_atomic(target, "new")
    assert target.read_text() == "new"
    assert _mode(target) == 0o640


@posix_only
def test_the_umask_is_left_as_it_was(tmp_path, umask_022):
    files.write_text_atomic(tmp_path / "map.html", "x")
    assert os.umask(0o022) == 0o022
