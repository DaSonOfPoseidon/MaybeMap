# Writes that never leave a half-written file behind.

import os
import stat
import tempfile
from pathlib import Path


def _final_mode(path: Path) -> int:
    # mkstemp creates files at 0600. A replaced file keeps the mode it had; a new
    # one gets what a plain open() would give it, 0666 minus the umask.
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def write_text_atomic(path: Path | str, text: str) -> Path:
    # Written beside the target and swapped in only once complete, so a failed or
    # interrupted write leaves the previous file in place rather than half of one
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, _final_mode(path))
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    return path
