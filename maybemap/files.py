# Writes that never leave a half-written file behind.

import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path | str, text: str) -> Path:
    # Written beside the target and swapped in only once complete, so a failed or
    # interrupted write leaves the previous file in place rather than half of one
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
    return path
