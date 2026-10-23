# The dispatch roster: which technician works out of which town, and which towns
# host staging sites. Both are small hand-kept CSVs beside the city export.

from pathlib import Path

import pandas as pd

from maybemap.config import DISPATCH_STAGING_PATH, DISPATCH_TECHS_PATH
from maybemap.states import state_abbrev


def _read(path: Path, required: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    missing = [c for c in required if c.lower() not in lower]
    if missing:
        raise ValueError(
            f"{path.name} must contain columns {', '.join(required)}. "
            f"Found: {df.columns.tolist()}"
        )
    df = df.rename(columns={lower[c.lower()]: c for c in required})[required]
    for c in required:
        df[c] = df[c].str.strip()
    df = df[df["Town"] != ""].copy()
    # Same normal form as the city export, so a tech's town matches its city row
    df["Town"] = df["Town"].str.title()
    df["State"] = df["State"].map(state_abbrev)
    return df.reset_index(drop=True)


def load_techs(path: Path | str = DISPATCH_TECHS_PATH) -> pd.DataFrame:
    # Columns Tech, Town, State. A missing file means no techs, not an error: the
    # map still routes town to town without them.
    path = Path(path)
    if not path.exists():
        print(f"No {path.name} found; the dispatch layer will carry no technicians")
        return pd.DataFrame(columns=["Tech", "Town", "State"])
    df = _read(path, ["Tech", "Town", "State"])
    df = df[df["Tech"] != ""]
    print(f"Loaded {len(df)} technicians from {path.name}")
    return df


def load_staging(path: Path | str = DISPATCH_STAGING_PATH) -> pd.DataFrame:
    # Columns Type, Town, State, in file order; each Type becomes one switch
    path = Path(path)
    if not path.exists():
        print(f"No {path.name} found; the map will carry no staging sites")
        return pd.DataFrame(columns=["Type", "Town", "State"])
    df = _read(path, ["Type", "Town", "State"])
    df = df[df["Type"] != ""]
    print(f"Loaded {len(df)} staging sites from {path.name}")
    return df
