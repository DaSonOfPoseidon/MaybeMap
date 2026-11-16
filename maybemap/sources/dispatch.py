# The dispatch roster: which technician works out of which town (or, lacking a
# town, which region), and which towns host staging sites.

from pathlib import Path

import pandas as pd

from maybemap.config import DISPATCH_STAGING_PATH, DISPATCH_TECHS_PATH
from maybemap.states import state_abbrev

# Optional tech columns from the MBS export, keyed by lower-cased header
TECH_OPTIONAL = {
    "active": "Active",
    "tech id": "TechId",
    "primary region": "Region",
    "departments": "Departments",
}
INACTIVE = {"n", "no", "false", "0", "inactive"}


def _read(path: Path, required: list[str], optional: dict[str, str] | None = None):
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    missing = [c for c in required if c.lower() not in lower]
    if missing:
        raise ValueError(
            f"{path.name} must contain columns {', '.join(required)}. "
            f"Found: {df.columns.tolist()}"
        )
    columns = {lower[c.lower()]: c for c in required}
    columns.update({lower[k]: name for k, name in (optional or {}).items() if k in lower})
    df = df.rename(columns=columns)[list(columns.values())]
    for name in (optional or {}).values():
        if name not in df:
            df[name] = ""
    for c in df.columns:
        df[c] = df[c].str.strip()
    # Same normal form as the city export, so a tech's town matches its city row
    df["Town"] = df["Town"].str.title()
    df["State"] = df["State"].map(lambda s: state_abbrev(s) if s else "")
    return df.reset_index(drop=True)


TECH_COLUMNS = ["Tech", "Town", "State", "TechId", "Region", "Departments"]


def load_techs(path: Path | str = DISPATCH_TECHS_PATH) -> pd.DataFrame:
    # Columns Tech, Town, State, TechId, Region, Departments (a list). Town may be
    # blank: such a tech is placed by Region instead. Inactive rows are dropped. A
    # missing file means no techs, not an error: the map still routes town to town.
    path = Path(path)
    if not path.exists():
        print(f"No {path.name} found; the dispatch layer will carry no technicians")
        return pd.DataFrame(columns=TECH_COLUMNS)
    df = _read(path, ["Tech", "Town", "State"], TECH_OPTIONAL)
    df = df[df["Tech"] != ""]
    inactive = df["Active"].str.lower().isin(INACTIVE)
    if inactive.any():
        print(f"Skipping {int(inactive.sum())} inactive technicians")
    df = df[~inactive].copy()
    df["Departments"] = df["Departments"].map(
        lambda v: [d.strip().upper() for d in v.split(",") if d.strip()]
    )
    print(f"Loaded {len(df)} technicians from {path.name}")
    return df[TECH_COLUMNS].reset_index(drop=True)


def load_staging(path: Path | str = DISPATCH_STAGING_PATH) -> pd.DataFrame:
    # Columns Type, Town, State, in file order; each Type becomes one switch
    path = Path(path)
    if not path.exists():
        print(f"No {path.name} found; the map will carry no staging sites")
        return pd.DataFrame(columns=["Type", "Town", "State"])
    df = _read(path, ["Type", "Town", "State"])
    df = df[(df["Type"] != "") & (df["Town"] != "")]
    print(f"Loaded {len(df)} staging sites from {path.name}")
    return df
