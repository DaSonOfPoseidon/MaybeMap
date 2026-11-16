# The dispatch roster: which technician works out of which town (or, lacking a
# town, which region), and which towns host staging sites.

from pathlib import Path

import pandas as pd

from maybemap.config import DISPATCH_STAGING_PATH, DISPATCH_TECHS_PATH, REGION_HUBS_PATH
from maybemap.states import state_abbrev

# Optional tech columns from the MBS export, keyed by lower-cased header
TECH_OPTIONAL = {
    "active": "Active",
    "tech id": "TechId",
    "primary region": "Region",
    "departments": "Departments",
}
INACTIVE = {"n", "no", "false", "0", "inactive"}


def _read(
    path: Path,
    required: list[str],
    optional: dict[str, str] | None = None,
    town: str = "Town",
):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
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
    df[town] = df[town].str.title()
    df["State"] = df["State"].map(lambda s: state_abbrev(s) if s else "")
    return df.reset_index(drop=True)


def _require_states(df: pd.DataFrame, label: str, path: Path, town: str = "Town"):
    # A town needs its state to be found. Without one the town would be quietly
    # ignored (a tech falls back to their region hub), so name the rows instead.
    # Runs on the rows that are kept, after blank and inactive ones are dropped.
    no_state = (df[town] != "") & (df["State"] == "")
    if no_state.any():
        rows = ", ".join(f"{r[label]} ({r[town]})" for _, r in df[no_state].iterrows())
        raise ValueError(f"{path.name} has a {town} with no State for: {rows}. Fill in the State.")


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
    _require_states(df, "Tech", path)
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
    _require_states(df, "Type", path)
    print(f"Loaded {len(df)} staging sites from {path.name}")
    return df


def load_region_hubs(path: Path | str = REGION_HUBS_PATH) -> dict[str, tuple[str, str]]:
    # Columns Region, Hub, State: the town that is a region's hub, whatever the
    # customer counts say. A missing file means no overrides. A Region with no Hub, a
    # Hub with no State, or one Region given two different hubs stops the build.
    path = Path(path)
    if not path.exists():
        print(f"No {path.name} found; every region's hub comes from the customer counts")
        return {}
    df = _read(path, ["Region", "Hub", "State"], town="Hub")
    df = df[(df["Region"] != "") | (df["Hub"] != "")]
    blank = df[(df["Region"] == "") | (df["Hub"] == "")]
    if not blank.empty:
        raise ValueError(f"{path.name} has a row with no Region or no Hub. Fill in both.")
    _require_states(df, "Region", path, town="Hub")
    hubs: dict[str, tuple[str, str]] = {}
    for row in df.itertuples(index=False):
        if hubs.setdefault(row.Region, (row.Hub, row.State)) != (row.Hub, row.State):
            raise ValueError(f"{path.name} gives region {row.Region} two different hubs.")
    print(f"Loaded {len(hubs)} region hubs from {path.name}")
    return hubs
