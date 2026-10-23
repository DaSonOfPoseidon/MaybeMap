# The hand-kept dispatch roster CSVs.

import pytest

from maybemap.sources.dispatch import load_staging, load_techs


def _csv(tmp_path, text, name="roster.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_techs_normalize_to_the_city_export_form(tmp_path):
    df = load_techs(_csv(tmp_path, "tech,town,state\n Nate ,SEDALIA,Missouri\n"))
    row = df.to_dict("records")[0]
    assert (row["Tech"], row["Town"], row["State"]) == ("Nate", "Sedalia", "MO")


def test_blank_rows_are_skipped(tmp_path):
    df = load_techs(_csv(tmp_path, "Tech,Town,State\nNate,Sedalia,MO\n,,\n,Clinton,MO\n"))
    assert df["Tech"].tolist() == ["Nate"]


def test_a_missing_column_names_what_was_found(tmp_path):
    with pytest.raises(ValueError, match="Found: \\['Tech', 'Town'\\]"):
        load_techs(_csv(tmp_path, "Tech,Town\nNate,Sedalia\n"))


def test_missing_files_mean_empty_rosters(tmp_path):
    assert load_techs(tmp_path / "nope.csv").empty
    assert load_staging(tmp_path / "nope.csv").empty


def test_staging_keeps_file_order(tmp_path):
    df = load_staging(_csv(tmp_path, "Type,Town,State\nOutage,Nevada,MO\nGenerator,Rich Hill,MO\n"))
    assert df["Type"].tolist() == ["Outage", "Generator"]


TECH_INFO = (
    "Tech,Town,State,Active,Tech ID,In Cities List,Primary Region,Departments\n"
    'Chris T,CLINTON,MO,Y,chtu0170,Y,Clinton-Butler,"FIELD, TC"\n'
    "Alan E,,,Y,alee0170,N,Jefferson City Area,FIELD\n"
    "Old Hand,SEDALIA,MO,N,olha0170,Y,Sedalia-La Monte,FIELD\n"
    "Timothy B,,,Y,tibe0170,N,,FIELDBUS\n"
)


def test_the_mbs_tech_export_is_read_in_full(tmp_path):
    df = load_techs(_csv(tmp_path, TECH_INFO))
    rows = {r["Tech"]: r for r in df.to_dict("records")}
    assert set(rows) == {"Chris T", "Alan E", "Timothy B"}  # inactive Old Hand skipped
    assert rows["Chris T"]["Town"] == "Clinton" and rows["Chris T"]["State"] == "MO"
    assert rows["Chris T"]["Departments"] == ["FIELD", "TC"]
    assert rows["Chris T"]["TechId"] == "chtu0170"
    # No town is allowed: the region places this tech instead
    assert rows["Alan E"]["Town"] == "" and rows["Alan E"]["Region"] == "Jefferson City Area"
    assert rows["Timothy B"]["Region"] == ""


def test_the_short_roster_still_works(tmp_path):
    df = load_techs(_csv(tmp_path, "Tech,Town,State\nNate,Sedalia,MO\n"))
    row = df.to_dict("records")[0]
    assert (row["Tech"], row["Town"], row["Departments"], row["Region"]) == (
        "Nate",
        "Sedalia",
        [],
        "",
    )
