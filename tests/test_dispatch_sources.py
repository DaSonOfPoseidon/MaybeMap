# The hand-kept dispatch roster CSVs.

import pytest

from maybemap.sources.dispatch import load_staging, load_techs


def _csv(tmp_path, text, name="roster.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_techs_normalize_to_the_city_export_form(tmp_path):
    df = load_techs(_csv(tmp_path, "tech,town,state\n Robin ,SEDALIA,Missouri\n"))
    row = df.to_dict("records")[0]
    assert (row["Tech"], row["Town"], row["State"]) == ("Robin", "Sedalia", "MO")


def test_blank_rows_are_skipped(tmp_path):
    df = load_techs(_csv(tmp_path, "Tech,Town,State\nRobin,Sedalia,MO\n,,\n,Clinton,MO\n"))
    assert df["Tech"].tolist() == ["Robin"]


def test_a_missing_column_names_what_was_found(tmp_path):
    with pytest.raises(ValueError, match="Found: \\['Tech', 'Town'\\]"):
        load_techs(_csv(tmp_path, "Tech,Town\nRobin,Sedalia\n"))


def test_missing_files_mean_empty_rosters(tmp_path):
    assert load_techs(tmp_path / "nope.csv").empty
    assert load_staging(tmp_path / "nope.csv").empty


def test_staging_keeps_file_order(tmp_path):
    df = load_staging(_csv(tmp_path, "Type,Town,State\nOutage,Nevada,MO\nGenerator,Rich Hill,MO\n"))
    assert df["Type"].tolist() == ["Outage", "Generator"]


TECH_INFO = (
    "Tech,Town,State,Active,Tech ID,In Cities List,Primary Region,Departments\n"
    'Pat Q,CLINTON,MO,Y,zzpq0001,Y,Clinton-Butler,"FIELD, TC"\n'
    "Sam R,,,Y,zzsr0001,N,Jefferson City Area,FIELD\n"
    "Old Hand,SEDALIA,MO,N,zzoh0001,Y,Sedalia-La Monte,FIELD\n"
    "Lee V,,,Y,zzlv0001,N,,FIELDBUS\n"
)


def test_the_mbs_tech_export_is_read_in_full(tmp_path):
    df = load_techs(_csv(tmp_path, TECH_INFO))
    rows = {r["Tech"]: r for r in df.to_dict("records")}
    assert set(rows) == {"Pat Q", "Sam R", "Lee V"}  # inactive Old Hand skipped
    assert rows["Pat Q"]["Town"] == "Clinton" and rows["Pat Q"]["State"] == "MO"
    assert rows["Pat Q"]["Departments"] == ["FIELD", "TC"]
    assert rows["Pat Q"]["TechId"] == "zzpq0001"
    # No town is allowed: the region places this tech instead
    assert rows["Sam R"]["Town"] == "" and rows["Sam R"]["Region"] == "Jefferson City Area"
    assert rows["Lee V"]["Region"] == ""


def test_the_short_roster_still_works(tmp_path):
    df = load_techs(_csv(tmp_path, "Tech,Town,State\nRobin,Sedalia,MO\n"))
    row = df.to_dict("records")[0]
    assert (row["Tech"], row["Town"], row["Departments"], row["Region"]) == (
        "Robin",
        "Sedalia",
        [],
        "",
    )


@pytest.mark.parametrize(
    "load,header,row,who",
    [
        (load_techs, "Tech,Town,State", "Pat Q,Clinton,", "Pat Q"),
        (load_staging, "Type,Town,State", "Outage,Nevada,", "Outage"),
    ],
    ids=["tech", "staging"],
)
def test_a_town_with_no_state_is_named_not_dropped(tmp_path, load, header, row, who):
    # Otherwise the town is silently ignored: a tech lands at their region hub, or
    # is listed as having no town at all
    with pytest.raises(ValueError, match=f"{who}.*Clinton|{who}.*Nevada"):
        load(_csv(tmp_path, f"{header}\n{row}\n"))


def test_an_inactive_tech_with_no_state_is_still_just_skipped(tmp_path):
    df = load_techs(
        _csv(tmp_path, "Tech,Town,State,Active\nPat Q,Clinton,,N\nSam R,Sedalia,MO,Y\n")
    )
    assert df["Tech"].tolist() == ["Sam R"]
