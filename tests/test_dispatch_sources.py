# The hand-kept dispatch roster CSVs.

import pytest

from maybemap.sources.dispatch import load_staging, load_techs


def _csv(tmp_path, text, name="roster.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_techs_normalize_to_the_city_export_form(tmp_path):
    df = load_techs(_csv(tmp_path, "tech,town,state\n Nate ,SEDALIA,Missouri\n"))
    assert df.to_dict("records") == [{"Tech": "Nate", "Town": "Sedalia", "State": "MO"}]


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
