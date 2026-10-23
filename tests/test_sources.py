# Covers the spreadsheet reader itself: the two file formats, the column aliases
# the two exports disagree on, and what happens to towns carrying no region.

import pandas as pd
import pytest

from maybemap.config import UNASSIGNED_GROUP_LABEL
from maybemap.sources import cities


def _frame(tmp_path, rows, columns, *, name="input.csv", unassigned_label=None):
    path = tmp_path / name
    df = pd.DataFrame(rows, columns=columns)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False, engine="openpyxl")
    return cities._read_frame(path, unassigned_label=unassigned_label)


def test_the_two_formats_agree(spreadsheet, city_csv):
    # The CSV names its region column differently and shouts its place names, so
    # the loader has to land both on the same frame
    from_xlsx = cities._read_frame(spreadsheet, unassigned_label=None)
    from_csv = cities._read_frame(city_csv, unassigned_label=None)

    columns = ["City", "State", "Group", "Customers", "Full Location"]
    pd.testing.assert_frame_equal(
        from_xlsx[columns].reset_index(drop=True),
        from_csv[columns].reset_index(drop=True),
    )


@pytest.mark.parametrize("region_column", ["Region", "Region Name", "Group"])
def test_the_region_column_is_matched_under_any_of_its_names(tmp_path, region_column):
    df = _frame(
        tmp_path,
        [("Columbia", "MO", "Central")],
        ["City", "State", region_column],
    )
    assert df["Group"].tolist() == ["Central"]


def test_missing_columns_name_what_was_found(tmp_path):
    with pytest.raises(ValueError, match="Found: \\['City', 'State'\\]"):
        _frame(tmp_path, [("Columbia", "MO")], ["City", "State"])


def test_blank_regions_are_collected_under_the_label(tmp_path):
    df = _frame(
        tmp_path,
        [("Columbia", "MO", "Central"), ("Springfield", "MO", "")],
        ["City", "State", "Region"],
        unassigned_label=UNASSIGNED_GROUP_LABEL,
    )
    assert df["Group"].tolist() == ["Central", UNASSIGNED_GROUP_LABEL]


def test_blank_regions_are_dropped_when_there_is_no_label(tmp_path):
    df = _frame(
        tmp_path,
        [("Columbia", "MO", "Central"), ("Springfield", "MO", "")],
        ["City", "State", "Region"],
        unassigned_label=None,
    )
    assert df["City"].tolist() == ["Columbia"]


def test_names_and_states_are_normalized(tmp_path):
    # The CSV shouts, the xlsx spells states out; classify_city_importance and the
    # geocode cache each want exactly one of those forms
    df = _frame(
        tmp_path,
        [("SAINT PAUL", "SAINT CLAIR", "Missouri", "Central")],
        ["City", "County", "State", "Region"],
    )
    row = df.iloc[0]
    assert (row.City, row.County, row.State) == ("Saint Paul", "Saint Clair", "MO")
    assert row["Full Location"] == "Saint Paul, MO"


def test_customers_is_optional(tmp_path):
    with_counts = _frame(
        tmp_path,
        [("Columbia", "MO", "Central", "5")],
        ["City", "State", "Region", "Customers"],
    )
    assert with_counts["Customers"].tolist() == [5]

    without = _frame(tmp_path, [("Columbia", "MO", "Central")], ["City", "State", "Region"])
    assert without["Customers"].isna().all()
