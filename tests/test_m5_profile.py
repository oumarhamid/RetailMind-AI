"""Tests unitaires du profilage initial M5."""

import csv
from pathlib import Path

import pytest

from retailmind.data.m5_profile import build_profile
from retailmind.data.m5_validator import (
    PRICE_COLUMNS,
    SALES_COLUMNS,
)


def write_csv(
    path: Path,
    header: list[str],
    rows: list[list[str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def m5_sample(tmp_path: Path) -> Path:
    write_csv(
        tmp_path / "calendar.csv",
        ["date", "d", "wm_yr_wk"],
        [
            ["2011-01-29", "d_1", "11101"],
            ["2011-01-30", "d_2", "11101"],
        ],
    )

    write_csv(
        tmp_path / "sell_prices.csv",
        PRICE_COLUMNS,
        [
            ["CA_1", "FOODS_1_001", "11101", "3.25"],
            ["CA_1", "FOODS_1_002", "11101", "0"],
        ],
    )

    write_csv(
        tmp_path / "sales_train_validation.csv",
        SALES_COLUMNS,
        [
            [
                "a",
                "FOODS_1_001",
                "FOODS_1",
                "FOODS",
                "CA_1",
                "CA",
                "2",
                "0",
            ]
            + ["0"] * 1911
        ],
    )

    return tmp_path


def test_build_profile(m5_sample: Path) -> None:
    report = build_profile(m5_sample)

    assert report["calendar"]["rows"] == 2
    assert report["calendar"]["first_date"] == "2011-01-29"
    assert report["sales_sample"]["daily_columns"] == 1913
    assert report["sales_sample"]["total_units_sample"] == 2
    assert report["prices_sample"]["nonpositive_prices"] == 1
    assert len(report["sources"]) == 3


def test_reject_nonpositive_sample_size(
    m5_sample: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="positives",
    ):
        build_profile(m5_sample, sales_rows=0)
