"""Tests de couverture des semaines commerciales M5."""

import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from retailmind.data.m5_price_calendar import (
    analyze_price_calendar,
    export_price_calendar,
)


def write_csv(
    path: Path,
    header: list[str],
    rows: list[list[str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def m5_sample(tmp_path: Path) -> Path:
    start = date(2011, 1, 29)

    write_csv(
        tmp_path / "calendar.csv",
        ["date", "d", "wm_yr_wk"],
        [
            [
                (start + timedelta(days=i)).isoformat(),
                f"d_{i + 1}",
                str(11101 + i // 7),
            ]
            for i in range(1913)
        ],
    )

    write_csv(
        tmp_path / "sales_train_validation.csv",
        [
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ]
        + [f"d_{i}" for i in range(1, 1914)],
        [
            [
                "one",
                "ITEM_1",
                "DEP",
                "FOODS",
                "CA_1",
                "CA",
            ]
            + ["1"] * 1913
        ],
    )

    write_csv(
        tmp_path / "sell_prices.csv",
        ["store_id", "item_id", "wm_yr_wk", "sell_price"],
        [
            ["CA_1", "ITEM_1", "11101", "2.50"],
            ["CA_1", "ITEM_1", "11102", "0"],
            ["CA_1", "ITEM_1", "99999", "3.00"],
        ],
    )

    return tmp_path


def test_full_price_week_coverage(m5_sample: Path) -> None:
    report, weeks = analyze_price_calendar(m5_sample, chunk_size=2)

    assert report["sales_days"] == 1913
    assert report["calendar_weeks"] == 274
    assert report["price_rows"] == 3
    assert report["price_weeks"] == 3
    assert report["nonpositive_prices"] == 1
    assert report["price_rows_outside_calendar"] == 1
    assert report["price_weeks_outside_calendar"] == ["99999"]
    assert len(report["sales_weeks_without_any_price"]) == 272
    assert weeks["price_rows"].sum() == 3


def test_export(m5_sample: Path, tmp_path: Path) -> None:
    result = analyze_price_calendar(m5_sample, chunk_size=1)

    output = tmp_path / "report"

    export_price_calendar(output, result)

    assert (output / "price_calendar_summary.json").is_file()

    assert (output / "price_rows_by_week.csv").is_file()

    assert len(pd.read_csv(output / "price_rows_by_week.csv")) == 275


def test_invalid_chunk_size(m5_sample: Path) -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        analyze_price_calendar(m5_sample, chunk_size=0)


def test_missing_sales_day_in_calendar(
    m5_sample: Path,
) -> None:
    calendar = pd.read_csv(m5_sample / "calendar.csv")

    calendar = calendar.iloc[:-1]

    calendar.to_csv(m5_sample / "calendar.csv", index=False)

    with pytest.raises(ValueError, match="Jours de vente absents"):
        analyze_price_calendar(m5_sample, chunk_size=2)
