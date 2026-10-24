"""Tests des cles et de la couverture des prix M5."""

import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from retailmind.data.m5_price_integrity import (
    analyze_price_integrity,
    export_price_integrity,
)
from retailmind.data.m5_validator import (
    PRICE_COLUMNS,
    SALES_COLUMNS,
)


def write_csv(
    path: Path,
    columns: list[str],
    rows: list[list[str]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def m5_example(tmp_path: Path) -> Path:
    first = date(2011, 1, 29)

    write_csv(
        tmp_path / "calendar.csv",
        ["date", "d", "wm_yr_wk"],
        [
            [
                (first + timedelta(days=i)).isoformat(),
                f"d_{i + 1}",
                str(11101 + i // 7),
            ]
            for i in range(1913)
        ],
    )

    write_csv(
        tmp_path / "sales_train_validation.csv",
        SALES_COLUMNS,
        [
            [
                "A",
                "ITEM_A",
                "DEP",
                "FOODS",
                "CA_1",
                "CA",
                "2",
                *(["0"] * 6),
                "1",
            ]
            + ["0"] * 1905,
            [
                "B",
                "ITEM_B",
                "DEP",
                "FOODS",
                "CA_1",
                "CA",
                "3",
            ]
            + ["0"] * 1912,
        ],
    )

    write_csv(
        tmp_path / "sell_prices.csv",
        PRICE_COLUMNS,
        [
            ["CA_1", "ITEM_A", "11101", "2.5"],
            ["CA_1", "ITEM_A", "11101", "2.5"],
            ["CA_1", "ITEM_A", "11102", "2.6"],
            ["CA_1", "ITEM_B", "11102", "3"],
            ["CA_1", "ITEM_X", "11101", "1"],
            ["CA_1", "ITEM_A", "99999", "1"],
        ],
    )

    return tmp_path


def test_integrity_and_missing_prices_across_chunks(
    m5_example: Path,
) -> None:
    report, weekly = analyze_price_integrity(
        m5_example,
        price_chunk_size=1,
        sales_chunk_size=1,
    )

    assert report["price_rows"] == 6
    assert report["price_duplicate_keys"] == 1
    assert report["price_rows_outside_calendar"] == 1
    assert report["price_pairs_absent_from_sales"] == 1
    assert report["sales_weeks"] == 274
    assert report["possible_sales_pair_weeks"] == 548
    assert report["pair_weeks_with_price"] == 3
    assert report["positive_sales_pair_weeks"] == 3

    assert report["positive_sales_pair_weeks_without_price"] == 1

    assert report["units_sold_without_weekly_price"] == 3
    assert report["units_sold_total"] == 6
    assert int(weekly["pairs_with_price"].sum()) == 3


def test_export_price_integrity(
    m5_example: Path,
    tmp_path: Path,
) -> None:
    result = analyze_price_integrity(
        m5_example,
        price_chunk_size=2,
        sales_chunk_size=2,
    )

    destination = tmp_path / "output"

    export_price_integrity(destination, result)

    assert (destination / "price_integrity_summary.json").is_file()

    assert len(pd.read_csv(destination / "weekly_price_coverage.csv")) == 274


@pytest.mark.parametrize(
    ("prices", "sales"),
    [(0, 1), (1, 0)],
)
def test_reject_invalid_chunk_size(
    m5_example: Path,
    prices: int,
    sales: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="strictement positives",
    ):
        analyze_price_integrity(
            m5_example,
            price_chunk_size=prices,
            sales_chunk_size=sales,
        )


def test_reject_negative_sales(
    m5_example: Path,
) -> None:
    path = m5_example / "sales_train_validation.csv"

    frame = pd.read_csv(path)
    frame.loc[0, "d_1"] = -1
    frame.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="Quantites vendues invalides",
    ):
        analyze_price_integrity(
            m5_example,
            price_chunk_size=2,
            sales_chunk_size=2,
        )
