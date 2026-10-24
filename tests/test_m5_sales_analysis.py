"""Tests de l'analyse complete des ventes M5."""

import csv
from datetime import date, timedelta
from pathlib import Path

import pytest

from retailmind.data.m5_sales_analysis import (
    analyze_sales,
    export_analysis,
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
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def m5_dataset(tmp_path: Path) -> Path:
    first = date(2011, 1, 29)

    write_csv(
        tmp_path / "calendar.csv",
        ["date", "d", "wm_yr_wk"],
        [
            [
                (first + timedelta(days=i)).isoformat(),
                f"d_{i + 1}",
                "11101",
            ]
            for i in range(1913)
        ],
    )

    write_csv(
        tmp_path / "sell_prices.csv",
        PRICE_COLUMNS,
        [["CA_1", "ITEM_A", "11101", "2.50"]],
    )

    write_csv(
        tmp_path / "sales_train_validation.csv",
        SALES_COLUMNS,
        [
            [
                "a",
                "ITEM_A",
                "DEP_1",
                "FOODS",
                "CA_1",
                "CA",
                "2",
                "0",
                "1",
            ]
            + ["0"] * 1910,
            [
                "b",
                "ITEM_B",
                "DEP_2",
                "HOUSEHOLD",
                "TX_1",
                "TX",
                "3",
                "1",
                "0",
            ]
            + ["0"] * 1910,
        ],
    )

    return tmp_path


def test_full_sales_analysis(m5_dataset: Path) -> None:
    summary, daily, categories, stores, items = analyze_sales(
        m5_dataset,
        chunk_size=1,
    )

    assert summary["sales_series"] == 2
    assert summary["sales_days"] == 1913
    assert summary["total_units_sold"] == 7
    assert summary["zero_sales_cells"] == 2 * 1913 - 4
    assert summary["all_zero_series"] == 0

    assert daily["units_sold"].iloc[:3].tolist() == [5, 1, 1]
    assert daily["units_ma7"].iloc[1] == 3.0

    assert categories.set_index("category").loc["FOODS", "units_sold"] == 3
    assert stores.set_index("store_id").loc["TX_1", "units_sold"] == 4
    assert items.iloc[0]["item_id"] == "ITEM_B"


def test_export_analysis(m5_dataset: Path, tmp_path: Path) -> None:
    result = analyze_sales(m5_dataset, chunk_size=2)
    output = tmp_path / "analysis"

    export_analysis(output, result)

    filenames = [
        "sales_summary.json",
        "daily_sales.csv",
        "sales_by_category.csv",
        "sales_by_store.csv",
        "top_items.csv",
        "sales_trend.png",
        "sales_categories.png",
    ]

    for filename in filenames:
        assert (output / filename).is_file()


def test_invalid_chunk_size(m5_dataset: Path) -> None:
    with pytest.raises(ValueError, match="strictement positif"):
        analyze_sales(m5_dataset, chunk_size=0)


def test_missing_calendar_day(m5_dataset: Path) -> None:
    calendar_file = m5_dataset / "calendar.csv"

    with calendar_file.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.reader(stream))

    write_csv(calendar_file, rows[0], rows[1:-1])

    with pytest.raises(ValueError, match="date valide"):
        analyze_sales(m5_dataset, chunk_size=2)
