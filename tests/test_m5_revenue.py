"""Tests du calcul du chiffre d'affaires M5."""

import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from retailmind.data.m5_revenue import analyze_revenue, export_revenue


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
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
                (start + timedelta(days=day - 1)).isoformat(),
                f"d_{day}",
                str(11101 + (day - 1) // 7),
            ]
            for day in range(1, 1970)
        ],
    )
    base = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    days = [f"d_{i}" for i in range(1, 1914)]
    a = ["A", "ITEM_A", "D1", "FOODS", "CA_1", "CA"] + [
        str({1: 2, 2: 1, 8: 3}.get(i, 0)) for i in range(1, 1914)
    ]
    b = ["B", "ITEM_B", "D2", "HOUSEHOLD", "TX_1", "TX"] + [
        str({1: 4, 9: 5}.get(i, 0)) for i in range(1, 1914)
    ]
    write_csv(tmp_path / "sales_train_validation.csv", base + days, [a, b])
    write_csv(
        tmp_path / "sell_prices.csv",
        ["store_id", "item_id", "wm_yr_wk", "sell_price"],
        [
            ["CA_1", "ITEM_A", "11101", "2.5"],
            ["CA_1", "ITEM_A", "11102", "3.0"],
            ["TX_1", "ITEM_B", "11101", "4.0"],
            ["TX_1", "ITEM_B", "11102", "5.0"],
            ["CA_1", "ITEM_A", "11382", "7.4"],
        ],
    )
    return tmp_path


def test_revenue_reconciles_across_chunks(m5_sample: Path) -> None:
    report, weekly, categories, stores, items = analyze_revenue(
        m5_sample, sales_chunk_size=1, price_chunk_size=2
    )
    assert report["total_units_sold"] == 15
    assert report["units_without_price"] == 0
    assert report["revenue_total"] == 57.5
    assert report["price_rows_checked"] == 5
    assert report["price_rows_future_weeks_excluded"] == 1
    assert report["sales_weeks"] == 274
    assert weekly["units_sold"].iloc[:2].tolist() == [7, 8]
    assert weekly["revenue"].iloc[:2].tolist() == [23.5, 34.0]
    assert categories.set_index("category").loc["FOODS", "revenue"] == 16.5
    assert stores.set_index("store_id").loc["TX_1", "revenue"] == 41
    assert items.iloc[0]["item_id"] == "ITEM_B"


def test_export_revenue(m5_sample: Path, tmp_path: Path) -> None:
    result = analyze_revenue(m5_sample, sales_chunk_size=2, price_chunk_size=2)
    output = tmp_path / "output"
    export_revenue(output, result)
    assert len(pd.read_csv(output / "weekly_revenue.csv")) == 274
    for name in (
        "revenue_summary.json",
        "revenue_by_category.csv",
        "revenue_by_store.csv",
        "top_items_by_revenue.csv",
    ):
        assert (output / name).is_file()


def test_missing_price_for_positive_sales(m5_sample: Path) -> None:
    path = m5_sample / "sell_prices.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        lines = list(csv.reader(stream))

    filtered = [row for row in lines[1:] if not (row[0] == "TX_1" and row[2] == "11102")]
    write_csv(path, lines[0], filtered)

    with pytest.raises(ValueError, match="Ventes positives sans prix"):
        analyze_revenue(m5_sample, sales_chunk_size=2, price_chunk_size=1)


def test_duplicate_price_key(m5_sample: Path) -> None:
    path = m5_sample / "sell_prices.csv"
    with path.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(["CA_1", "ITEM_A", "11101", "2.5"])

    with pytest.raises(ValueError, match="dupliquees"):
        analyze_revenue(m5_sample, sales_chunk_size=2, price_chunk_size=2)


@pytest.mark.parametrize(
    ("sales_size", "price_size"),
    [(0, 1), (1, 0)],
)
def test_invalid_chunk_size(
    m5_sample: Path,
    sales_size: int,
    price_size: int,
) -> None:
    with pytest.raises(ValueError, match="strictement positives"):
        analyze_revenue(
            m5_sample,
            sales_chunk_size=sales_size,
            price_chunk_size=price_size,
        )
