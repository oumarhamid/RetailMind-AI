"""Tests du rapport EDA consolide."""

import csv
import json
from pathlib import Path

import pytest

from retailmind.data.m5_eda_report import build_eda_report


def write_json(root: Path, name: str, contents: dict) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents), encoding="utf-8")


def write_csv(root: Path, name: str, rows: list[dict]) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sample_reports(tmp_path: Path) -> Path:
    write_json(
        tmp_path,
        "inventory.json",
        {
            "sales": {
                "unique_items": 2,
                "unique_stores": 2,
                "unique_categories": 2,
            }
        },
    )
    write_json(
        tmp_path,
        "sales_analysis/sales_summary.json",
        {
            "total_units_sold": 10,
            "sales_series": 2,
            "sales_days": 7,
            "first_date": "2011-01-29",
            "last_date": "2011-02-04",
            "zero_sales_pct": 50,
        },
    )
    write_json(
        tmp_path,
        "seasonality/seasonality_summary.json",
        {
            "total_units_sold": 10,
            "calendar_continuity": "VALID",
        },
    )
    write_json(
        tmp_path,
        "price_calendar/price_calendar_summary.json",
        {
            "price_weeks": 1,
            "sales_weeks_without_any_price": [],
        },
    )
    write_json(
        tmp_path,
        "price_integrity/price_integrity_summary.json",
        {
            "units_sold_total": 10,
            "price_rows": 2,
            "sales_weeks": 1,
            "pair_weeks_without_price": 0,
            "price_duplicate_keys": 0,
            "units_sold_without_weekly_price": 0,
        },
    )
    write_json(
        tmp_path,
        "revenue/revenue_summary.json",
        {
            "total_units_sold": 10,
            "revenue_total": 32.5,
            "sales_weeks": 1,
            "units_without_price": 0,
        },
    )
    write_csv(
        tmp_path,
        "revenue/weekly_revenue.csv",
        [
            {
                "wm_yr_wk": "11101",
                "units_sold": "10",
                "revenue": "32.50",
            }
        ],
    )
    write_csv(
        tmp_path,
        "revenue/revenue_by_category.csv",
        [
            {
                "category": "FOODS",
                "units_sold": "6",
                "revenue": "18.00",
            },
            {
                "category": "HOBBIES",
                "units_sold": "4",
                "revenue": "14.50",
            },
        ],
    )
    write_csv(
        tmp_path,
        "revenue/revenue_by_store.csv",
        [
            {
                "store_id": "CA_1",
                "units_sold": "6",
                "revenue": "18.00",
            },
            {
                "store_id": "TX_1",
                "units_sold": "4",
                "revenue": "14.50",
            },
        ],
    )
    write_csv(
        tmp_path,
        "revenue/top_items_by_revenue.csv",
        [
            {
                "item_id": "ITEM_A",
                "revenue": "18.00",
            }
        ],
    )
    write_csv(
        tmp_path,
        "seasonality/sales_by_weekday.csv",
        [
            {
                "weekday_name": name,
                "mean_units_per_day": value,
            }
            for name, value in (
                ("Lundi", "1"),
                ("Mardi", "2"),
                ("Mercredi", "3"),
                ("Jeudi", "4"),
                ("Vendredi", "5"),
                ("Samedi", "10"),
                ("Dimanche", "6"),
            )
        ],
    )
    return tmp_path


def test_build_eda_report(sample_reports: Path) -> None:
    report = build_eda_report(sample_reports)

    assert "32,50" in report
    assert "Samedi" in report
    assert "ITEM_A" in report
    assert "Reconciliation des unites et revenus" in report


def test_inconsistent_units_rejected(
    sample_reports: Path,
) -> None:
    path = sample_reports / "revenue/revenue_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["total_units_sold"] = 11
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="revenus"):
        build_eda_report(sample_reports)


def test_inconsistent_revenue_rejected(
    sample_reports: Path,
) -> None:
    path = sample_reports / "revenue/revenue_by_category.csv"
    content = path.read_text(encoding="utf-8")
    path.write_text(
        content.replace("18.00", "25.00"),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="revenu ventes par categorie",
    ):
        build_eda_report(sample_reports)


def test_missing_report_rejected(
    sample_reports: Path,
) -> None:
    (sample_reports / "inventory.json").unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Rapport requis introuvable",
    ):
        build_eda_report(sample_reports)
