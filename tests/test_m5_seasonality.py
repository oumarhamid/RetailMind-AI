"""Tests de saisonnalite et des controles temporels M5."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from retailmind.data.m5_seasonality import (
    analyze_seasonality,
    export_seasonality,
)


@pytest.fixture
def daily_csv(tmp_path: Path) -> Path:
    start = date(2011, 1, 29)
    rows = []

    for index in range(42):
        current = start + timedelta(days=index)
        rows.append(
            {
                "date": current.isoformat(),
                "d": f"d_{index + 1}",
                "units_sold": (200 if current.weekday() >= 5 else 100),
            }
        )

    path = tmp_path / "daily_sales.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def test_weekday_and_monthly_reconciliation(
    daily_csv: Path,
) -> None:
    summary, daily, weekdays, months, seasonal_months = analyze_seasonality(daily_csv)

    assert summary["days"] == 42
    assert summary["total_units_sold"] == 5400
    assert summary["calendar_continuity"] == "VALID"

    assert len(weekdays) == 7
    assert weekdays["days_observed"].tolist() == [6] * 7

    sunday_mean = weekdays.loc[
        weekdays["weekday_number"] == 6,
        "mean_units_per_day",
    ].item()

    assert sunday_mean == 200
    assert int(months["total_units"].sum()) == 5400
    assert int(seasonal_months["total_units"].sum()) == 5400

    assert pd.isna(daily["units_ma28"].iloc[26])
    assert pd.notna(daily["units_ma28"].iloc[27])


def test_exports(
    daily_csv: Path,
    tmp_path: Path,
) -> None:
    result = analyze_seasonality(daily_csv)
    directory = tmp_path / "reports"

    export_seasonality(directory, result)

    filenames = [
        "seasonality_summary.json",
        "daily_trend.csv",
        "sales_by_weekday.csv",
        "monthly_trend.csv",
        "sales_by_calendar_month.csv",
        "sales_trend_7_28.png",
        "weekday_seasonality.png",
        "monthly_trend.png",
    ]

    for filename in filenames:
        assert (directory / filename).is_file()


def test_missing_calendar_day(daily_csv: Path) -> None:
    frame = pd.read_csv(daily_csv)
    frame = frame.drop(index=10)
    frame.to_csv(daily_csv, index=False)

    with pytest.raises(
        ValueError,
        match="Jours manquants",
    ):
        analyze_seasonality(daily_csv)


def test_duplicate_date(daily_csv: Path) -> None:
    frame = pd.read_csv(daily_csv)
    frame.loc[5, "date"] = frame.loc[4, "date"]
    frame.to_csv(daily_csv, index=False)

    with pytest.raises(
        ValueError,
        match="Dates dupliquees",
    ):
        analyze_seasonality(daily_csv)


def test_negative_quantity(daily_csv: Path) -> None:
    frame = pd.read_csv(daily_csv)
    frame.loc[0, "units_sold"] = -1
    frame.to_csv(daily_csv, index=False)

    with pytest.raises(
        ValueError,
        match="negatives",
    ):
        analyze_seasonality(daily_csv)


def test_inconsistent_day_id(daily_csv: Path) -> None:
    frame = pd.read_csv(daily_csv)
    frame.loc[5, "d"] = "d_800"
    frame.to_csv(daily_csv, index=False)

    with pytest.raises(
        ValueError,
        match="Identifiants des jours",
    ):
        analyze_seasonality(daily_csv)
