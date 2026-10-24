"""Tests unitaires du validateur M5 de RetailMind AI."""

import csv
from pathlib import Path

import pytest

from retailmind.data.m5_validator import (
    FILES,
    PRICE_COLUMNS,
    SALES_COLUMNS,
    M5ValidationError,
    validate_csv,
    validate_dataset,
)

CALENDAR_COLUMNS = ["date", "d", "wm_yr_wk"]
CALENDAR_ROW = ["2011-01-29", "d_1", "11101"]

PRICE_ROW = [
    "CA_1",
    "FOODS_1_001",
    "11101",
    "3.25",
]

SALES_ROW = [
    "FOODS_1_001_CA_1_validation",
    "FOODS_1_001",
    "FOODS_1",
    "FOODS",
    "CA_1",
    "CA",
] + ["0"] * 1913


def write_csv(
    path: Path,
    columns: list[str],
    rows: list[list[str]],
) -> None:
    """Creer un petit fichier CSV temporaire."""

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    """Creer les trois sources M5 minimales et valides."""

    write_csv(
        tmp_path / "calendar.csv",
        CALENDAR_COLUMNS,
        [CALENDAR_ROW],
    )

    write_csv(
        tmp_path / "sell_prices.csv",
        PRICE_COLUMNS,
        [PRICE_ROW],
    )

    write_csv(
        tmp_path / "sales_train_validation.csv",
        SALES_COLUMNS,
        [SALES_ROW],
    )

    return tmp_path


@pytest.mark.parametrize(
    ("filename", "expected_columns"),
    [
        ("calendar.csv", 3),
        ("sell_prices.csv", 4),
        ("sales_train_validation.csv", 1919),
    ],
)
def test_valid_csv(
    valid_dataset: Path,
    filename: str,
    expected_columns: int,
) -> None:
    result = validate_csv(valid_dataset / filename)

    assert result["status"] == "VALID"
    assert result["file"] == filename
    assert result["columns"] == expected_columns
    assert result["size_bytes"] > 0


def test_validate_complete_dataset(
    valid_dataset: Path,
) -> None:
    results = validate_dataset(valid_dataset)

    assert len(results) == 3
    assert [r["file"] for r in results] == list(FILES)
    assert all(r["status"] == "VALID" for r in results)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(
        M5ValidationError,
        match="Fichier introuvable",
    ):
        validate_csv(tmp_path / "calendar.csv")


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(
        M5ValidationError,
        match="Fichier vide",
    ):
        validate_csv(path)


def test_header_without_data(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"
    write_csv(path, CALENDAR_COLUMNS, [])

    with pytest.raises(
        M5ValidationError,
        match="Aucune ligne de donnees",
    ):
        validate_csv(path)


def test_incorrect_row_length(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"

    write_csv(
        path,
        CALENDAR_COLUMNS,
        [["2011-01-29"]],
    )

    with pytest.raises(
        M5ValidationError,
        match="Nombre de valeurs incorrect",
    ):
        validate_csv(path)


def test_duplicate_columns(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"

    write_csv(
        path,
        ["date", "d", "wm_yr_wk", "date"],
        [["2011-01-29", "d_1", "11101", "2011-01-29"]],
    )

    with pytest.raises(
        M5ValidationError,
        match="Colonnes dupliquees",
    ):
        validate_csv(path)


def test_missing_calendar_column(tmp_path: Path) -> None:
    path = tmp_path / "calendar.csv"

    write_csv(
        path,
        ["date", "d"],
        [["2011-01-29", "d_1"]],
    )

    with pytest.raises(
        M5ValidationError,
        match="Colonnes calendrier manquantes",
    ):
        validate_csv(path)


def test_invalid_prices_schema(tmp_path: Path) -> None:
    path = tmp_path / "sell_prices.csv"

    write_csv(
        path,
        ["item_id", "store_id", "wm_yr_wk", "sell_price"],
        [PRICE_ROW],
    )

    with pytest.raises(
        M5ValidationError,
        match="Schema sell_prices.csv incorrect",
    ):
        validate_csv(path)


def test_invalid_sales_schema(tmp_path: Path) -> None:
    path = tmp_path / "sales_train_validation.csv"

    wrong_columns = SALES_COLUMNS[:-1] + ["d_9999"]

    write_csv(
        path,
        wrong_columns,
        [SALES_ROW],
    )

    with pytest.raises(
        M5ValidationError,
        match="Schema sales_train_validation.csv incorrect",
    ):
        validate_csv(path)


def test_unsupported_file(tmp_path: Path) -> None:
    path = tmp_path / "unknown.csv"

    write_csv(
        path,
        ["column_a", "column_b"],
        [["1", "2"]],
    )

    with pytest.raises(
        M5ValidationError,
        match="Fichier non pris en charge",
    ):
        validate_csv(path)


def test_dataset_with_missing_source(
    valid_dataset: Path,
) -> None:
    (valid_dataset / "sell_prices.csv").unlink()

    with pytest.raises(
        M5ValidationError,
        match="Fichier introuvable",
    ):
        validate_dataset(valid_dataset)
