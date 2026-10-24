"""Profilage initial de M5 : calendrier complet et echantillons controles."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailmind.data.m5_validator import validate_dataset


def build_profile(
    data_dir: Path,
    sales_rows: int = 100,
    price_rows: int = 1000,
) -> dict:
    """Produire des indicateurs sans charger tous les CSV en memoire."""

    if sales_rows < 1 or price_rows < 1:
        raise ValueError("Les tailles des echantillons doivent etre positives")

    sources = validate_dataset(data_dir)

    calendar = pd.read_csv(data_dir / "calendar.csv")

    sales = pd.read_csv(
        data_dir / "sales_train_validation.csv",
        nrows=sales_rows,
    )

    prices = pd.read_csv(
        data_dir / "sell_prices.csv",
        nrows=price_rows,
    )

    day_columns = [column for column in sales.columns if column.startswith("d_")]

    if sales.empty or prices.empty or not day_columns:
        raise ValueError("Echantillons vides ou colonnes de ventes absentes")

    dates = pd.to_datetime(
        calendar["date"],
        errors="coerce",
    )

    daily_sales = sales[day_columns].to_numpy(dtype=np.float64)

    valid_sales = np.isfinite(daily_sales)

    numeric_prices = pd.to_numeric(
        prices["sell_price"],
        errors="coerce",
    )

    return {
        "method": {
            "calendar": "Toutes les lignes",
            "sales": "Premieres lignes uniquement (echantillon non aleatoire)",
            "prices": "Premieres lignes uniquement (echantillon non aleatoire)",
            "warning": ("Ne pas extrapoler les statistiques echantillonnees au dataset complet."),
        },
        "sources": sources,
        "calendar": {
            "rows": len(calendar),
            "first_date": (dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None),
            "last_date": (dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None),
            "invalid_dates": int(dates.isna().sum()),
            "duplicate_days": int(calendar["d"].duplicated().sum()),
        },
        "sales_sample": {
            "rows": len(sales),
            "daily_columns": len(day_columns),
            "cells_examined": int(daily_sales.size),
            "invalid_cells": int((~valid_sales).sum()),
            "negative_cells": int((daily_sales < 0).sum()),
            "zero_cells": int((daily_sales == 0).sum()),
            "total_units_sample": int(np.nansum(daily_sales)),
        },
        "prices_sample": {
            "rows": len(prices),
            "missing_or_invalid_prices": int(numeric_prices.isna().sum()),
            "nonpositive_prices": int((numeric_prices <= 0).sum()),
            "min_price": (float(numeric_prices.min()) if numeric_prices.notna().any() else None),
            "max_price": (float(numeric_prices.max()) if numeric_prices.notna().any() else None),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profilage initial du dataset M5")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/m5"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/m5/profile_sample.json"),
    )

    args = parser.parse_args()

    report = build_profile(args.data_dir)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Rapport cree : {args.output}")
    print("Attention : statistiques des ventes et prix calculees sur echantillons.")


if __name__ == "__main__":
    main()
