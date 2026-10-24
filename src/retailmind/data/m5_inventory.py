"""Inventaire complet des identifiants M5 et des relations ventes-prix."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailmind.data.m5_validator import validate_dataset

SALES_IDENTIFIERS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
]


def build_inventory(data_dir: Path, price_chunk_size: int = 100_000) -> dict:
    """Explorer les identifiants et tous les prix par lots."""

    if price_chunk_size < 1:
        raise ValueError("price_chunk_size doit etre strictement positif")

    validate_dataset(data_dir)

    sales_path = data_dir / "sales_train_validation.csv"
    prices_path = data_dir / "sell_prices.csv"

    # Les colonnes quotidiennes ne sont pas chargees en memoire.
    sales = pd.read_csv(
        sales_path,
        usecols=SALES_IDENTIFIERS,
        dtype="string",
    )

    sales_header = pd.read_csv(sales_path, nrows=0).columns
    sales_days = {name for name in sales_header if name.startswith("d_")}

    calendar = pd.read_csv(
        data_dir / "calendar.csv",
        usecols=["d", "wm_yr_wk"],
    )

    calendar_days = set(calendar["d"].dropna().astype(str))

    sales_pairs = set(sales[["store_id", "item_id"]].dropna().itertuples(index=False, name=None))

    price_pairs: set[tuple[str, str]] = set()
    price_weeks: set[int] = set()

    price_rows = 0
    missing_price_keys = 0
    invalid_price_rows = 0
    nonpositive_prices = 0
    chunks_processed = 0

    min_positive_price: float | None = None
    max_positive_price: float | None = None

    # Toutes les lignes de prix sont parcourues par lots.
    for chunk in pd.read_csv(
        prices_path,
        chunksize=price_chunk_size,
    ):
        chunks_processed += 1
        price_rows += len(chunk)

        missing_price_keys += int(
            chunk[["store_id", "item_id", "wm_yr_wk"]].isna().any(axis=1).sum()
        )

        price_pairs.update(
            chunk[["store_id", "item_id"]]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        weeks = pd.to_numeric(
            chunk["wm_yr_wk"],
            errors="coerce",
        )

        price_weeks.update(int(week) for week in weeks.dropna().unique())

        amounts = pd.to_numeric(
            chunk["sell_price"],
            errors="coerce",
        ).to_numpy(dtype="float64", na_value=np.nan)

        finite = np.isfinite(amounts)
        positive = amounts[finite & (amounts > 0)]

        invalid_price_rows += int((~finite).sum())

        nonpositive_prices += int((finite & (amounts <= 0)).sum())

        if positive.size:
            chunk_min = float(positive.min())
            chunk_max = float(positive.max())

            min_positive_price = (
                chunk_min if min_positive_price is None else min(min_positive_price, chunk_min)
            )

            max_positive_price = (
                chunk_max if max_positive_price is None else max(max_positive_price, chunk_max)
            )

    without_price = sales_pairs - price_pairs
    outside_sales = price_pairs - sales_pairs

    return {
        "method": {
            "sales": ("Toutes les series : lecture des 6 colonnes d'identification"),
            "prices": "Toutes les lignes : lecture par lots",
            "scope": ("Inventaire et couverture des paires produit-magasin, pas des prix par jour"),
        },
        "calendar": {
            "days": len(calendar),
            "sales_day_columns": len(sales_days),
            "sales_days_missing_in_calendar": len(sales_days - calendar_days),
        },
        "sales": {
            "series": len(sales),
            "unique_items": int(sales["item_id"].nunique()),
            "unique_departments": int(sales["dept_id"].nunique()),
            "unique_categories": int(sales["cat_id"].nunique()),
            "unique_stores": int(sales["store_id"].nunique()),
            "unique_states": int(sales["state_id"].nunique()),
            "duplicate_series_ids": int(sales["id"].duplicated().sum()),
            "duplicate_item_store_pairs": int(sales.duplicated(["item_id", "store_id"]).sum()),
            "missing_identifiers": int(sales[SALES_IDENTIFIERS].isna().sum().sum()),
            "series_by_category": {
                str(name): int(value)
                for name, value in (sales["cat_id"].value_counts().sort_index().items())
            },
            "series_by_state": {
                str(name): int(value)
                for name, value in (sales["state_id"].value_counts().sort_index().items())
            },
        },
        "prices": {
            "rows": price_rows,
            "chunks_processed": chunks_processed,
            "unique_item_store_pairs": len(price_pairs),
            "unique_weeks": len(price_weeks),
            "rows_missing_identifiers": missing_price_keys,
            "missing_or_invalid_amounts": invalid_price_rows,
            "nonpositive_amounts": nonpositive_prices,
            "min_positive_price": min_positive_price,
            "max_positive_price": max_positive_price,
        },
        "relationships": {
            "sales_pairs_without_any_price": len(without_price),
            "price_pairs_absent_from_sales": len(outside_sales),
            "sales_pairs_without_price_examples": [
                list(pair) for pair in sorted(without_price)[:5]
            ],
            "unknown_price_pair_examples": [list(pair) for pair in sorted(outside_sales)[:5]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventaire complet M5")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/m5"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/m5/inventory.json"),
    )

    parser.add_argument(
        "--price-chunk-size",
        type=int,
        default=100_000,
    )

    args = parser.parse_args()

    report = build_inventory(
        args.data_dir,
        price_chunk_size=args.price_chunk_size,
    )

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

    print(f"Inventaire cree : {args.output}")
    print(f"Series de ventes : {report['sales']['series']}")
    print(f"Lignes de prix analysees : {report['prices']['rows']}")


if __name__ == "__main__":
    main()
