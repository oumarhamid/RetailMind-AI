"""Calcul du chiffre d'affaires hebdomadaire M5 avec lecture par lots."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailmind.data.m5_validator import validate_dataset


def analyze_revenue(
    data_dir: Path,
    sales_chunk_size: int = 128,
    price_chunk_size: int = 100_000,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Joindre ventes et prix par couple produit-magasin et semaine."""
    if sales_chunk_size < 1 or price_chunk_size < 1:
        raise ValueError("Les tailles des lots doivent etre strictement positives")

    validate_dataset(data_dir)
    sales_path = data_dir / "sales_train_validation.csv"
    prices_path = data_dir / "sell_prices.csv"

    calendar = pd.read_csv(
        data_dir / "calendar.csv",
        usecols=["d", "wm_yr_wk"],
        dtype="string",
    )
    if calendar[["d", "wm_yr_wk"]].isna().any().any() or calendar["d"].duplicated().any():
        raise ValueError("Calendrier invalide")

    day_columns = [
        name for name in pd.read_csv(sales_path, nrows=0).columns if name.startswith("d_")
    ]
    weeks_by_day = calendar.set_index("d")["wm_yr_wk"].reindex(day_columns)
    if weeks_by_day.isna().any():
        raise ValueError("Jours de ventes sans semaine commerciale")

    day_weeks = weeks_by_day.astype(str).tolist()
    first_day_indices = [
        i for i, week in enumerate(day_weeks) if i == 0 or week != day_weeks[i - 1]
    ]
    sales_weeks = [day_weeks[i] for i in first_day_indices]
    if len(set(sales_weeks)) != len(sales_weeks):
        raise ValueError("Semaines de ventes non contigues")
    week_index = {week: i for i, week in enumerate(sales_weeks)}
    calendar_weeks = set(calendar["wm_yr_wk"].astype(str))

    sales_ids = pd.read_csv(
        sales_path,
        usecols=["store_id", "item_id", "cat_id"],
        dtype="string",
    )
    if sales_ids.empty or sales_ids.isna().any().any():
        raise ValueError("Identifiants de ventes manquants")
    if sales_ids.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Couples produit-magasin dupliques dans les ventes")

    sales_pairs = pd.MultiIndex.from_frame(sales_ids[["store_id", "item_id"]])
    series_count = len(sales_ids)
    week_count = len(sales_weeks)
    weekly_units = np.empty((series_count, week_count), dtype=np.int64)
    position = 0

    for chunk in pd.read_csv(
        sales_path,
        usecols=day_columns,
        dtype="float64",
        chunksize=sales_chunk_size,
    ):
        values = chunk[day_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Quantites de ventes invalides")
        if (values != np.floor(values)).any():
            raise ValueError("Quantites de ventes non entieres")
        size = len(chunk)
        weekly_units[position : position + size] = np.add.reduceat(
            values.astype(np.int64), first_day_indices, axis=1
        )
        position += size

    if position != series_count:
        raise ValueError("Nombre de series incoherent")

    category_names = sorted(sales_ids["cat_id"].unique().tolist())
    store_names = sorted(sales_ids["store_id"].unique().tolist())
    item_names = sorted(sales_ids["item_id"].unique().tolist())
    category_codes = pd.Index(category_names).get_indexer(sales_ids["cat_id"])
    store_codes = pd.Index(store_names).get_indexer(sales_ids["store_id"])
    item_codes = pd.Index(item_names).get_indexer(sales_ids["item_id"])

    covered = np.zeros(weekly_units.shape, dtype=bool)
    weekly_revenue = np.zeros(week_count, dtype=np.float64)
    category_revenue = np.zeros(len(category_names), dtype=np.float64)
    store_revenue = np.zeros(len(store_names), dtype=np.float64)
    item_revenue = np.zeros(len(item_names), dtype=np.float64)
    rows_read = 0
    future_price_rows = 0

    for chunk in pd.read_csv(
        prices_path,
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "string"},
        chunksize=price_chunk_size,
    ):
        rows_read += len(chunk)
        if chunk[["store_id", "item_id", "wm_yr_wk"]].isna().any().any():
            raise ValueError("Cles de prix manquantes")
        prices = pd.to_numeric(chunk["sell_price"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Prix manquants, negatifs ou nuls")
        if not chunk["wm_yr_wk"].isin(calendar_weeks).all():
            raise ValueError("Prix avec semaine absente du calendrier")

        pair_indices = sales_pairs.get_indexer(
            pd.MultiIndex.from_frame(chunk[["store_id", "item_id"]])
        )
        if (pair_indices < 0).any():
            raise ValueError("Prix pour un produit-magasin absent des ventes")
        price_week_indices = chunk["wm_yr_wk"].map(week_index).fillna(-1).to_numpy(dtype=np.int64)
        historical = price_week_indices >= 0
        future_price_rows += int((~historical).sum())

        row_indices = pair_indices[historical]
        column_indices = price_week_indices[historical]
        if len(row_indices) == 0:
            continue
        keys = pd.MultiIndex.from_arrays([row_indices, column_indices])
        if keys.has_duplicates or covered[row_indices, column_indices].any():
            raise ValueError("Cles de prix hebdomadaires dupliquees")
        covered[row_indices, column_indices] = True

        amount = weekly_units[row_indices, column_indices] * prices[historical]
        weekly_revenue += np.bincount(column_indices, weights=amount, minlength=week_count)
        category_revenue += np.bincount(
            category_codes[row_indices], weights=amount, minlength=len(category_names)
        )
        store_revenue += np.bincount(
            store_codes[row_indices], weights=amount, minlength=len(store_names)
        )
        item_revenue += np.bincount(
            item_codes[row_indices], weights=amount, minlength=len(item_names)
        )

    unpriced = (weekly_units > 0) & (~covered)
    unpriced_units = int(weekly_units[unpriced].sum())
    if unpriced_units:
        raise ValueError(f"Ventes positives sans prix hebdomadaire : {unpriced_units} unites")

    units_by_series = weekly_units.sum(axis=1)
    total_units = int(units_by_series.sum())
    total_revenue = float(weekly_revenue.sum())
    if any(
        not np.isclose(total_revenue, float(values.sum()), atol=0.01, rtol=0)
        for values in (category_revenue, store_revenue, item_revenue)
    ):
        raise ValueError("Montants non reconcilies entre les agregations")

    weekly = pd.DataFrame(
        {
            "wm_yr_wk": sales_weeks,
            "days_observed": np.diff([*first_day_indices, len(day_columns)]),
            "units_sold": weekly_units.sum(axis=0),
            "revenue": np.round(weekly_revenue, 2),
        }
    )
    categories = pd.DataFrame(
        {
            "category": category_names,
            "units_sold": np.bincount(
                category_codes, weights=units_by_series, minlength=len(category_names)
            ).astype(np.int64),
            "revenue": np.round(category_revenue, 2),
        }
    ).sort_values("revenue", ascending=False)
    stores = pd.DataFrame(
        {
            "store_id": store_names,
            "units_sold": np.bincount(
                store_codes, weights=units_by_series, minlength=len(store_names)
            ).astype(np.int64),
            "revenue": np.round(store_revenue, 2),
        }
    ).sort_values("revenue", ascending=False)
    items = (
        pd.DataFrame({"item_id": item_names, "revenue": np.round(item_revenue, 2)})
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    report = {
        "scope": "Historique M5 complet ; prix hebdomadaires par produit et magasin",
        "sales_series": series_count,
        "sales_weeks": week_count,
        "price_rows_checked": rows_read,
        "price_rows_future_weeks_excluded": future_price_rows,
        "total_units_sold": total_units,
        "units_without_price": unpriced_units,
        "revenue_total": round(total_revenue, 2),
        "revenue_definition": "Somme des quantites vendues multipliees par les prix hebdomadaires",
        "revenue_limitations": (
            "Estimation brute aux prix listes : remises transactionnelles, "
            "taxes et retours non disponibles dans les fichiers M5."
        ),
    }
    return report, weekly, categories, stores, items


def export_revenue(
    output_dir: Path,
    result: tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Exporter les resultats sans versionner les fichiers de donnees."""
    report, weekly, categories, stores, items = result
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "revenue_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    weekly.to_csv(output_dir / "weekly_revenue.csv", index=False)
    categories.to_csv(output_dir / "revenue_by_category.csv", index=False)
    stores.to_csv(output_dir / "revenue_by_store.csv", index=False)
    items.to_csv(output_dir / "top_items_by_revenue.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chiffre d'affaires historique M5")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/m5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/m5/revenue"))
    parser.add_argument("--sales-chunk-size", type=int, default=128)
    parser.add_argument("--price-chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    result = analyze_revenue(args.data_dir, args.sales_chunk_size, args.price_chunk_size)
    export_revenue(args.output_dir, result)
    print(f"Unites : {result[0]['total_units_sold']}")
    print(f"Chiffre d'affaires estime : {result[0]['revenue_total']:.2f}")
    print(f"Rapports : {args.output_dir}")


if __name__ == "__main__":
    main()
