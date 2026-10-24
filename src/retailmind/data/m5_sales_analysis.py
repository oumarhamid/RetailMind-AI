"""Analyse exhaustive des ventes quotidiennes M5, par lecture par lots."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from retailmind.data.m5_validator import validate_dataset

matplotlib.use("Agg")


def _group_report(totals: dict, key: str) -> pd.DataFrame:
    """Transformer les agregats par groupe en tableau trie."""
    records = [
        {key: name, "series": count, "units_sold": int(units)}
        for name, (count, units) in totals.items()
    ]
    return pd.DataFrame(records).sort_values("units_sold", ascending=False).reset_index(drop=True)


def analyze_sales(data_dir: Path, chunk_size: int = 512) -> tuple:
    """Calculer les ventes sur toutes les series, sans charger le CSV entier."""
    if chunk_size < 1:
        raise ValueError("chunk_size doit etre strictement positif")

    validate_dataset(data_dir)
    sales_path = data_dir / "sales_train_validation.csv"
    header = pd.read_csv(sales_path, nrows=0).columns
    day_columns = [name for name in header if name.startswith("d_")]

    calendar = pd.read_csv(data_dir / "calendar.csv", usecols=["date", "d"])
    if calendar["d"].duplicated().any():
        raise ValueError("Le calendrier contient des jours dupliques")

    dates = pd.to_datetime(
        calendar.set_index("d").reindex(day_columns)["date"],
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError("Certains jours de ventes n'ont pas de date valide")

    daily_units = np.zeros(len(day_columns), dtype=np.float64)
    daily_zeros = np.zeros(len(day_columns), dtype=np.int64)
    series_count = 0
    all_zero_series = 0

    by_category = defaultdict(lambda: [0, 0.0])
    by_store = defaultdict(lambda: [0, 0.0])
    by_item = defaultdict(float)

    columns = ["item_id", "cat_id", "store_id", *day_columns]
    dtypes = {day: "float64" for day in day_columns}
    dtypes.update({key: "string" for key in ["item_id", "cat_id", "store_id"]})

    for chunk in pd.read_csv(
        sales_path,
        usecols=columns,
        dtype=dtypes,
        chunksize=chunk_size,
    ):
        if chunk[["item_id", "cat_id", "store_id"]].isna().any().any():
            raise ValueError("Identifiant manquant dans les ventes")

        values = chunk[day_columns].to_numpy(
            dtype=np.float64,
            copy=False,
        )

        if not np.isfinite(values).all():
            raise ValueError("Vente manquante ou non numerique detectee")

        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Les quantites vendues doivent etre entieres et positives ou nulles")

        series_count += len(chunk)
        daily_units += values.sum(axis=0, dtype=np.float64)
        daily_zeros += (values == 0).sum(axis=0)
        all_zero_series += int((values == 0).all(axis=1).sum())

        row_units = values.sum(axis=1, dtype=np.float64)

        groups = pd.DataFrame(
            {
                "item_id": chunk["item_id"].to_numpy(),
                "cat_id": chunk["cat_id"].to_numpy(),
                "store_id": chunk["store_id"].to_numpy(),
                "units_sold": row_units,
            }
        )

        for category, group in groups.groupby("cat_id"):
            by_category[str(category)][0] += len(group)
            by_category[str(category)][1] += float(group["units_sold"].sum())

        for store, group in groups.groupby("store_id"):
            by_store[str(store)][0] += len(group)
            by_store[str(store)][1] += float(group["units_sold"].sum())

        for item, value in groups.groupby("item_id")["units_sold"].sum().items():
            by_item[str(item)] += float(value)

    if series_count == 0:
        raise ValueError("Aucune serie de ventes analysee")

    daily = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d").to_numpy(),
            "d": day_columns,
            "units_sold": daily_units.astype(np.int64),
            "zero_sales_series": daily_zeros,
            "zero_sales_pct": (100 * daily_zeros / series_count).round(2),
        }
    )

    daily["units_ma7"] = daily["units_sold"].rolling(7, min_periods=1).mean().round(2)

    by_category_df = _group_report(by_category, "category")
    by_store_df = _group_report(by_store, "store_id")

    top_items = (
        pd.DataFrame(
            [{"item_id": item, "units_sold": int(units)} for item, units in by_item.items()]
        )
        .sort_values("units_sold", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    total_cells = series_count * len(day_columns)

    summary = {
        "scope": ("Toutes les series et tous les jours de sales_train_validation.csv"),
        "sales_series": series_count,
        "sales_days": len(day_columns),
        "first_date": daily["date"].iloc[0],
        "last_date": daily["date"].iloc[-1],
        "cells_examined": total_cells,
        "zero_sales_cells": int(daily_zeros.sum()),
        "zero_sales_pct": round(100 * int(daily_zeros.sum()) / total_cells, 2),
        "all_zero_series": all_zero_series,
        "total_units_sold": int(daily_units.sum()),
        "lowest_daily_units": int(daily_units.min()),
        "highest_daily_units": int(daily_units.max()),
        "invalid_or_negative_sales": 0,
    }

    return summary, daily, by_category_df, by_store_df, top_items


def export_analysis(output_dir: Path, result: tuple) -> None:
    """Enregistrer les indicateurs et deux figures lisibles."""
    import matplotlib.pyplot as plt

    summary, daily, categories, stores, top_items = result
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "sales_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    daily.to_csv(output_dir / "daily_sales.csv", index=False)
    categories.to_csv(output_dir / "sales_by_category.csv", index=False)
    stores.to_csv(output_dir / "sales_by_store.csv", index=False)
    top_items.to_csv(output_dir / "top_items.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    timeline = pd.to_datetime(daily["date"])

    ax.plot(
        timeline,
        daily["units_sold"],
        alpha=0.4,
        label="Ventes quotidiennes",
    )
    ax.plot(
        timeline,
        daily["units_ma7"],
        linewidth=1.5,
        label="Moyenne mobile 7 jours",
    )

    ax.set(
        title="M5 - Evolution des ventes",
        xlabel="Date",
        ylabel="Unites vendues",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(output_dir / "sales_trend.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(categories["category"], categories["units_sold"])

    ax.set(
        title="M5 - Ventes par categorie",
        xlabel="Categorie",
        ylabel="Unites vendues",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "sales_categories.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse complete des ventes M5")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/m5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/m5/sales_analysis"),
    )
    parser.add_argument("--chunk-size", type=int, default=512)

    args = parser.parse_args()

    result = analyze_sales(
        args.data_dir,
        chunk_size=args.chunk_size,
    )

    export_analysis(args.output_dir, result)

    print(f"Series analysees : {result[0]['sales_series']}")
    print(f"Jours analyses : {result[0]['sales_days']}")
    print(f"Unites vendues : {result[0]['total_units_sold']}")
    print(f"Rapports et graphiques : {args.output_dir}")


if __name__ == "__main__":
    main()
