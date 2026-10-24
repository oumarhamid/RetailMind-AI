"""Controle des prix par produit, magasin et semaine de vente M5."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailmind.data.m5_validator import validate_dataset


def analyze_price_integrity(
    data_dir: Path,
    price_chunk_size: int = 100_000,
    sales_chunk_size: int = 256,
) -> tuple[dict, pd.DataFrame]:
    """Verifier les cles de prix et les ventes sans prix par lecture par lots."""
    if price_chunk_size < 1 or sales_chunk_size < 1:
        raise ValueError("Les tailles des lots doivent etre strictement positives")

    validate_dataset(data_dir)
    sales_path = data_dir / "sales_train_validation.csv"

    calendar = pd.read_csv(
        data_dir / "calendar.csv",
        usecols=["d", "wm_yr_wk"],
        dtype="string",
    )

    if calendar[["d", "wm_yr_wk"]].isna().any().any() or calendar["d"].duplicated().any():
        raise ValueError("Calendrier invalide : jours ou semaines manquants ou dupliques")

    day_columns = [
        name for name in pd.read_csv(sales_path, nrows=0).columns if name.startswith("d_")
    ]

    day_week = calendar.set_index("d")["wm_yr_wk"]

    if set(day_columns) - set(day_week.index):
        raise ValueError("Jours de ventes absents du calendrier")

    calendar_weeks = sorted(calendar["wm_yr_wk"].unique().tolist())
    week_bits = {week: 1 << index for index, week in enumerate(calendar_weeks)}

    sales_week_days: dict[str, list[int]] = {}

    for index, day in enumerate(day_columns):
        sales_week_days.setdefault(str(day_week.loc[day]), []).append(index)

    sales_ids = pd.read_csv(
        sales_path,
        usecols=["store_id", "item_id"],
        dtype="string",
    )

    if sales_ids.isna().any().any() or sales_ids.duplicated().any():
        raise ValueError("Identifiants des couples produit-magasin invalides")

    sales_pairs = set(sales_ids[["store_id", "item_id"]].itertuples(index=False, name=None))

    # Chaque couple possede un masque binaire des semaines disponibles.
    # Ceci evite de conserver des millions de tuples en memoire.
    price_masks: dict[tuple[str, str], int] = {}

    price_rows = 0
    duplicate_keys = 0
    missing_price_keys = 0
    unknown_price_week_rows = 0
    duplicate_examples: list[list[str]] = []

    for chunk in pd.read_csv(
        data_dir / "sell_prices.csv",
        usecols=["store_id", "item_id", "wm_yr_wk"],
        dtype="string",
        chunksize=price_chunk_size,
    ):
        price_rows += len(chunk)

        for store, item, week in chunk.itertuples(index=False, name=None):
            if pd.isna(store) or pd.isna(item) or pd.isna(week):
                missing_price_keys += 1
                continue

            bit = week_bits.get(week)

            if bit is None:
                unknown_price_week_rows += 1
                continue

            pair = (str(store), str(item))
            current = price_masks.get(pair, 0)

            if current & bit:
                duplicate_keys += 1

                if len(duplicate_examples) < 5:
                    duplicate_examples.append([str(store), str(item), str(week)])
            else:
                price_masks[pair] = current | bit

    weekly_stats: dict[str, dict[str, int]] = {}

    for week in sales_week_days:
        bit = week_bits[week]

        weekly_stats[week] = {
            "pairs_with_price": sum(bool(price_masks.get(pair, 0) & bit) for pair in sales_pairs),
            "pairs_with_positive_sales": 0,
            "positive_sales_pairs_without_price": 0,
            "units_without_price": 0,
        }

    positive_pair_weeks = 0
    uncovered_positive_pair_weeks = 0
    uncovered_units = 0
    total_units = 0
    missing_examples: list[dict] = []

    dtypes = {
        "store_id": "string",
        "item_id": "string",
    }

    dtypes.update({day: "float64" for day in day_columns})

    for chunk in pd.read_csv(
        sales_path,
        usecols=["store_id", "item_id", *day_columns],
        dtype=dtypes,
        chunksize=sales_chunk_size,
    ):
        values = chunk[day_columns].to_numpy(dtype=np.float64)

        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values != np.floor(values)).any()
        ):
            raise ValueError("Quantites vendues invalides")

        total_units += int(values.sum())

        pairs = list(chunk[["store_id", "item_id"]].itertuples(index=False, name=None))

        for week, positions in sales_week_days.items():
            week_units = values[:, positions].sum(axis=1)
            positive_indices = np.flatnonzero(week_units > 0)

            stats = weekly_stats[week]
            stats["pairs_with_positive_sales"] += len(positive_indices)

            positive_pair_weeks += len(positive_indices)
            bit = week_bits[week]

            for index in positive_indices:
                pair = pairs[int(index)]

                if not (price_masks.get(pair, 0) & bit):
                    units = int(week_units[index])

                    stats["positive_sales_pairs_without_price"] += 1
                    stats["units_without_price"] += units

                    uncovered_positive_pair_weeks += 1
                    uncovered_units += units

                    if len(missing_examples) < 10:
                        missing_examples.append(
                            {
                                "store_id": str(pair[0]),
                                "item_id": str(pair[1]),
                                "wm_yr_wk": week,
                                "units_sold": units,
                            }
                        )

    sales_pair_week_count = len(sales_pairs) * len(sales_week_days)

    sales_weeks_mask = sum(week_bits[week] for week in sales_week_days)

    covered_pair_weeks = sum(
        (price_masks.get(pair, 0) & sales_weeks_mask).bit_count() for pair in sales_pairs
    )

    weekly = pd.DataFrame(
        [
            {
                "wm_yr_wk": week,
                "sales_pairs": len(sales_pairs),
                **weekly_stats[week],
            }
            for week in sorted(sales_week_days)
        ]
    )

    report = {
        "scope": ("Toutes les lignes de prix et toutes les ventes historiques M5"),
        "price_rows": price_rows,
        "price_duplicate_keys": duplicate_keys,
        "duplicate_key_examples": duplicate_examples,
        "price_rows_missing_keys": missing_price_keys,
        "price_rows_outside_calendar": unknown_price_week_rows,
        "price_pairs_absent_from_sales": len(set(price_masks) - sales_pairs),
        "sales_pairs": len(sales_pairs),
        "sales_weeks": len(sales_week_days),
        "possible_sales_pair_weeks": sales_pair_week_count,
        "pair_weeks_with_price": covered_pair_weeks,
        "pair_weeks_without_price": (sales_pair_week_count - covered_pair_weeks),
        "positive_sales_pair_weeks": positive_pair_weeks,
        "positive_sales_pair_weeks_without_price": (uncovered_positive_pair_weeks),
        "units_sold_total": total_units,
        "units_sold_without_weekly_price": uncovered_units,
        "unpriced_sales_examples": missing_examples,
        "caution": (
            "Les semaines sans prix et sans ventes "
            "peuvent preceder la commercialisation du produit."
        ),
    }

    return report, weekly


def export_price_integrity(
    output_dir: Path,
    result: tuple[dict, pd.DataFrame],
) -> None:
    """Exporter la synthese et la couverture des semaines."""
    report, weekly = result

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "price_integrity_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    weekly.to_csv(
        output_dir / "weekly_price_coverage.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrite des prix M5")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/m5"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/m5/price_integrity"),
    )

    parser.add_argument(
        "--price-chunk-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--sales-chunk-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    result = analyze_price_integrity(
        args.data_dir,
        args.price_chunk_size,
        args.sales_chunk_size,
    )

    export_price_integrity(args.output_dir, result)

    print(f"Prix analyses : {result[0]['price_rows']}")
    print(f"Doublons de prix : {result[0]['price_duplicate_keys']}")
    print(f"Unites vendues sans prix hebdomadaire : {result[0]['units_sold_without_weekly_price']}")
    print(f"Rapports : {args.output_dir}")


if __name__ == "__main__":
    main()
