"""Controle de couverture des semaines du calendrier par les prix M5."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from retailmind.data.m5_validator import validate_dataset


def analyze_price_calendar(data_dir: Path, chunk_size: int = 100_000) -> tuple[dict, pd.DataFrame]:
    """Parcourir tous les prix sans charger les ventes journalieres."""
    if chunk_size < 1:
        raise ValueError("chunk_size doit etre strictement positif")

    validate_dataset(data_dir)

    calendar = pd.read_csv(
        data_dir / "calendar.csv",
        usecols=["date", "d", "wm_yr_wk"],
        dtype={"d": "string", "wm_yr_wk": "string"},
    )

    dates = pd.to_datetime(calendar["date"], errors="coerce")

    if (
        calendar.empty
        or dates.isna().any()
        or calendar[["d", "wm_yr_wk"]].isna().any().any()
        or calendar["d"].duplicated().any()
    ):
        raise ValueError("Calendrier incomplet ou invalide")

    sales_header = pd.read_csv(data_dir / "sales_train_validation.csv", nrows=0).columns

    sales_days = [column for column in sales_header if column.startswith("d_")]

    calendar_weeks = calendar.set_index("d")["wm_yr_wk"]

    missing_days = sorted(set(sales_days) - set(calendar_weeks.index))

    if missing_days:
        raise ValueError(f"Jours de vente absents du calendrier : {missing_days[:5]}")

    week_calendar_days = Counter(calendar["wm_yr_wk"].astype(str))

    week_sales_days = Counter(calendar_weeks.loc[sales_days].astype(str))

    price_week_rows: Counter = Counter()

    price_rows = 0
    missing_keys = 0
    bad_prices = 0
    nonpositive_prices = 0

    min_price = None
    max_price = None

    for chunk in pd.read_csv(
        data_dir / "sell_prices.csv",
        dtype={
            "store_id": "string",
            "item_id": "string",
            "wm_yr_wk": "string",
        },
        chunksize=chunk_size,
    ):
        price_rows += len(chunk)

        missing_keys += int(chunk[["store_id", "item_id", "wm_yr_wk"]].isna().any(axis=1).sum())

        price_week_rows.update(
            {str(week): int(count) for week, count in (chunk["wm_yr_wk"].value_counts().items())}
        )

        amounts = pd.to_numeric(chunk["sell_price"], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )

        finite = np.isfinite(amounts)

        bad_prices += int((~finite).sum())

        nonpositive_prices += int((finite & (amounts <= 0)).sum())

        positive = amounts[finite & (amounts > 0)]

        if positive.size:
            lo = float(positive.min())
            hi = float(positive.max())

            min_price = lo if min_price is None else min(min_price, lo)
            max_price = hi if max_price is None else max(max_price, hi)

    known_weeks = set(week_calendar_days)
    observed_weeks = set(price_week_rows)

    weeks_without_prices = sorted(set(week_sales_days) - observed_weeks)

    unknown_weeks = sorted(observed_weeks - known_weeks)

    weekly = pd.DataFrame(
        [
            {
                "wm_yr_wk": week,
                "calendar_days": week_calendar_days[week],
                "sales_days": week_sales_days[week],
                "price_rows": price_week_rows[week],
            }
            for week in sorted(known_weeks | observed_weeks)
        ]
    )

    report = {
        "scope": ("Toutes les lignes de sell_prices.csv ; couverture globale par semaine"),
        "calendar_days": len(calendar),
        "sales_days": len(sales_days),
        "calendar_weeks": len(known_weeks),
        "sales_weeks": len(week_sales_days),
        "price_rows": price_rows,
        "price_weeks": len(observed_weeks),
        "missing_price_keys": missing_keys,
        "missing_or_invalid_prices": bad_prices,
        "nonpositive_prices": nonpositive_prices,
        "min_positive_price": min_price,
        "max_positive_price": max_price,
        "sales_weeks_without_any_price": weeks_without_prices,
        "price_weeks_outside_calendar": unknown_weeks,
        "price_rows_outside_calendar": sum(price_week_rows[week] for week in unknown_weeks),
        "caution": (
            "Une semaine couverte globalement ne garantit "
            "pas un prix pour chaque couple produit-magasin. "
            "Les doublons de cles restent a verifier."
        ),
    }

    return report, weekly


def export_price_calendar(
    output_dir: Path,
    result: tuple[dict, pd.DataFrame],
) -> None:
    """Exporter le bilan et les comptages hebdomadaires."""
    report, weekly = result

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "price_calendar_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    weekly.to_csv(
        output_dir / "price_rows_by_week.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Couverture des prix et semaines M5")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/m5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/m5/price_calendar"),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
    )

    args = parser.parse_args()

    result = analyze_price_calendar(
        args.data_dir,
        args.chunk_size,
    )

    export_price_calendar(args.output_dir, result)

    report = result[0]

    print(f"Lignes de prix analysees : {report['price_rows']}")
    print(f"Semaines calendrier : {report['calendar_weeks']}")
    print(f"Semaines de prix : {report['price_weeks']}")
    print(f"Rapports : {args.output_dir}")


if __name__ == "__main__":
    main()
