"""Analyse de la saisonnalite et de la continuite des ventes quotidiennes M5."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

WEEKDAY_NAMES = (
    "Lundi",
    "Mardi",
    "Mercredi",
    "Jeudi",
    "Vendredi",
    "Samedi",
    "Dimanche",
)


def analyze_seasonality(csv_path: Path) -> tuple:
    """Analyser toutes les dates agregees, sans recharger les ventes brutes."""
    source = pd.read_csv(csv_path)
    required = {"date", "d", "units_sold"}
    missing = required - set(source.columns)

    if missing:
        raise ValueError(f"Colonnes obligatoires manquantes : {sorted(missing)}")

    if source.empty:
        raise ValueError("Le fichier de ventes quotidiennes est vide")

    dates = pd.to_datetime(source["date"], format="%Y-%m-%d", errors="coerce")
    quantities = pd.to_numeric(source["units_sold"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Dates invalides dans les ventes quotidiennes")

    if dates.duplicated().any():
        raise ValueError("Dates dupliquees dans les ventes quotidiennes")

    if quantities.isna().any() or not np.isfinite(quantities.to_numpy()).all():
        raise ValueError("Quantites manquantes ou non numeriques")

    if (quantities < 0).any() or (quantities != np.floor(quantities)).any():
        raise ValueError("Quantites negatives ou non entieres")

    daily = (
        pd.DataFrame(
            {
                "date": dates,
                "d": source["d"].astype("string"),
                "units_sold": quantities.astype("int64"),
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    intervals = daily["date"].diff().dropna()

    if not intervals.eq(pd.Timedelta(days=1)).all():
        raise ValueError("Jours manquants dans la serie temporelle")

    day_ids = pd.to_numeric(
        daily["d"].str.extract(r"^d_(\d+)$", expand=False),
        errors="coerce",
    )

    if day_ids.isna().any() or not day_ids.diff().dropna().eq(1).all():
        raise ValueError("Identifiants des jours incoherents avec le calendrier")

    daily["units_ma7"] = daily["units_sold"].rolling(7, min_periods=7).mean().round(2)
    daily["units_ma28"] = daily["units_sold"].rolling(28, min_periods=28).mean().round(2)

    daily["weekday_number"] = daily["date"].dt.weekday
    daily["month_number"] = daily["date"].dt.month
    daily["year_month"] = daily["date"].dt.strftime("%Y-%m")

    weekdays = (
        daily.groupby("weekday_number", as_index=False)
        .agg(
            days_observed=("units_sold", "size"),
            total_units=("units_sold", "sum"),
        )
        .sort_values("weekday_number")
    )

    weekdays["weekday_name"] = weekdays["weekday_number"].map(dict(enumerate(WEEKDAY_NAMES)))

    weekdays["mean_units_per_day"] = (weekdays["total_units"] / weekdays["days_observed"]).round(2)

    weekdays = weekdays[
        [
            "weekday_number",
            "weekday_name",
            "days_observed",
            "total_units",
            "mean_units_per_day",
        ]
    ]

    months = (
        daily.groupby("year_month", as_index=False)
        .agg(
            days_observed=("units_sold", "size"),
            total_units=("units_sold", "sum"),
        )
        .sort_values("year_month")
    )

    months["mean_units_per_day"] = (months["total_units"] / months["days_observed"]).round(2)

    seasonal_months = (
        daily.groupby("month_number", as_index=False)
        .agg(
            days_observed=("units_sold", "size"),
            total_units=("units_sold", "sum"),
        )
        .sort_values("month_number")
    )

    seasonal_months["mean_units_per_day"] = (
        seasonal_months["total_units"] / seasonal_months["days_observed"]
    ).round(2)

    total = int(daily["units_sold"].sum())

    if int(weekdays["total_units"].sum()) != total:
        raise ValueError("Sommes incoherentes entre jours et jours de semaine")

    if int(months["total_units"].sum()) != total:
        raise ValueError("Sommes incoherentes entre jours et mois")

    max_weekday = weekdays.loc[weekdays["mean_units_per_day"].idxmax()]

    summary = {
        "scope": ("Ventes quotidiennes agregees sur toutes les series M5"),
        "days": len(daily),
        "first_date": daily["date"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": daily["date"].iloc[-1].strftime("%Y-%m-%d"),
        "total_units_sold": total,
        "mean_units_per_day": round(float(daily["units_sold"].mean()), 2),
        "weekday_highest_mean": str(max_weekday["weekday_name"]),
        "calendar_continuity": "VALID",
        "note": (
            "Statistiques descriptives, non causales. "
            "Les moyennes mensuelles tiennent compte "
            "des jours observes. Les premieres fenetres "
            "de moyennes mobiles sont vides."
        ),
    }

    return summary, daily, weekdays, months, seasonal_months


def export_seasonality(output_dir: Path, result: tuple) -> None:
    """Exporter les indicateurs et trois graphiques."""
    import matplotlib.pyplot as plt

    summary, daily, weekdays, months, seasonal_months = result

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "seasonality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    daily.to_csv(
        output_dir / "daily_trend.csv",
        index=False,
        date_format="%Y-%m-%d",
    )

    weekdays.to_csv(output_dir / "sales_by_weekday.csv", index=False)

    months.to_csv(output_dir / "monthly_trend.csv", index=False)

    seasonal_months.to_csv(output_dir / "sales_by_calendar_month.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        daily["date"],
        daily["units_sold"],
        alpha=0.2,
        label="Ventes quotidiennes",
    )
    ax.plot(
        daily["date"],
        daily["units_ma7"],
        label="Moyenne mobile 7 jours",
    )
    ax.plot(
        daily["date"],
        daily["units_ma28"],
        label="Moyenne mobile 28 jours",
    )
    ax.set(
        title="M5 - Tendances des ventes",
        xlabel="Date",
        ylabel="Unites vendues",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "sales_trend_7_28.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        weekdays["weekday_name"],
        weekdays["mean_units_per_day"],
    )
    ax.set(
        title="M5 - Ventes moyennes par jour de semaine",
        ylabel="Unites par jour",
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "weekday_seasonality.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))

    x_positions = np.arange(len(months))
    ax.plot(
        x_positions,
        months["mean_units_per_day"],
    )

    ax.set(
        title="M5 - Ventes moyennes par mois",
        xlabel="Mois",
        ylabel="Unites par jour",
    )

    step = max(1, len(months) // 12)
    ax.set_xticks(
        x_positions[::step],
        months["year_month"].iloc[::step],
        rotation=45,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "monthly_trend.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse de la saisonnalite M5")

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/m5/sales_analysis/daily_sales.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/m5/seasonality"),
    )

    args = parser.parse_args()

    result = analyze_seasonality(args.input)
    export_seasonality(args.output_dir, result)

    print(f"Jours analyses : {result[0]['days']}")
    print(f"Total unites : {result[0]['total_units_sold']}")
    print(f"Rapports et graphiques : {args.output_dir}")


if __name__ == "__main__":
    main()
