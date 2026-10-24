"""Consolider les indicateurs M5 dans un rapport EDA reproductible."""

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path


def _load_json(root: Path, name: str) -> dict:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Rapport requis introuvable : {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(root: Path, name: str) -> list[dict]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Table requise introuvable : {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Table vide : {path}")
    return rows


def _sum_column(rows: list[dict], column: str) -> Decimal:
    return sum((Decimal(row[column]) for row in rows), Decimal(0))


def _check_total(
    label: str,
    actual: Decimal,
    expected: Decimal,
    tolerance: str = "0",
) -> None:
    if abs(actual - expected) > Decimal(tolerance):
        raise ValueError(f"Reconciliation impossible pour {label} : {actual} != {expected}")


def _fmt_int(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def _fmt_amount(value: Decimal) -> str:
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def build_eda_report(root: Path) -> str:
    """Valider les indicateurs et produire le rapport Markdown."""
    inventory = _load_json(root, "inventory.json")
    sales = _load_json(root, "sales_analysis/sales_summary.json")
    seasonality = _load_json(root, "seasonality/seasonality_summary.json")
    coverage = _load_json(root, "price_calendar/price_calendar_summary.json")
    integrity = _load_json(root, "price_integrity/price_integrity_summary.json")
    revenue = _load_json(root, "revenue/revenue_summary.json")

    weeks = _load_csv(root, "revenue/weekly_revenue.csv")
    categories = _load_csv(root, "revenue/revenue_by_category.csv")
    stores = _load_csv(root, "revenue/revenue_by_store.csv")
    top_items = _load_csv(root, "revenue/top_items_by_revenue.csv")
    weekdays = _load_csv(root, "seasonality/sales_by_weekday.csv")

    units = Decimal(sales["total_units_sold"])
    gross_revenue = Decimal(str(revenue["revenue_total"]))

    for label, value in (
        ("saisonnalite", seasonality["total_units_sold"]),
        ("integrite des prix", integrity["units_sold_total"]),
        ("revenus", revenue["total_units_sold"]),
    ):
        _check_total(label, Decimal(value), units)

    for label, rows in (
        ("ventes hebdomadaires", weeks),
        ("ventes par categorie", categories),
        ("ventes par magasin", stores),
    ):
        _check_total(label, _sum_column(rows, "units_sold"), units)
        _check_total(
            f"revenu {label}",
            _sum_column(rows, "revenue"),
            gross_revenue,
            "1.50" if label == "ventes hebdomadaires" else "0.10",
        )

    if len(weeks) != revenue["sales_weeks"]:
        raise ValueError("Nombre de semaines de revenu incoherent")
    if len(categories) != inventory["sales"]["unique_categories"]:
        raise ValueError("Nombre de categories incoherent")
    if len(stores) != inventory["sales"]["unique_stores"]:
        raise ValueError("Nombre de magasins incoherent")

    if len(weekdays) != 7 or seasonality["calendar_continuity"] != "VALID":
        raise ValueError("Continuite temporelle ou jours de semaine invalides")

    if integrity["price_duplicate_keys"] != 0:
        raise ValueError("Doublons de cles de prix detectes")

    if integrity["units_sold_without_weekly_price"] != 0 or revenue["units_without_price"] != 0:
        raise ValueError("Ventes positives sans prix hebdomadaire")

    if coverage["sales_weeks_without_any_price"]:
        raise ValueError("Semaines de ventes totalement sans prix")

    top_categories = sorted(
        categories,
        key=lambda row: Decimal(row["revenue"]),
        reverse=True,
    )
    top_stores = sorted(
        stores,
        key=lambda row: Decimal(row["revenue"]),
        reverse=True,
    )
    best_weekday = max(
        weekdays,
        key=lambda row: Decimal(row["mean_units_per_day"]),
    )

    lines = [
        "# RetailMind AI — Rapport EDA M5",
        "",
        "## 1. Perimetre et methode",
        "",
        "Analyse descriptive de `sales_train_validation.csv`, `calendar.csv` et `sell_prices.csv`.",
        "Les controles de coherence sont recalcules a partir des rapports locaux.",
        "",
        "## 2. Indicateurs principaux",
        "",
        "| Indicateur | Valeur |",
        "|---|---:|",
        f"| Series produit-magasin | {_fmt_int(sales['sales_series'])} |",
        f"| Produits distincts | {_fmt_int(inventory['sales']['unique_items'])} |",
        f"| Magasins | {_fmt_int(inventory['sales']['unique_stores'])} |",
        f"| Jours de ventes | {_fmt_int(sales['sales_days'])} |",
        f"| Periode | {sales['first_date']} au {sales['last_date']} |",
        f"| Unites vendues | {_fmt_int(int(units))} |",
        f"| Cellules sans vente | {sales['zero_sales_pct']} % |",
        f"| Lignes de prix | {_fmt_int(integrity['price_rows'])} |",
        f"| Revenu brut estime (unites monetaires M5) | {_fmt_amount(gross_revenue)} |",
        "",
        "## 3. Resultats commerciaux",
        "",
        "### Categories (revenu decroissant)",
        "",
        "| Categorie | Unites | Revenu estime |",
        "|---|---:|---:|",
    ]

    for row in top_categories:
        lines.append(
            f"| {row['category']} | "
            f"{_fmt_int(row['units_sold'])} | "
            f"{_fmt_amount(Decimal(row['revenue']))} |"
        )

    lines += [
        "",
        "### Magasins (revenu decroissant)",
        "",
        "| Magasin | Unites | Revenu estime |",
        "|---|---:|---:|",
    ]

    for row in top_stores:
        lines.append(
            f"| {row['store_id']} | "
            f"{_fmt_int(row['units_sold'])} | "
            f"{_fmt_amount(Decimal(row['revenue']))} |"
        )

    lines += [
        "",
        "### Premier produit par revenu estime",
        "",
        f"- {top_items[0]['item_id']} : {_fmt_amount(Decimal(top_items[0]['revenue']))}.",
        "",
        "## 4. Saisonnalite et qualite",
        "",
        f"- Jour de semaine avec la moyenne la plus elevee : {best_weekday['weekday_name']}.",
        f"- Semaines de ventes : {integrity['sales_weeks']}.",
        f"- Semaines calendaires couvertes par les prix : {coverage['price_weeks']}.",
        (
            "- Couples produit-magasin-semaine sans prix : "
            f"{_fmt_int(integrity['pair_weeks_without_price'])}."
        ),
        "- Unites vendues sans prix correspondant : 0.",
        "- Doublons de cles produit-magasin-semaine : 0.",
        "- Reconciliation des unites et revenus par semaine, categorie et magasin : validee.",
        "",
        "## 5. Limites et interpretation",
        "",
        "- Le revenu est estime a partir des prix hebdomadaires listes.",
        "- Il ne constitue pas un chiffre d'affaires comptable.",
        "- Taxes, retours et remises transactionnelles indisponibles.",
        (
            "- Une combinaison sans prix n'est pas une anomalie "
            "si elle ne comporte pas de vente positive sur la semaine."
        ),
        "- Les tendances observees ne prouvent pas de relation causale.",
        "- Les donnees M5 sont historiques et ne decrivent pas les ventes actuelles.",
        "",
        "## 6. Suite du projet",
        "",
        (
            "Construire le modele analytique PostgreSQL, "
            "les pipelines ETL, les tableaux de bord BI "
            "et les modeles de prevision."
        ),
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rapport EDA consolide RetailMind AI")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/m5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/eda/m5_eda_report.md"),
    )
    args = parser.parse_args()

    markdown = build_eda_report(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")

    print(f"Rapport EDA cree : {args.output}")
    print("Reconciliations des quantites et des revenus : OK")


if __name__ == "__main__":
    main()
