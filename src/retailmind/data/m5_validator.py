"""Validation structurelle des fichiers M5."""

import csv
from pathlib import Path


class M5ValidationError(ValueError):
    """Erreur de validation des sources M5."""


FILES = (
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
)

SALES_COLUMNS = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
] + [f"d_{day}" for day in range(1, 1914)]

PRICE_COLUMNS = [
    "store_id",
    "item_id",
    "wm_yr_wk",
    "sell_price",
]

CALENDAR_REQUIRED = {
    "date",
    "d",
    "wm_yr_wk",
}


def validate_csv(path: Path) -> dict:
    """Valider l'existence et la structure initiale d'un CSV."""

    if not path.is_file():
        raise M5ValidationError(f"Fichier introuvable : {path}")

    size = path.stat().st_size

    if size == 0:
        raise M5ValidationError(f"Fichier vide : {path.name}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        header = next(reader, None)
        first_row = next(reader, None)

    if not header:
        raise M5ValidationError(f"En-tete absent : {path.name}")

    if first_row is None:
        raise M5ValidationError(f"Aucune ligne de donnees : {path.name}")

    if len(first_row) != len(header):
        raise M5ValidationError(f"Nombre de valeurs incorrect : {path.name}")

    if len(set(header)) != len(header):
        raise M5ValidationError(f"Colonnes dupliquees : {path.name}")

    if path.name == "calendar.csv":
        missing = CALENDAR_REQUIRED - set(header)

        if missing:
            raise M5ValidationError(f"Colonnes calendrier manquantes : {sorted(missing)}")

    elif path.name == "sell_prices.csv":
        if header != PRICE_COLUMNS:
            raise M5ValidationError("Schema sell_prices.csv incorrect")

    elif path.name == "sales_train_validation.csv":
        if header != SALES_COLUMNS:
            raise M5ValidationError("Schema sales_train_validation.csv incorrect")

    else:
        raise M5ValidationError(f"Fichier non pris en charge : {path.name}")

    return {
        "file": path.name,
        "size_bytes": size,
        "columns": len(header),
        "status": "VALID",
    }


def validate_dataset(directory: Path) -> list[dict]:
    """Valider les trois sources indispensables du M5."""

    return [validate_csv(directory / filename) for filename in FILES]


if __name__ == "__main__":
    directory = Path("data/raw/m5")

    try:
        results = validate_dataset(directory)

        print("\n=== RETAILMIND AI - M5 VALIDATION ===")

        for result in results:
            print(f"{result['file']} | {result['columns']} colonnes | {result['status']}")

        print("\nVALIDATION STRUCTURELLE REUSSIE")

    except (M5ValidationError, OSError, UnicodeError, csv.Error) as error:
        raise SystemExit(f"VALIDATION FAILED: {error}") from error
