"""Tests du rapport d'inventaire M5."""

import csv
from pathlib import Path

import pytest

from retailmind.data.m5_inventory import build_inventory
from retailmind.data.m5_validator import (
    PRICE_COLUMNS,
    SALES_COLUMNS,
)


def write_csv(
    path: Path,
    header: list[str],
    rows: list[list[str]],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def sample_m5(tmp_path: Path) -> Path:
    write_csv(
        tmp_path / "calendar.csv",
        ["date", "d", "wm_yr_wk"],
        [
            ["2011-01-29", "d_1", "11101"],
            ["2011-01-30", "d_2", "11101"],
        ],
    )

    write_csv(
        tmp_path / "sales_train_validation.csv",
        SALES_COLUMNS,
        [
            [
                "one",
                "ITEM_1",
                "DEP_1",
                "CAT_1",
                "CA_1",
                "CA",
            ]
            + ["1"] * 1913,
            [
                "two",
                "ITEM_2",
                "DEP_1",
                "CAT_1",
                "CA_1",
                "CA",
            ]
            + ["0"] * 1913,
        ],
    )

    write_csv(
        tmp_path / "sell_prices.csv",
        PRICE_COLUMNS,
        [
            ["CA_1", "ITEM_1", "11101", "2.5"],
            ["CA_1", "ITEM_1", "11102", "0"],
            ["CA_1", "ITEM_OTHER", "11101", "5"],
        ],
    )

    return tmp_path


def test_inventory_across_price_chunks(
    sample_m5: Path,
) -> None:
    report = build_inventory(
        sample_m5,
        price_chunk_size=2,
    )

    assert report["sales"]["series"] == 2
    assert report["sales"]["unique_items"] == 2
    assert report["sales"]["series_by_category"] == {"CAT_1": 2}
    assert report["sales"]["duplicate_series_ids"] == 0

    assert report["prices"]["rows"] == 3
    assert report["prices"]["chunks_processed"] == 2
    assert report["prices"]["unique_item_store_pairs"] == 2
    assert report["prices"]["nonpositive_amounts"] == 1
    assert report["prices"]["min_positive_price"] == 2.5

    assert report["relationships"]["sales_pairs_without_any_price"] == 1

    assert report["relationships"]["price_pairs_absent_from_sales"] == 1

    assert report["relationships"]["unknown_price_pair_examples"] == [["CA_1", "ITEM_OTHER"]]


def test_reject_invalid_chunk_size(
    sample_m5: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="strictement positif",
    ):
        build_inventory(
            sample_m5,
            price_chunk_size=0,
        )
