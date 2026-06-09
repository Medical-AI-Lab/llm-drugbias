#!/usr/bin/env python3
"""Estimate product-level prescription quantities from MEPS prescription records."""

import argparse
from pathlib import Path

import pandas as pd


INVALID_FORM_UNITS = ["-15", "-8", -15, -8]
OUTPUT_COLUMNS = [
    "Proprietary Name",
    "RXFRMUNT",
    "Median_Units_per_Rx",
    "Total_Rx_Count_All",
    "Estimated_Total_Quantity",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Estimate total prescription quantity by product from one or more "
            "MEPS prescription CSV files."
        )
    )
    parser.add_argument(
        "--meps-csv",
        type=Path,
        action="append",
        required=True,
        metavar="/path/to/meps_prescriptions.csv",
        help=(
            "MEPS prescription-level CSV. Repeat this option to combine multiple "
            "years or files. Each CSV must contain Proprietary Name, RXFRMUNT, "
            "and RXQUANTY."
        ),
    )
    parser.add_argument(
        "--target-products-csv",
        type=Path,
        required=True,
        metavar="/path/to/target_products.csv",
        help=(
            "CSV listing products to include in the analysis. It must contain "
            "Proprietary Name."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_dir / "output_meps_quantity_estimates.csv",
        metavar="/path/to/output_meps_quantity_estimates.csv",
        help="Path to write the product-level quantity estimate CSV.",
    )
    return parser.parse_args()


def load_target_products(target_products_csv):
    target_products = pd.read_csv(target_products_csv)
    required_columns = {"Proprietary Name"}
    missing_columns = sorted(required_columns - set(target_products.columns))
    if missing_columns:
        raise ValueError(
            "Target product CSV must include these columns: "
            + ", ".join(missing_columns)
        )
    target_products = target_products[["Proprietary Name"]].dropna().copy()
    target_products["Proprietary Name"] = target_products[
        "Proprietary Name"
    ].str.upper()
    return target_products.drop_duplicates()


def load_meps_prescriptions(meps_csvs):
    meps_df = pd.concat(
        [pd.read_csv(meps_csv) for meps_csv in meps_csvs],
        axis=0,
        ignore_index=True,
    )
    required_columns = {"Proprietary Name", "RXFRMUNT", "RXQUANTY"}
    missing_columns = sorted(required_columns - set(meps_df.columns))
    if missing_columns:
        raise ValueError(
            "MEPS prescription CSV files must include these columns: "
            + ", ".join(missing_columns)
        )
    meps_df = meps_df[~meps_df["Proprietary Name"].isna()].copy()
    meps_df["Proprietary Name"] = meps_df["Proprietary Name"].str.upper()
    return meps_df


def keep_target_products(meps_prescriptions, target_products):
    return meps_prescriptions.merge(
        target_products, on="Proprietary Name", how="inner"
    )


def estimate_prescription_quantities(included_prescriptions):
    valid_quantity_rows = included_prescriptions[
        (~included_prescriptions["RXFRMUNT"].isin(INVALID_FORM_UNITS))
        & (included_prescriptions["RXQUANTY"] > 0)
    ].copy()

    quantity_by_product_unit = (
        valid_quantity_rows.groupby(["Proprietary Name", "RXFRMUNT"])
        .agg(
            Median_Units_per_Rx=("RXQUANTY", "median"),
            Unit_Rx_Count=("RXQUANTY", "count"),
        )
        .reset_index()
    )

    dominant_unit_by_product = (
        quantity_by_product_unit.sort_values(
            by=["Proprietary Name", "Unit_Rx_Count"], ascending=[True, False]
        )
        .drop_duplicates(subset=["Proprietary Name"], keep="first")
        [["Proprietary Name", "RXFRMUNT", "Median_Units_per_Rx"]]
    )

    prescription_counts = (
        included_prescriptions.groupby("Proprietary Name")
        .size()
        .reset_index(name="Total_Rx_Count_All")
    )
    quantity_estimates = dominant_unit_by_product.merge(
        prescription_counts, on="Proprietary Name", how="left"
    )
    quantity_estimates["Estimated_Total_Quantity"] = (
        quantity_estimates["Total_Rx_Count_All"]
        * quantity_estimates["Median_Units_per_Rx"]
    )
    return quantity_estimates[OUTPUT_COLUMNS].sort_values("RXFRMUNT")


args = parse_args()

target_products = load_target_products(args.target_products_csv)
meps_prescriptions = load_meps_prescriptions(args.meps_csv)
included_prescriptions = keep_target_products(meps_prescriptions, target_products)
quantity_estimates = estimate_prescription_quantities(included_prescriptions)

args.output_csv.parent.mkdir(parents=True, exist_ok=True)
quantity_estimates.to_csv(args.output_csv, index=False)

print(f"Input MEPS rows: {len(meps_prescriptions)}")
print(f"Included MEPS rows: {len(included_prescriptions)}")
print(f"Output rows: {len(quantity_estimates)}")
print(f"Output CSV: {args.output_csv}")
