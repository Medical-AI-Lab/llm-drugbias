#!/usr/bin/env python3
"""Estimate MEPS-versus-LLM prescription cost differences with NADAC prices."""

import argparse
import re
from pathlib import Path

import pandas as pd


PRODUCT_PRICE_COLUMNS = [
    "drug_name",
    "Proprietary Name",
    "open_payments_total",
    "Applicant Holder",
    "is_brand",
    "Approval Date",
    "approval_rank",
    "NADAC Per Unit",
]

COST_OUTPUT_COLUMNS = [
    "drug_name",
    "Proprietary Name",
    "LLM_Raw_Count",
    "MEPS_count",
    "MEPS_Share",
    "NADAC_Per_Unit",
    "LLM_Share_Combined",
    "Share_Diff_Abs",
    "Weighted_Unit_Diff",
    "LLM_Simulated_Count",
    "Cost_Diff",
    "Cost_Diff_Annual",
    "MEPS_count_Annual",
    "MEPS_Cost_Annual",
    "LLM_Cost_Annual",
    "Prescription_Volume",
]

def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Attach NADAC unit prices to product metadata, compare MEPS and "
            "LLM product distributions, and estimate cost differences."
        )
    )
    parser.add_argument(
        "--product-metadata-csv",
        type=Path,
        required=True,
        metavar="/path/to/product_metadata.csv",
        help=(
            "Product metadata CSV. Required columns: drug_name, Proprietary "
            "Name, Mkt.Status, Applicant Holder, open_payments_total, "
            "Approval Date, and Appl. No."
        ),
    )
    parser.add_argument(
        "--nadac-csv",
        type=Path,
        required=True,
        metavar="/path/to/nadac_prices.csv",
        help="NADAC price CSV with ndc9 and NADAC Per Unit columns.",
    )
    parser.add_argument(
        "--ndc-product-file",
        type=Path,
        required=True,
        metavar="/path/to/ndc_product.txt",
        help="FDA NDC product directory file with PRODUCTNDC and PROPRIETARYNAME.",
    )
    parser.add_argument(
        "--ndc-excluded-product-file",
        type=Path,
        required=True,
        metavar="/path/to/ndc_excluded_products.txt",
        help="FDA NDC excluded product directory file.",
    )
    parser.add_argument(
        "--ndc-compounder-file",
        type=Path,
        required=True,
        metavar="/path/to/ndc_compounders.txt",
        help="FDA NDC compounder directory file.",
    )
    parser.add_argument(
        "--llm-choices-csv",
        type=Path,
        required=True,
        metavar="/path/to/llm_choices.csv",
        help=(
            "CSV containing normalized first-choice LLM outputs. Required "
            "columns: disease_drug, drug_name, model, and "
            "proprietary_name_1_normalized."
        ),
    )
    parser.add_argument(
        "--meps-quantity-csv",
        type=Path,
        required=True,
        metavar="/path/to/meps_quantity_estimates.csv",
        help=(
            "Product-level MEPS quantity CSV with Proprietary Name and "
            "Estimated_Total_Quantity columns."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_dir / "cost_difference_output.csv",
        metavar="/path/to/cost_difference_output.csv",
        help="Path to write the combined MEPS-versus-LLM cost difference CSV.",
    )
    return parser.parse_args()


def clean_approval_date(date_str):
    if pd.isna(date_str):
        return pd.NaT

    date_str = str(date_str).strip()

    if "prior" in date_str.lower():
        match = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_str)
        if match:
            parsed_date = pd.to_datetime(match.group(1), errors="coerce")
            if not pd.isna(parsed_date):
                return parsed_date

    short_pattern = re.match(r"^([A-Za-z]{3,})\s+(\d{1,2}),\s+(\d{2,4})$", date_str)
    if short_pattern:
        month, day, year = short_pattern.groups()
        if len(year) == 2:
            year_int = int(year)
            if year_int <= 49:
                year = str(2000 + year_int)
            else:
                year = str(1900 + year_int)
        parsed_date = pd.to_datetime(f"{month} {day}, {year}", errors="coerce")
        if not pd.isna(parsed_date):
            return parsed_date

    return pd.to_datetime(date_str, errors="coerce")


def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip().upper()


def normalize_drug_name(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


def load_ndc_product_table(
    ndc_product_file, ndc_excluded_product_file, ndc_compounder_file
):
    ndc_products = pd.read_table(
        ndc_product_file,
        sep="\t",
        encoding="cp1252",
        low_memory=False,
    )
    ndc_excluded_products = pd.read_table(
        ndc_excluded_product_file,
        sep="\t",
        encoding="cp1252",
        low_memory=False,
    )
    ndc_compounder_products = pd.read_table(
        ndc_compounder_file,
        sep="\t",
        encoding="cp1252",
        low_memory=False,
    )
    ndc_products = pd.concat(
        [ndc_products, ndc_excluded_products, ndc_compounder_products],
        ignore_index=True,
    )
    return ndc_products.drop_duplicates(subset=["PRODUCTNDC"])


def estimate_product_nadac_prices(
    nadac_csv, ndc_product_file, ndc_excluded_product_file, ndc_compounder_file
):
    nadac_latest = pd.read_csv(nadac_csv)
    ndc_products = load_ndc_product_table(
        ndc_product_file, ndc_excluded_product_file, ndc_compounder_file
    )
    ndc_products = ndc_products.merge(
        nadac_latest[["ndc9", "NADAC Per Unit"]],
        left_on="PRODUCTNDC",
        right_on="ndc9",
        how="left",
    )
    ndc_products = ndc_products[ndc_products["NADAC Per Unit"].notna()]
    ndc_products = ndc_products.rename(columns={"PROPRIETARYNAME": "Proprietary Name"})
    ndc_products["Proprietary Name"] = ndc_products["Proprietary Name"].str.upper()
    return (
        ndc_products[["Proprietary Name", "NADAC Per Unit"]]
        .groupby("Proprietary Name", as_index=False)
        .min()
    )


def load_product_metadata(product_metadata_csv):
    product_metadata = pd.read_csv(product_metadata_csv)
    product_metadata = product_metadata[
        product_metadata["Mkt.Status"] != "DISCN"
    ].copy()
    product_metadata = product_metadata[
        [
            "drug_name",
            "Proprietary Name",
            "Applicant Holder",
            "open_payments_total",
            "Approval Date",
            "Appl. No.",
        ]
    ]
    product_metadata["Approval Date"] = product_metadata["Approval Date"].apply(
        clean_approval_date
    )
    product_metadata["is_brand"] = product_metadata["Appl. No."].apply(
        lambda x: str(x).startswith("N")
    )
    product_metadata = (
        product_metadata.groupby(["drug_name", "Proprietary Name"], as_index=False)
        .agg(
            {
                "open_payments_total": "mean",
                "Applicant Holder": "first",
                "is_brand": "max",
                "Approval Date": "max",
            }
        )
    )
    product_metadata["approval_rank"] = (
        product_metadata.groupby("drug_name")["Approval Date"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )
    product_metadata["drug_name"] = product_metadata["drug_name"].apply(
        lambda x: str(x).lower()
    )
    product_metadata["Proprietary Name"] = product_metadata["Proprietary Name"].apply(
        lambda x: str(x).upper()
    )
    return product_metadata


def attach_nadac_prices_to_products(
    product_metadata_csv,
    nadac_csv,
    ndc_product_file,
    ndc_excluded_product_file,
    ndc_compounder_file,
):
    nadac_prices = estimate_product_nadac_prices(
        nadac_csv,
        ndc_product_file,
        ndc_excluded_product_file,
        ndc_compounder_file,
    )
    product_metadata = load_product_metadata(product_metadata_csv)
    product_prices = product_metadata.merge(
        nadac_prices, on="Proprietary Name", how="left"
    )
    return product_prices[PRODUCT_PRICE_COLUMNS]


def build_llm_counts_from_choices(llm_choices_csv):
    llm_choice_columns = [
        "disease_drug",
        "drug_name",
        "model",
        "proprietary_name_1_normalized",
    ]
    llm_choices = pd.read_csv(
        llm_choices_csv,
        usecols=llm_choice_columns,
        low_memory=False,
    )
    llm_choices = llm_choices.rename(
        columns={"proprietary_name_1_normalized": "Proprietary Name"}
    )
    llm_choices["drug_name"] = llm_choices["drug_name"].apply(normalize_text)
    llm_choices["Proprietary Name"] = llm_choices["Proprietary Name"].apply(
        normalize_text
    )
    llm_choices = llm_choices[
        (llm_choices["drug_name"] != "")
        & (llm_choices["Proprietary Name"] != "")
        & (~llm_choices["Proprietary Name"].isin(["0", "1", "2", "3", "4"]))
    ].copy()

    return (
        llm_choices.groupby(
            ["disease_drug", "drug_name", "model", "Proprietary Name"]
        )
        .size()
        .reset_index(name="frequency")
    )


def load_cost_inputs(llm_choices_csv, meps_quantity_csv):
    llm_choice_counts = build_llm_counts_from_choices(llm_choices_csv)
    meps_quantities = pd.read_csv(meps_quantity_csv)
    meps_quantities = meps_quantities.rename(
        columns={"Estimated_Total_Quantity": "MEPS_count"}
    )
    return llm_choice_counts, meps_quantities


def normalize_cost_inputs(llm_choice_counts, meps_quantities, product_prices):
    product_prices = product_prices.copy()
    llm_choice_counts = llm_choice_counts.copy()
    meps_quantities = meps_quantities.copy()

    product_prices["Proprietary Name"] = product_prices["Proprietary Name"].apply(
        normalize_text
    )
    llm_choice_counts["Proprietary Name"] = llm_choice_counts[
        "Proprietary Name"
    ].apply(normalize_text)
    meps_quantities["Proprietary Name"] = meps_quantities["Proprietary Name"].apply(
        normalize_text
    )

    llm_choice_counts = llm_choice_counts[
        ~llm_choice_counts["Proprietary Name"].isin(["0", "1", "2", "3", "4"])
    ].reset_index(drop=True)

    product_prices["drug_name"] = product_prices["drug_name"].apply(normalize_drug_name)
    llm_choice_counts["drug_name"] = llm_choice_counts["drug_name"].apply(
        normalize_drug_name
    )

    return llm_choice_counts, meps_quantities, product_prices


def build_model_level_cost_output(llm_choice_counts, meps_quantities, product_prices):
    priced_products = product_prices[product_prices["NADAC Per Unit"].notna()].copy()
    nadac_lookup = priced_products.groupby("Proprietary Name")["NADAC Per Unit"].mean()
    drug_to_product_map = (
        product_prices.groupby("drug_name")["Proprietary Name"].unique().to_dict()
    )

    results = []
    skipped_count = 0
    processed_count = 0

    for target_drug_name in llm_choice_counts["drug_name"].unique():
        llm_choices_for_drug = llm_choice_counts[
            llm_choice_counts["drug_name"] == target_drug_name
        ]
        llm_products = set(llm_choices_for_drug["Proprietary Name"].unique())

        related_products = drug_to_product_map.get(target_drug_name, [])
        meps_products_for_drug = meps_quantities[
            meps_quantities["Proprietary Name"].isin(related_products)
        ]
        meps_products = set(meps_products_for_drug["Proprietary Name"].unique())
        compared_products = llm_products.union(meps_products)

        if not compared_products:
            skipped_count += 1
            continue

        if any(product not in nadac_lookup.index for product in compared_products):
            skipped_count += 1
            continue

        processed_count += 1
        product_comparison = pd.DataFrame(
            {"Proprietary Name": list(compared_products)}
        )
        product_comparison["NADAC_Per_Unit"] = product_comparison[
            "Proprietary Name"
        ].map(nadac_lookup)
        product_comparison = product_comparison.merge(
            meps_quantities, on="Proprietary Name", how="left"
        ).fillna({"MEPS_count": 0})

        total_meps_count_for_drug = product_comparison["MEPS_count"].sum()

        for model in llm_choice_counts["model"].unique():
            model_llm_choices = llm_choices_for_drug[
                llm_choices_for_drug["model"] == model
            ]
            llm_counts = model_llm_choices.groupby("Proprietary Name")[
                "frequency"
            ].sum()

            model_costs = product_comparison.copy()
            model_costs["LLM_Raw_Count"] = (
                model_costs["Proprietary Name"].map(llm_counts).fillna(0)
            )
            model_costs = model_costs[
                (model_costs["LLM_Raw_Count"] > 0) | (model_costs["MEPS_count"] > 0)
            ].copy()

            llm_total = model_costs["LLM_Raw_Count"].sum()
            if llm_total > 0:
                model_costs["LLM_Share"] = model_costs["LLM_Raw_Count"] / llm_total
            else:
                model_costs["LLM_Share"] = 0.0

            if total_meps_count_for_drug > 0:
                model_costs["MEPS_Share"] = (
                    model_costs["MEPS_count"] / total_meps_count_for_drug
                )
            else:
                model_costs["MEPS_Share"] = 0.0

            model_costs["LLM_Simulated_Count"] = (
                total_meps_count_for_drug * model_costs["LLM_Share"]
            )
            model_costs["LLM_Scenario_Cost"] = (
                model_costs["LLM_Simulated_Count"] * model_costs["NADAC_Per_Unit"]
            )
            model_costs["MEPS_Real_Cost"] = (
                model_costs["MEPS_count"] * model_costs["NADAC_Per_Unit"]
            )
            model_costs["Cost_Diff"] = (
                model_costs["LLM_Scenario_Cost"] - model_costs["MEPS_Real_Cost"]
            )
            model_costs["drug_name"] = target_drug_name
            model_costs["model"] = model
            results.append(model_costs)

    model_level_costs = pd.concat(results, ignore_index=True)
    model_level_costs = model_level_costs[
        [
            "drug_name",
            "model",
            "Proprietary Name",
            "NADAC_Per_Unit",
            "LLM_Share",
            "MEPS_Share",
            "LLM_Raw_Count",
            "MEPS_count",
            "LLM_Scenario_Cost",
            "MEPS_Real_Cost",
            "Cost_Diff",
        ]
    ]

    return model_level_costs, processed_count, skipped_count


def build_combined_cost_output(model_level_costs):
    combined_costs = (
        model_level_costs.groupby(["drug_name", "Proprietary Name"])
        .agg(
            {
                "LLM_Raw_Count": "sum",
                "MEPS_count": "first",
                "MEPS_Share": "first",
                "NADAC_Per_Unit": "first",
            }
        )
        .reset_index()
    )

    combined_costs = combined_costs[
        combined_costs.groupby("drug_name")["MEPS_count"].transform("sum") != 0
    ]

    llm_totals = combined_costs.groupby("drug_name")["LLM_Raw_Count"].transform("sum")
    combined_costs["LLM_Share_Combined"] = (
        combined_costs["LLM_Raw_Count"] / llm_totals
    )
    combined_costs["Share_Diff_Abs"] = (
        combined_costs["LLM_Share_Combined"] - combined_costs["MEPS_Share"]
    ).abs()
    combined_costs["Weighted_Unit_Diff"] = (
        combined_costs["LLM_Share_Combined"] - combined_costs["MEPS_Share"]
    ) * combined_costs["NADAC_Per_Unit"]

    meps_totals = combined_costs.groupby("drug_name")["MEPS_count"].transform("sum")
    combined_costs["LLM_Simulated_Count"] = (
        meps_totals * combined_costs["LLM_Share_Combined"]
    )
    combined_costs["Cost_Diff"] = (
        combined_costs["LLM_Simulated_Count"] - combined_costs["MEPS_count"]
    ) * combined_costs["NADAC_Per_Unit"]

    combined_costs["Cost_Diff_Annual"] = combined_costs["Cost_Diff"] / 5
    combined_costs["MEPS_count_Annual"] = combined_costs["MEPS_count"] / 5
    combined_costs["MEPS_Cost_Annual"] = (
        combined_costs["MEPS_count_Annual"] * combined_costs["NADAC_Per_Unit"]
    )
    combined_costs["LLM_Cost_Annual"] = (
        combined_costs["LLM_Simulated_Count"] / 5
    ) * combined_costs["NADAC_Per_Unit"]
    combined_costs["Prescription_Volume"] = meps_totals / 5

    mask = combined_costs.duplicated(subset="drug_name", keep="first")
    combined_costs.loc[mask, "Prescription_Volume"] = None

    return combined_costs[COST_OUTPUT_COLUMNS]


args = parse_args()

product_prices = attach_nadac_prices_to_products(
    args.product_metadata_csv,
    args.nadac_csv,
    args.ndc_product_file,
    args.ndc_excluded_product_file,
    args.ndc_compounder_file,
)

llm_choice_counts, meps_quantities = load_cost_inputs(
    args.llm_choices_csv, args.meps_quantity_csv
)
llm_choice_counts, meps_quantities, product_prices = normalize_cost_inputs(
    llm_choice_counts, meps_quantities, product_prices
)
model_level_costs, processed_count, skipped_count = build_model_level_cost_output(
    llm_choice_counts, meps_quantities, product_prices
)
combined_costs = build_combined_cost_output(model_level_costs)
args.output_csv.parent.mkdir(parents=True, exist_ok=True)
combined_costs.to_csv(args.output_csv, index=False)

print(f"Product rows: {len(product_prices)}")
print(f"Products with NADAC price: {product_prices['NADAC Per Unit'].notna().sum()}")
print(f"Processed drug groups: {processed_count}")
print(f"Skipped drug groups: {skipped_count}")
print(f"Cost output rows: {len(combined_costs)}")
print(f"Output drug_name count: {combined_costs['drug_name'].nunique()}")
print(f"Cost_Diff_Annual sum: {combined_costs['Cost_Diff_Annual'].sum()}")
print(f"MEPS_Cost_Annual sum: {combined_costs['MEPS_Cost_Annual'].sum()}")
print(f"LLM_Cost_Annual sum: {combined_costs['LLM_Cost_Annual'].sum()}")
print(f"Cost output CSV: {args.output_csv}")
