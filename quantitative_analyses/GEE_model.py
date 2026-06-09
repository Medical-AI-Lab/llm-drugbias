#!/usr/bin/env python3
"""Run a GEE model for first-choice drug selection."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrices
from statsmodels.genmod import families
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE


DRUG_FEATURE_COLUMNS = [
    "is_brand",
    "approval_rank",
    "log_google_count",
    "wiki_exists",
    "log_MEPS_count",
    "log_open_payments",
    "word_count_normalized",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run a binomial GEE model for whether an LLM-selected product is "
            "the most frequent product for the target drug."
        )
    )
    parser.add_argument(
        "--analysis-csv",
        type=Path,
        required=True,
        metavar="/path/to/gee_analysis_data.csv",
        help=(
            "Analysis-ready CSV containing the GEE outcome, prompt group, "
            "patient attributes, model name, and product feature columns."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=script_dir / "gee_first_choice_odds_ratios.csv",
        metavar="/path/to/gee_first_choice_odds_ratios.csv",
        help="Path to write the rounded odds ratio table.",
    )
    return parser.parse_args()


def prepare_gee_dataframe(analysis_csv):
    gee_df = pd.read_csv(analysis_csv).copy()

    gee_df["is_most_frequent"] = gee_df["is_most_frequent"].astype(int)
    gee_df["is_brand"] = gee_df["is_brand"].astype(float)
    gee_df["wiki_exists"] = gee_df["wiki_exists"].astype(float)

    gee_df["log_google_count"] = np.log1p(gee_df["google_count"])
    gee_df["log_open_payments"] = np.log1p(gee_df["open_payments_total"])
    gee_df["log_MEPS_count"] = np.log1p(gee_df["MEPS_count"])
    gee_df["group_id"] = pd.factorize(gee_df["user_prompt"])[0]

    return gee_df


def fit_gee(formula, gee_df):
    y, x = dmatrices(formula, gee_df, return_type="dataframe")
    groups = gee_df.loc[x.index, "group_id"]
    model = GEE(
        endog=y,
        exog=x,
        groups=groups,
        family=families.Binomial(),
        cov_struct=Exchangeable(),
    )
    result = model.fit()
    print(f"Outcome: is_most_frequent")
    print(f"Samples used by Patsy/statsmodels: {x.shape[0]}")
    print(f"Explanatory variables: {x.shape[1]}")
    print("GEE family: binomial")
    print("Working correlation: exchangeable")
    print("Cluster variable: user_prompt")
    print(result.summary())
    return result


def build_odds_ratio_table(result):
    conf_int = result.conf_int()
    return pd.DataFrame(
        {
            "Variable": result.params.index,
            "coef": result.params.values,
            "OR": np.exp(result.params.values),
            "OR_lower": np.exp(conf_int[0].values),
            "OR_upper": np.exp(conf_int[1].values),
            "P>|z|": result.pvalues.values,
        }
    )


def print_odds_ratio_table(table):
    exclude_pattern = (
        r"Intercept|C\(model\)|C\(age|C\(sex|C\(race_ethnicity|"
        r"C\(lgbtq_identity|C\(income_status"
    )
    display_table = table[
        ~table["Variable"].str.contains(exclude_pattern, regex=True)
    ].copy()

    print("\n" + "=" * 70)
    print("First-choice drug-feature odds ratios")
    print("=" * 70)
    print(display_table.to_string(index=False))


def write_odds_ratio_csv(table, output_csv):
    csv_table = table.copy()
    numeric_columns = ["coef", "OR", "OR_lower", "OR_upper", "P>|z|"]
    csv_table[numeric_columns] = csv_table[numeric_columns].round(4)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_table.to_csv(output_csv, index=False)
    print(f"Odds ratio CSV: {output_csv}")


args = parse_args()
gee_df = prepare_gee_dataframe(args.analysis_csv)

formula_parts = [
    *DRUG_FEATURE_COLUMNS,
    "C(age)",
    "C(sex)",
    "C(race_ethnicity)",
    "C(lgbtq_identity)",
    "C(income_status)",
    "C(model)",
]
formula = "is_most_frequent ~ " + " + ".join(formula_parts)

print(f"Loaded analysis rows: {len(gee_df)}")
print("GEE formula:")
print(formula)

result = fit_gee(formula, gee_df)
odds_ratio_table = build_odds_ratio_table(result)
print_odds_ratio_table(odds_ratio_table)
write_odds_ratio_csv(odds_ratio_table, args.output_csv)
