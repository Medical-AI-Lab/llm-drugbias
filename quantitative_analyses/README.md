# Quantitative analysis scripts

Last updated: 2026-06-08 (US Eastern Time)

## 1. MEPS preprocessing

Script: `MEPS_preprocessing.py`

Purpose: estimate product-level MEPS prescription quantity from prescription-level MEPS data.

Inputs:

- MEPS prescription CSV(s): must include `Proprietary Name`, `RXFRMUNT`, `RXQUANTY`
- Target product CSV: must include `Proprietary Name`

Example:

```bash
python MEPS_preprocessing.py \
  --meps-csv /path/to/meps_prescriptions_2019.csv \
  --meps-csv /path/to/meps_prescriptions_2020.csv \
  --target-products-csv /path/to/target_products.csv \
  --output-csv /path/to/meps_quantity_estimates.csv
```

Output: product-level MEPS quantity estimate CSV.

## 2. NADAC cost estimation

Script: `NADAC_costestimate.py`

Purpose: attach NADAC unit prices to products and estimate MEPS-versus-LLM cost differences.

Inputs:

- Product metadata CSV: includes product names and product-level metadata
- NADAC CSV: must include `ndc9`, `NADAC Per Unit`
- FDA NDC files: product directory, excluded product directory, compounder directory
- LLM first-choice CSV: must include `disease_drug`, `drug_name`, `model`, `proprietary_name_1_normalized`
- MEPS quantity CSV: must include `Proprietary Name`, `Estimated_Total_Quantity`

Example:

```bash
python NADAC_costestimate.py \
  --product-metadata-csv /path/to/product_metadata.csv \
  --nadac-csv /path/to/nadac_prices.csv \
  --ndc-product-file /path/to/ndc_product.txt \
  --ndc-excluded-product-file /path/to/ndc_excluded_products.txt \
  --ndc-compounder-file /path/to/ndc_compounders.txt \
  --llm-choices-csv /path/to/llm_choices.csv \
  --meps-quantity-csv /path/to/meps_quantity_estimates.csv \
  --output-csv /path/to/cost_difference_output.csv
```

Output: product-level cost difference CSV.

## 3. GEE model

Script: `GEE_model.py`

Purpose: run a binomial GEE model for first-choice drug selection.

Input:

- Analysis-ready GEE CSV
- Required columns: `is_most_frequent`, `user_prompt`, `age`, `sex`, `race_ethnicity`, `lgbtq_identity`, `income_status`, `model`, `is_brand`, `approval_rank`, `google_count`, `wiki_exists`, `MEPS_count`, `open_payments_total`, `word_count_normalized`

Example:

```bash
python GEE_model.py \
  --analysis-csv /path/to/gee_analysis_data.csv \
  --output-csv /path/to/gee_first_choice_odds_ratios.csv
```

Output: odds ratio CSV.
