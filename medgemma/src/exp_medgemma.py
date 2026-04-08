import pandas as pd
import json
import yaml
import math
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter
from typing import Dict, Any, List

import sys
import gc
import re

import torch
import random
import numpy as np
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True  # Suppress errors and continue
torch._dynamo.config.disable = True          # Disable Dynamo entirely (recommended)

def extract_json(text: str) -> str:
    """Extract only the JSON portion from a response."""
    text = text.strip()
    # Try parsing the text as-is first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # Extract { ... } block and retry (first { to last })
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # Return original text if extraction fails
    return text


def set_deterministic():
    """Configure full determinism for reproducibility."""
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model_and_tokenizer():
    """Load the MedGemma model"""
    set_deterministic()
    print("torch.cuda.device_count():", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-27b-it")
    tokenizer.padding_side = "left"  # decoder-only models require left-padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "google/medgemma-27b-it",
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across available GPUs
        attn_implementation="flash_attention_2"
    )
    model.eval()
    return model, tokenizer

def load_system_prompt(yaml_file: str = "../../data/drug_bias_prompt.yaml") -> str:
    """Load system prompt from YAML file"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['system_prompt']

def process_excel_with_medgemma(
    excel_file: str,
    column_name: str,
    output_file: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 512,
    temperature: float = 0,
    top_p: float = None,
    batch_size: int = 16
) -> None:
    model, tokenizer = load_model_and_tokenizer()
    first_device = next(model.parameters()).device

    df = pd.read_excel(excel_file)
    results: List[Dict[str, Any]] = []

    RELOAD_INTERVAL = 999999  # Reload not needed for 27B model
    num_batches = math.ceil(len(df) / batch_size)

    for batch_start in tqdm(range(0, len(df), batch_size), total=num_batches, desc="Batches"):
        # Periodically reload model
        if batch_start > 0 and batch_start % RELOAD_INTERVAL == 0:
            print(f"\n[INFO] Reloading model at item {batch_start}...")
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            model, tokenizer = load_model_and_tokenizer()
            first_device = next(model.parameters()).device
            print("[INFO] Model reloaded successfully")

        batch_df = df.iloc[batch_start:batch_start + batch_size]
        prompt_texts = [
            f"{system_prompt}\n{str(row[column_name])}"
            for _, row in batch_df.iterrows()
        ]

        t0 = perf_counter()

        try:
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(first_device)

            input_len = inputs['input_ids'].shape[1]

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )

            t1 = perf_counter()
            per_item_time = (t1 - t0) / len(batch_df)

            for i, (original_idx, row) in enumerate(batch_df.iterrows()):
                response = extract_json(tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True))
                results.append({
                    "row_index": original_idx + 1,
                    "Case": row["Case"],
                    "disease": row["disease"],
                    "drug_or_class": row["drug_or_class"],
                    "age": row["age"],
                    "race_ethnicity": row["race_ethnicity"],
                    "sex": row["sex"],
                    "lgbtq_identity": row["lgbtq_identity"],
                    "income_status": row["income_status"],
                    "user_prompt": str(row[column_name]),
                    "system_prompt": system_prompt,
                    "response": response,
                    "processing_time": per_item_time,
                    "model": "medgemma-27b-it",
                    "error": None
                })

        except Exception as e:
            print(f"ERROR in batch starting at item {batch_start}: {str(e)}")
            t1 = perf_counter()
            per_item_time = (t1 - t0) / len(batch_df)
            for original_idx, row in batch_df.iterrows():
                results.append({
                    "row_index": original_idx + 1,
                    "Case": row["Case"],
                    "disease": row["disease"],
                    "drug_or_class": row["drug_or_class"],
                    "age": row["age"],
                    "race_ethnicity": row["race_ethnicity"],
                    "sex": row["sex"],
                    "lgbtq_identity": row["lgbtq_identity"],
                    "income_status": row["income_status"],
                    "user_prompt": str(row[column_name]),
                    "system_prompt": system_prompt,
                    "response": "",
                    "processing_time": per_item_time,
                    "model": "medgemma-27b-it",
                    "error": str(e)
                })

    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"ERROR saving file: {str(e)}")
        return

    error_count = sum(1 for r in results if r['error'] is not None)
    print(f"Total: {len(results)}, Success: {len(results) - error_count}, Errors: {error_count}")

def main():
    """Example usage"""
    excel_file = "../../data/drug_bias_prompts.xlsx"
    column_name = "user_prompt"
    output_file = "../../output/medgemma_results.jsonl"
    system_prompt = load_system_prompt()
    
    
    process_excel_with_medgemma(
        excel_file=excel_file,
        column_name=column_name,
        output_file=output_file,
        system_prompt=system_prompt,
        max_tokens=512,
        temperature=0,
        top_p=None,
        batch_size=32
    )

if __name__ == "__main__":
    main()
