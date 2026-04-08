import pandas as pd
import json
import yaml
import requests
from time import perf_counter
from typing import Dict, Any, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:8080"
MAX_TOKENS = 1024
WORKERS = 32  # Should match --n_parallel on the server side


def load_system_prompt(yaml_file: str = "../../data/drug_bias_prompt.yaml") -> str:
    with open(yaml_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["system_prompt"]


def send_one(args_tuple) -> Dict[str, Any]:
    row, system_prompt, max_tokens = args_tuple
    user_prompt = str(row["user_prompt"])

    t0 = perf_counter()
    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 42,
                "response_format": {"type": "json_object"},
                "stop": ["<|user|>", "<|system|>"],
            },
            timeout=300,
        )
        resp.raise_for_status()
        t1 = perf_counter()
        res = resp.json()

        return {
            "row_index": int(row.name) + 1,
            **row.to_dict(),
            "response": res["choices"][0]["message"]["content"],
            "completion_tokens": res["usage"]["completion_tokens"],
            "prompt_tokens": res["usage"]["prompt_tokens"],
            "total_tokens": res["usage"]["total_tokens"],
            "processing_time": t1 - t0,
            "error": None,
        }
    except Exception as e:
        t1 = perf_counter()
        return {
            "row_index": int(row.name) + 1,
            **row.to_dict(),
            "response": None,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "processing_time": t1 - t0,
            "error": str(e),
        }


def process_excel(
    excel_file: str,
    output_file: str,
    system_prompt: str,
    max_tokens: int = MAX_TOKENS,
    workers: int = WORKERS,
) -> None:
    df = pd.read_excel(excel_file)
    results: List[Dict[str, Any]] = []
    errors = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(send_one, (row, system_prompt, max_tokens))
            for _, row in df.iterrows()
        ]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(fut.result())
            except Exception as e:
                errors += 1
                results.append({"error": str(e)})

    results.sort(key=lambda x: x.get("row_index", 0))

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(results)}, Successful: {len(results) - errors}, Errors: {errors}")


def main():
    excel_file = "../../data/drug_bias_prompts.xlsx"
    output_file = "../../output/llama4_results.jsonl"
    system_prompt = load_system_prompt("../../data/drug_bias_prompt.yaml")

    process_excel(
        excel_file=excel_file,
        output_file=output_file,
        system_prompt=system_prompt,
        max_tokens=MAX_TOKENS,
        workers=WORKERS,
    )


if __name__ == "__main__":
    main()
