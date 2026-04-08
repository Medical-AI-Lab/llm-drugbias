import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent  # llm_bias/
sys.path.insert(0, str(project_root))


from dotenv import load_dotenv
import os
from src.llmrunner.data.input_data import InputData, get_input_data_from_excel
from src.llmrunner.data.prompt import PromptType, Prompt
from src.llmrunner.runner.anthropic_runner import AnthropicRunner
from src.llmrunner.runner.openai_runner import OpenAIRunner
from src.llmrunner.runner.gemini_runner import GeminiRunner
import datetime
import json
import time
import argparse
import math


dot_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dot_env_path, override=True)


# # Google Cloud authentication setup
# google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# if google_credentials_path and not os.path.exists(google_credentials_path):
#     # Convert relative path to absolute if necessary
#     google_credentials_path = str(project_root / google_credentials_path)
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path


anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

anthropic_llm_model = os.getenv("ANTHROPIC_LLM_MODEL")
if not anthropic_llm_model:
    raise ValueError("ANTHROPIC_LLM_MODEL environment variable is not set.")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
openai_llm_model = os.getenv("OPENAI_LLM_MODEL")
if not openai_llm_model:
    raise ValueError("OPENAI_LLM_MODEL environment variable is not set.")

# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY environment variable is not set.")
gemini_llm_model = os.getenv("GEMINI_LLM_MODEL")
if not gemini_llm_model:
    raise ValueError("GEMINI_LLM_MODEL environment variable is not set.")


# TODO: Update paths as needed
SRC_DATA_PATH = "../../data/drug_bias_prompts.xlsx"
OUTPUT_DIR = "../../output/chatgpt/test"


def get_args():
    parser = argparse.ArgumentParser(description="LLM Runner")
    parser.add_argument("--srcdatapath", type=str, default=SRC_DATA_PATH, help="Path to the source data Excel file")
    parser.add_argument("--llm", type=str, choices=["anthropic", "gemini", "chatgpt"], default="chatgpt", help="LLM to use")
    parser.add_argument(
        "--prompttype",
        type=str,
        choices=["basic"],
        default="basic",
        help="Type of prompt to use"
    )
    parser.add_argument("--outputdir", type=str, default=OUTPUT_DIR, help="Directory to save output files")
    return parser.parse_args()

def print_info(message: str):
    date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{date_time}] {message}")

def get_llm_runner(llm: str, prompt_type: str, output_dir: str, src_data_path: str) -> object:
    prompt = Prompt.get_prompt(get_prompt(prompt_type))
    
    if llm.lower() == "chatgpt":
        return OpenAIRunner(
            model_name=openai_llm_model,
            prompt=prompt,
            output_dir=output_dir
        )
    elif llm.lower() == "anthropic":
        return AnthropicRunner(
            model_name=anthropic_llm_model,
            prompt=prompt,
        )
    elif llm.lower() == "gemini":
        return GeminiRunner(
            model_name=gemini_llm_model,
            prompt=prompt,
            output_dir=output_dir,
            src_data_path=src_data_path
        )
        
    raise ValueError(f"Unsupported LLM: {llm}. Currently only 'anthropic' or 'chatgpt' are supported.")

def get_prompt(prompt_type: str) -> PromptType:
    prompt_type_map = {
        "basic": PromptType.BASIC,
    }

    if prompt_type not in prompt_type_map:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Use one of {list(prompt_type_map.keys())}.")
    return prompt_type_map[prompt_type]


def save_status_to_json(batch_info: dict, output_dir: str, file_name: str = "batch_status.json"):
    file_path = os.path.join(output_dir, f"{file_name}")
    with open(file_path, 'w') as f:
        json.dump(batch_info, f, indent=4)

def save_results_to_jsonl(results: str, output_dir: str, file_name: str = "results.jsonl"):
    if results is None or results == "":
        print_info("No results to save.")
        return
    
    file_path = os.path.join(output_dir, f"{file_name}")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(results)
    print_info(f"Results saved to {file_path}")

def save_args_to_json(args: argparse.Namespace, output_dir: str, file_name: str = "args.json"):
    file_path = os.path.join(output_dir, f"{file_name}")
    with open(file_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print_info(f"Arguments saved to {file_path}")
    

        
if __name__ == "__main__":
    
    args = get_args()
    
    start_time = datetime.datetime.now()
    output_dir = os.path.join(args.outputdir, start_time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    
    input_data_list, is_vision = get_input_data_from_excel(args.srcdatapath)
    runner = get_llm_runner(args.llm, args.prompttype, output_dir, args.srcdatapath)
    
    save_args_to_json(args, output_dir)
    
    def split_list(lst, n):
        """Split a list into chunks of size n."""
        chank_size = math.ceil(len(lst) / n)
        for i in range(0, len(lst), chank_size):
            yield lst[i:i + chank_size]
    
    num_splits = 1 if not is_vision else 5
    for i, input_data_chunk in enumerate(split_list(input_data_list, num_splits)):
        batch_info, batch_id = runner.run(input_data_chunk, i+1)
        save_status_to_json(batch_info, output_dir, f"batch_info_{i+1}.json")
        
        check_count = 0
        while True:
            batch_status, is_completed = runner.check_batch_status(batch_id)
            save_status_to_json(batch_status, output_dir, f"latest_batch_status_{i+1}.json")
            if is_completed:
                print_info("Batch processing completed.")
                break
            else:
                if check_count % 10 == 0:
                    print_info(f"Batch processing is still ongoing. Check count: {check_count}")
                time.sleep(30)
            check_count += 1
        
        results = runner.get_results(batch_id)  # For Gemini, use: runner.get_results(batch_id, index=i+1)
        save_results_to_jsonl(results, output_dir, f"results_{i+1}.jsonl")