from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions
from src.llmrunner.data.input_data import InputData
import json
from pathlib import Path
from google.cloud import storage
import re
import os

class GeminiRunner:
    def __init__(self, model_name: str = None, prompt: str = "", output_dir: str = None, src_data_path: str = None):
        self.__model_name = model_name
        self.__prompt = prompt
        self.__genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
        self.__output_dir = output_dir if output_dir else Path.cwd()
        self.__storage_client = storage.Client()
        self.__backet_name = "llmdrugbias"  # TODO: Update bucket name as needed
        self.__file_name = os.path.splitext(os.path.basename(src_data_path))[0]

    def run(self, input_data_list: list[InputData], index: int = 0) -> tuple[dict, str]:
        blob = self.__make_and_upload_batch_file(input_data_list, index)
        input_uri = f"gs://{blob.bucket.name}/{blob.name}"
        job = self.__genai_client.batches.create(
            model=self.__model_name,
            src=input_uri,
            config=CreateBatchJobConfig(dest=f"gs://{self.__backet_name}/output")
        )
        return job.model_dump(mode="json"), job.name  # Return the job details and ID 
        
    def __make_and_upload_batch_file(self, input_data_list: list[InputData], index: int = 0) -> storage.Blob:
        batch_file = self.__make_batch_file_contents(input_data_list)
        saved_path = self.__save_jsonl(batch_file, self.__output_dir, index)
        blob = self.__get_backet().blob(f"input_request/{self.__file_name}_{index}.jsonl")
        if not blob.exists():
            blob.upload_from_filename(saved_path)
        return blob
    
    def __get_backet(self):
        return self.__storage_client.get_bucket(self.__backet_name)

    def check_batch_status(self, batch_id: str):
        job = self.__genai_client.batches.get(name=batch_id)
        return job.model_dump(mode="json"), self.__is_completed(job)

    def get_results(self, batch_id: str) -> str:
        job_name = batch_id
        latest_blob = self.__get_latest_blob(job_name)
        contents = latest_blob.download_as_text(encoding="utf-8")
        return contents
    
    def __get_job_create_time(self, job):
        job_create_time = job.create_time
        return job_create_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    def __get_latest_blob(self, job_name: str):
        job = self.__genai_client.batches.get(name=job_name)
        create_time = self.__get_job_create_time(job)
        bolbs = self.__get_backet().list_blobs(prefix=f"output")
        pattern = f"{create_time}.*predictions\.jsonl"
        files = [blob for blob in bolbs if  re.search(pattern, blob.name)]
        if len(files) != 1:
            raise ValueError(f"Expected one file matching pattern {pattern}, but found {len(files)} files.")
        return files[0]
    
    def __is_completed(self, job):
        completed_states = [
            "JOB_STETE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_FAILED",
            "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_SUCCEEDED"
        ]
        return job.state in completed_states
    
    def __make_batch_file_contents(self, input_data_list: list[InputData]) -> str:
        retval = []
        for input_data in input_data_list:
            batch_file = self.__make_batch_file_inner(input_data)
            retval.append(batch_file)
        return retval
    
    def __make_batch_file_inner(self, input_data: InputData) -> str:
        contens = self.__make_contents(input_data)
        return {
            "request":{
                "contents": contens,
                "system_instruction": self.__make_system_instruction(),
                "generation_config": {
                    "temperature": 0,
                    "max_output_tokens": 65536,
                    "response_mime_type": "application/json" 
                }
            }
        }
        
    def __make_system_instruction(self):
        return {
            "parts": [{"text": self.__prompt}],
        }
    
    def __make_contents(self, input_data: InputData):
        parts = [{"text": input_data.input_text}]
        return [{"role": "user", "parts": parts}]

    def __save_jsonl(self, jsonl_contents: list, output_path: str, index: int = 0) -> Path:
        output_path = Path(self.__output_dir) / f"requests_{index}.jsonl"
        with open(output_path, 'w', encoding="utf-8") as f:
            for content in jsonl_contents:
                f.write(json.dumps(content, ensure_ascii=False) + "\n")
        return output_path