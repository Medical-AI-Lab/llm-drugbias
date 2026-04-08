from openai import OpenAI
from src.llmrunner.data.input_data import InputData
import os
import json

class OpenAIRunner:
    def __init__(self, model_name: str = None, prompt: str = "", output_dir: str = None):
        self.__model_name = model_name
        self.__prompt = prompt
        self.__output_dir = output_dir
        self.__client = OpenAI()
    
    def run(self, input_data_list: list[InputData], index: int = 0) -> tuple[dict, str]:
        jsonl_contents = self.__get_jsonl_contents(input_data_list)
        output_path = self.__save_jsonl(jsonl_contents, self.__output_dir, index)
        file_id = self.__upload_batch_file(output_path)
        bathch_status = self.__create_batch(file_id)
        return bathch_status.model_dump(mode="json"), bathch_status.id

    def check_batch_status(self, batch_id: str):
        batch_status = self.__client.batches.retrieve(batch_id)
        return batch_status.model_dump(mode="json"), self.__is_completed(batch_status)
    
    def get_results(self, batch_id: str) -> list[dict]:
        batch_status, _ = self.check_batch_status(batch_id)
        file_id = batch_status["output_file_id"] if batch_status["output_file_id"] else batch_status["error_file_id"]
        if file_id is None:
            return "" # No results available
        file_response = self.__client.files.content(file_id)
        return file_response.text
    
    def __is_completed(self, batch_status: dict) -> bool:
        status = batch_status.status
        return (status == "failed") or (status == "completed") or (status == "cancelled")
    
    def __create_batch(self, file_id: str):
        batch_status = self.__client.batches.create(
            input_file_id = file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch_status
    
    
    def __upload_batch_file(self, file_path: str):
        batch_input_file = self.__client.files.create(
            file=open(file_path, 'rb'),
            purpose='batch'
        )
        return batch_input_file.model_dump(mode="json")["id"]
    
    def __save_jsonl(self, jsonl_contents: list, output_path: str, index: int = 0) -> str:
        output_path = os.path.join(self.__output_dir, f"requests_{index}.jsonl")
        
        with open(output_path, 'w', encoding="utf-8") as f:
            for content in jsonl_contents:
                f.write(json.dumps(content, ensure_ascii=False) + "\n")
        return output_path
            
    
    def __get_jsonl_contents(self, input_data_list: list[InputData]) -> str:
        requests = []
        for input_data in input_data_list:
            request = self.__get_single_request(input_data)
            requests.append(request)
        return requests
        
    def __get_single_request(self, input_data: InputData) -> dict:
        request = {}
        request["custom_id"] = f"{input_data.case_no}"
        request["method"] = "POST"
        request["url"] = f"/v1/chat/completions"
        request["body"] = self.__get_body(input_data)
        return request
    
    def __get_body(self, input_data: InputData) -> dict:
        body ={}
        body["model"] = self.__model_name
        body["messages"] = self.__get_messages(input_data)
        body["max_completion_tokens"] = 8000
        body["temperature"] = 0
        #body["response_format"] = { "type": "json_object" } #newly added for structured response
        return body
    
    def __get_messages(self, input_data: InputData) -> list[dict]:
        return [
            {"role": "system", "content": self.__prompt},
            {"role": "user", "content": input_data.input_text},
        ]
        