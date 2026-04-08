import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from src.llmrunner.data.input_data import InputData


class AnthropicRunner:
    def __init__(self, model_name: str = None, prompt :str = ""):
        self.__model_name = model_name
        self.__prompt = prompt
        self.__client = anthropic.Anthropic()
    
    # returns the message batch in JSON format(type: di)
    def run(self, input_data_list: list[InputData], index: int = 0) -> tuple[dict, str]:
        requests = self.__get_requests(input_data_list)
        message_batch = self.__client.messages.batches.create(requests=requests)
        return message_batch.model_dump(mode="json"), message_batch.id # returns dict, string
    
    def check_batch_status(self, batch_id: str) -> str:
        message_batch = self.__client.messages.batches.retrieve(batch_id)
        return message_batch.model_dump(mode="json"), message_batch.processing_status == "ended" # returns dict, bool
    
    def get_results(self, batch_id:str) -> list[dict]:
        results = self.__client.messages.batches.results(batch_id)
        retval = []
        for result in results:
            retval.append(result.model_dump_json())  # Convert to JSON string to unify output format with other LLM runners (note: check_batch_status returns a dict)
        return "\n".join(retval)
    
    def __get_requests(self, input_data_list: list[InputData]) -> list[Request]:
        requests = []
        for input_data in input_data_list:
            request = self.__get_single_request(input_data)
            requests.append(request)
        return requests
    
    def __get_single_request(self, input_data: InputData) -> Request:
        params = self.__get_params(input_data)
        custom_id = f"{input_data.case_no}"
        return Request(
            params=params,
            custom_id=custom_id
        )
    
    def __get_params(self, input_data: InputData) -> MessageCreateParamsNonStreaming:
        message = [{"role": "user", "content": input_data.input_text}]
        return MessageCreateParamsNonStreaming(
            model=self.__model_name,
            system=self.__prompt,
            messages=message,
            max_tokens=3000,
            temperature=0
        )
