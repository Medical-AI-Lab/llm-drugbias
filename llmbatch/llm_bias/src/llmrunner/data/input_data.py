
import pandas as pd
import base64
import ast

class InputData:
    @property
    def has_image_list(self) -> bool:
        return self.image_path_list is not None and len(self.image_path_list) > 0
    
    def __init__(self, case_no: str, input_text: str, image_path_list: list = None):
        self.case_no = case_no
        self.input_text = input_text
        self.image_path_list = image_path_list
    
    def get_encoded_images(self) -> list[str]:
        if not self.has_image_list:
            return []
        ret_val = []
        for image in self.image_path_list:
            with open(image, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                ret_val.append(encoded_string)
        return ret_val

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_str

def get_image_base64_list(image_paths: list) -> list[str]:
    base64_list = []
    for image_path in image_paths:
        if image_path:
            base64_str = encode_image_to_base64(image_path)
            base64_list.append(base64_str)
        else:
            base64_list.append(None)
    return base64_list

def read_src_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, engine='openpyxl')
    cols = df.columns.tolist()
    is_vision = "Text for Vision model" in cols
    use_cols = ["Case", "user_prompt"] if not is_vision else ["Case", "user_prompt", "imgpath"]
    df = df[use_cols]
    return df, is_vision

def get_input_data_only_text(df: pd.DataFrame) -> list[InputData]:
    input_data_list = []
    for case_no, input_text in zip(df["Case"], df["user_prompt"]):
        input_data = InputData(case_no=case_no, input_text=input_text)
        input_data_list.append(input_data)
    return input_data_list

def get_input_data_with_image(df: pd.DataFrame) -> list[InputData]:
    input_data_list = []    
    df["imgpath"] = df["imgpath"].apply(lambda x: ast.literal_eval(x))
    for case_no, input_text, img_path_list in zip(df["Case"], df["user_prompt"], df["imgpath"]):
        input_data = InputData(case_no, input_text, img_path_list)
        input_data_list.append(input_data)
    return input_data_list

def get_input_data_from_excel(file_path: str) -> list[InputData]:
    df, is_vision = read_src_data(file_path)
    if is_vision:
        return get_input_data_with_image(df), is_vision
    else:
        return get_input_data_only_text(df), is_vision
