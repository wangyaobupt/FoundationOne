import json
import re
from typing import Optional, Type

from pydantic import BaseModel, ValidationError


class LLMResponseHelper:
    @staticmethod
    def extract_contents_from_response(llm_response_str, lang: str="json") -> list[str]:
        """
        Extract JSON or Markdown code block content from an LLM response.

        Args:
            llm_response_str (str): The full response string from an LLM.
            lang (str): json, python, markdown, etc. default is json

        Returns:
            a list of string
        """
        # Regular expressions for matching
        json_pattern = f"```{lang}(.*?)```"

        # Extract JSON blocks as a list of strings
        json_matches = [match.strip() for match in re.findall(json_pattern, llm_response_str, re.DOTALL|re.IGNORECASE)]
        return json_matches

    @staticmethod
    def extract_last_part_of_content(llm_response_str: str, lang: str="json") -> Optional[str]:
        str_list = LLMResponseHelper.extract_contents_from_response(llm_response_str, lang=lang)
        if len(str_list) > 0:
            return str_list[-1]
        else:
            return None

    @staticmethod
    def convert_LLM_json_response_to_obj(llm_response_str: str, target_class: Type[BaseModel] = None):
        """
            从LLM回复的文本抽取JSON，转为指定的Data Model
            Args:
                llm_response_str
                target_class: 目标model的类型，如果为None，则返回json.load的结果

            Returns:
                is_success: boolean
                failed_reason_or_obj: 如果is_success为True，这一项为返回的object；反之，这一项为失败原因字符串
        """
        json_text = LLMResponseHelper.extract_last_part_of_content(llm_response_str)
        if not isinstance(json_text, str):
            # return False, "No JSON text enclosed in ```json(.*?)``` found in response."
            json_text = llm_response_str # 如果LLM没有返回json文本，直接使用整个回复文本

        if target_class is None:
            try:
                ret_obj = json.loads(json_text)
                return True, ret_obj
            except json.JSONDecodeError as e:
                return False, f"We cannot decode object from your JSON text, the decode error is: {e}"

        # target_class is NOT None
        try:
            ret_obj = target_class.model_validate_json(json_text)
            return True, ret_obj
        except ValidationError as e:
            return False, f"The JSON object cannot be decoded correctly: {e}"

