from typing import Any, Optional

from pydantic import BaseModel

from f1.common.llm_chat_agent import LLMChatAgent
from f1.common.llm_response_helper import LLMResponseHelper
from f1.common.schema import LLMMessageTextOnly, LLMRole, AbstractLLMMessage, LLMMessageVisual


class SimpleSchema(BaseModel):
    name: str


class HelloWorldChatAgent(LLMChatAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is_visual=False)

    async def async_run(self, name: str) -> Any:
        return await super().async_run(name=name)

    def build_system_msg(self, name: str) -> Optional[LLMMessageTextOnly]:
        return LLMMessageTextOnly(role=LLMRole.SYSTEM, content=f"As an personal assistant, your name is \'{name}\'")

    def build_init_user_msg(self, **kwargs) -> LLMMessageTextOnly:
        return LLMMessageTextOnly(role=LLMRole.USER,
                                  content=f"What's your name? Reply in JSON format, the schema of data model is "
                                          f"dumped by using pydantic.BaseModel.model_json_schema as be"
                                          f"low: ```{SimpleSchema.model_json_schema()}```.")

    def validate(self, response_str: str, **kwargs) -> (bool, SimpleSchema):
        is_succ, ret_obj_or_failed_reason = LLMResponseHelper.convert_LLM_json_response_to_obj(
            llm_response_str=response_str,
            target_class=SimpleSchema
        )

        if not is_succ:
            failed_reason = ret_obj_or_failed_reason
            return False, failed_reason

        ret_obj: SimpleSchema = ret_obj_or_failed_reason
        if ret_obj.name != kwargs['name']:
            return False, f"Name in your response is NOT the expected name as I told you: {kwargs}"

        return True, ret_obj

    def build_next_user_msg(self, failed_reason_or_supplement_info: str, **kwargs) -> LLMMessageTextOnly:
        return LLMMessageTextOnly(role=LLMRole.USER,
                                  content=f"Please correct your answer according to the following comments: {failed_reason_or_supplement_info}")

class HelloWorldVisualChatAgent(LLMChatAgent):
    async def async_run(self, **kwargs) -> Any:
        return await super().async_run(**kwargs)

    def build_system_msg(self, **kwargs) -> Optional[LLMMessageVisual]:
        return LLMMessageVisual(role=LLMRole.SYSTEM,
                                content=[LLMMessageVisual.TextContent(
                                    text="You are an assistant "
                                         "who need to read my image and provide description of that image")])

    def build_init_user_msg(self, image_url: str) -> AbstractLLMMessage:

        return LLMMessageVisual(role=LLMRole.USER,
                                    content=[
                                        LLMMessageVisual.TextContent(text="What's in the picture?"),
                                        LLMMessageVisual.ImageContent(image_url=LLMMessageVisual.ImageContent.ImageURL(
                                            url=image_url
                                        ))
                                    ])

    def validate(self, response_str: str, **kwargs) -> (bool, Any):
        return True, response_str

    def build_next_user_msg(self, failed_reason_or_supplement_info: str, **kwargs) -> AbstractLLMMessage:
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is_visual=True)



class BaichuanResponseModel(BaseModel):
    answer: str
    evidence_list: Optional[list[dict]] = None


class BaichuanEvidenceChatAgent(LLMChatAgent):
    async def async_run(self, prompt: str) -> Any:
        return await super().async_run(prompt=prompt)

    def build_system_msg(self, **kwargs) -> Optional[LLMMessageTextOnly]:
        return None

    def build_init_user_msg(self, prompt: str) -> LLMMessageTextOnly:
        return LLMMessageTextOnly(role=LLMRole.USER,
                                  content=prompt)

    def validate(self, response_str: str, response_object: Optional[dict] = None,
        **kwargs) -> (bool, BaichuanResponseModel):
        return True, BaichuanResponseModel(answer=response_str,
                                           evidence_list=response_object.get('grounding', {}).get('evidence') if response_object else None)


    def build_next_user_msg(self, failed_reason_or_supplement_info: str, **kwargs) -> AbstractLLMMessage:
        raise NotImplementedError()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is_visual=False)
