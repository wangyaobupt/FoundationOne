import asyncio
import base64
import logging
import time
from typing import Any, Type, Optional

from abc import ABC, abstractmethod
from f1.common.schema import LLMConfig, LLMRole, AbstractLLMMessage, LLMMessageTextOnly, LLMMessageVisual
from f1.common.llm_provider import AbstractLLMProvider, LLMProviderFactory


class LLMChatAgent(ABC):
    """
        LLM Agent with Validation capability
    """
    logger: logging.Logger
    llm_provider: AbstractLLMProvider
    messages_history: list[AbstractLLMMessage]
    is_visual: bool # Is the agent support sending images to LLM?
    llm_message_type: Type[AbstractLLMMessage]

    count_of_conversation: int
    threshold_of_conversation: int
    count_of_retry_single_msg: int
    threshold_of_retry_single_msg: int

    def __init__(self, llm_config: LLMConfig,
                 is_visual: bool,
                 threshold_of_conversation: int = 3,
                 threshold_of_retry_single_msg: int = 1):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.llm_provider = LLMProviderFactory.create_instance(llm_config=llm_config)
        self.is_visual = is_visual
        if self.is_visual:
            self.llm_message_type = LLMMessageVisual
        else:
            self.llm_message_type = LLMMessageTextOnly
        self.logger.info(f"is_visual = {self.is_visual}, llm_message_type = {self.llm_message_type}")
        self.messages_history = list()
        self.count_of_conversation = 0
        self.threshold_of_conversation = threshold_of_conversation
        self.count_of_retry_single_msg = 0
        self.threshold_of_retry_single_msg = threshold_of_retry_single_msg

    async def async_run(self, **kwargs) -> Any:
        """
            执行与LLM的会话，使用校验器验证会话结果是否正确
        :param kwargs:
        :return:
            如果成功，返回调用者预期的数据类型
            如果失败，返回None
        """
        system_msg = self.build_system_msg(**kwargs)
        if system_msg is not None:
            assert (isinstance(system_msg, self.llm_message_type))
            system_msg_content = system_msg.get_serialized_content()
            if len(system_msg_content) > 1000:
                self.logger.debug(f"[Agent->LLM system msg: {system_msg_content[:1000]} ...")
            else:
                self.logger.debug(f"[Agent->LLM system msg: {system_msg_content}")
            self.messages_history.append(system_msg)
        init_user_msg = self.build_init_user_msg(**kwargs)
        assert (isinstance(init_user_msg, self.llm_message_type))
        next_user_msg = init_user_msg

        is_success: bool = False
        returned_obj = None

        self.count_of_retry_single_msg = 0
        while not is_success:
            response_str = ""
            response_object: dict = {}  # 除了返回的字符串，很多大模型可能会多返回证据列表等结构化数据，这里response_object完整保存大模型返回的choices[0]的内容
            try:
                response_str = await self.llm_provider.async_chat_completion(self.messages_history + [next_user_msg])
            except RuntimeError as e:
                self.logger.error(f"Failed to send user message: {e}")
                self.count_of_retry_single_msg += 1
                if self.count_of_retry_single_msg > self.threshold_of_retry_single_msg:
                    self.logger.error(f"Failed to send msg to LLM after retry {self.threshold_of_retry_single_msg} times, check service status")
                    break

            assert response_str
            self.count_of_retry_single_msg = 0 # clear counter after successfully send one message
            self.messages_history.extend([
                next_user_msg, LLMMessageTextOnly(role=LLMRole.ASSISTANT, content=response_str)
            ])

            is_success, detail = self.validate(response_str, response_object=self.llm_provider.last_trunk_meta_dict, **kwargs)
            if is_success:
                returned_obj = detail
            else:
                failed_reason = detail
                next_user_msg = self.build_next_user_msg(failed_reason, **kwargs)
                assert (isinstance(next_user_msg, self.llm_message_type))
                self.count_of_conversation += 1
                if self.count_of_conversation > self.threshold_of_conversation:
                    self.logger.error(f"LLM cannot correct itself after {self.count_of_conversation} rounds of conversation")
                    break

        if is_success:
            return returned_obj
        else:
            return None

    @abstractmethod
    def build_system_msg(self, **kwargs) -> Optional[AbstractLLMMessage]:
        raise NotImplementedError()

    @abstractmethod
    def build_init_user_msg(self, **kwargs) -> AbstractLLMMessage:
        raise NotImplementedError()

    @abstractmethod
    def validate(self, response_str: str, response_obj: Optional[dict] = None, **kwargs) -> (bool, Any):
        """
            如果LLM返回的消息符合要求，返回 （True， converted LLM response: object)
            反之，返回 （False, failed_reason: str)
        """
        raise NotImplementedError()

    @abstractmethod
    def build_next_user_msg(self, failed_reason_or_supplement_info: Any, **kwargs) -> AbstractLLMMessage:
        raise NotImplementedError()





