import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, AsyncIterator

from f1.common.llm_provider import AbstractLLMProvider, LLMProviderFactory
from f1.common.schema import AbstractLLMMessage, LLMConfig, LLMMessageTextOnly, LLMRole


class LLMChatStreamAgent:
    logger: logging.Logger
    llm_provider: AbstractLLMProvider
    messages_history: list[AbstractLLMMessage]

    def __init__(self, llm_config: LLMConfig, system_msg_list:list[AbstractLLMMessage], **kwargs):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        if "streaming_monitored_trunk_keys" in kwargs:
            streaming_monitored_trunk_keys = kwargs.pop("streaming_monitored_trunk_keys")
            self.logger.info(f"LLMChatStreamAgent initialized with streaming_monitored_trunk_keys={streaming_monitored_trunk_keys}")
        else:
            streaming_monitored_trunk_keys = []
        self.llm_provider = LLMProviderFactory.create_instance(llm_config=llm_config, streaming_monitored_trunk_keys=streaming_monitored_trunk_keys)
        self.messages_history = list()
        if system_msg_list:
            self.messages_history.extend(system_msg_list)


    def handle_error(self,
                     user_message: AbstractLLMMessage,
                     chunk_list: list[str],
                     error: Exception) -> None:
        """
        define how to handle errors during the streaming process.

        Args:
            user_message: The user's message that caused the error
            chunk_list: List of response chunks received before the error
            error: The exception that occurred
        """
        self.messages_history.append(user_message)
        self.messages_history.append(LLMMessageTextOnly(
            role=LLMRole.ASSISTANT,
            content=f"在回答中发生了错误，错误信息：\'{str(error)}\', 出错前的答案片段如下：\n" + "".join(chunk_list)
        ))

    # implement an async run function, which intake a AbstractLLMMessage,
    # adding it to messages_history and send to LLM, returning response in stream mode
    async def async_run(self,
                        user_message: AbstractLLMMessage
                        ) -> AsyncIterator[str]:
        """
        Process a user message and stream the LLM's response.

        Args:
            user_message: The user's message to be sent to the LLM

        Yields:
            Response chunks from the LLM as they arrive
        """
        outgoing_msg_list: list[AbstractLLMMessage] = list()
        outgoing_msg_list.extend(self.messages_history)

        chunk_list: list[str] = list()

        outgoing_msg_list.append(user_message)

        try:
            # Stream the response
            async for chunk in self.llm_provider.chat_completion_stream(outgoing_msg_list):
                chunk_list.append(chunk)
                yield chunk

            llm_response: LLMMessageTextOnly = LLMMessageTextOnly(
                role=LLMRole.ASSISTANT,
                content="".join(chunk_list)
            )
            self.messages_history.append(user_message)
            # Save LLM response when success
            self.messages_history.append(llm_response)
        except Exception as e:
            self.logger.error(f"Error during streaming chat completion: {e}")
            self.handle_error(user_message, chunk_list, e)

    def get_streaming_trunk_meta_dict(self) -> dict:
        """
        Get the streaming trunk meta data collected during the last streaming operation.

        Returns:
            A dictionary containing the streaming trunk meta data.
        """
        return self.llm_provider.streaming_trunk_meta_dict


class HelloWorldChatStreamAgent(LLMChatStreamAgent):
    """
    A simple chat agent that interacts with LLM to get a name from the user.
    """

    def __init__(self, llm_config: LLMConfig, name: str, **kwargs):
        super().__init__(llm_config=llm_config,
                         system_msg_list=self.prepare_system_messages(name=name),
                         **kwargs)

    @staticmethod
    def prepare_system_messages(name: Optional[str] = None) -> list[AbstractLLMMessage]:
        system_msg = LLMMessageTextOnly(
            role=LLMRole.SYSTEM,
            content=f"As a personal assistant, your name is '{name}'"
        )
        return [system_msg]