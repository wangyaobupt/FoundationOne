import asyncio
import unittest
from unittest import mock
from typing import AsyncGenerator

from f1 import config
from f1.common.llm_chat_stream_agent import HelloWorldChatStreamAgent
from f1.common.schema import LLMConfig, LLMMessageTextOnly, LLMRole
from f1.common.logging_helper import configure_module_logging


class TestHelloWorldChatStreamAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        configure_module_logging(['f1'])
        # Create mock LLMConfig for testing
        self.llm_config = LLMConfig(**config["chatbot-llm"])

        # Test agent name
        self.agent_name = "TestBot"

    async def test_behavior(self):
        agent = HelloWorldChatStreamAgent(llm_config=self.llm_config, name=self.agent_name)

        user_msg = LLMMessageTextOnly(
            role=LLMRole.USER, content="Hello, agent! What's your name?"
        )
        response_chunks = list()
        async for chunk in agent.async_run(user_msg):
            if not chunk:
                print("EMPTY")
            else:
                print(chunk)
            response_chunks.append(chunk)

        self.assertEqual("".join(response_chunks), agent.messages_history[-1].get_serialized_content())
        self.assertEqual(user_msg, agent.messages_history[-2])

    async def test_run_with_history(self):
        agent = HelloWorldChatStreamAgent(llm_config=self.llm_config, name="Alice")
        first_user_msg = LLMMessageTextOnly(
                role=LLMRole.USER, content="Hello, agent! What's your name?"
            )
        first_response_msg = LLMMessageTextOnly(
                role=LLMRole.ASSISTANT, content="I am Alice."
            )
        agent.messages_history.extend(
            [first_user_msg, first_response_msg]
        )

        user_msg = LLMMessageTextOnly(
            role=LLMRole.USER, content="Can you tell the origin of your name?"
        )
        response_chunks = list()
        async for chunk in agent.async_run(user_msg):
            if not chunk:
                print("EMPTY")
            else:
                print(chunk)
            response_chunks.append(chunk)

        print("".join(response_chunks))

        self.assertEqual("".join(response_chunks), agent.messages_history[-1].get_serialized_content())
        self.assertEqual(user_msg, agent.messages_history[-2])
        self.assertEqual(first_response_msg, agent.messages_history[-3])
        self.assertEqual(first_user_msg, agent.messages_history[-4])
        self.assertTrue(agent.messages_history[-5].role == LLMRole.SYSTEM)



if __name__ == "__main__":
    unittest.main()