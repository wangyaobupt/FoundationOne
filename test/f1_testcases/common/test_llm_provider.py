import asyncio
import logging
import unittest

from f1 import config
from f1.common.schema import LLMConfig, LLMMessageTextOnly
from f1.common.llm_provider import OpenAIProvider
from f1.common.logging_helper import configure_module_logging


class MyTestCase(unittest.TestCase):
    def setUp(self):
        configure_module_logging(level=logging.INFO)

    def test_hello_world(self):
        provider = OpenAIProvider(LLMConfig(**config['chatbot-llm']))
        resp_content = provider.chat_completion(messages=[
            LLMMessageTextOnly(**{"role": "system", "content": "You are a helpful assistant"}),
            LLMMessageTextOnly(**{"role": "user", "content": "Hello"}),
        ])
        print(resp_content)
        self.assertTrue(resp_content)

    def test_async_hello_world(self):
        provider = OpenAIProvider(LLMConfig(**config['chatbot-llm']))
        resp_content = asyncio.run(provider.async_chat_completion(messages=[
            LLMMessageTextOnly(**{"role": "system", "content": "You are a helpful assistant"}),
            LLMMessageTextOnly(**{"role": "user", "content": "Hello"}),
        ]))
        print(resp_content)
        self.assertTrue(resp_content)

if __name__ == '__main__':
    unittest.main()
