import asyncio
import logging
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from f1.common.schema import LLMConfig, LLMProviderEnum, LLMMessageTextOnly, LLMRole
from f1.common.llm_provider import (
    LiteLLMProvider, LLMProviderFactory, OpenAIProvider, VolcanoProvider
)
from f1.common.logging_helper import configure_module_logging


def _make_litellm_response(content="Hello from LiteLLM"):
    """Helper to build a mock litellm response object"""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    choice.model_dump = MagicMock(return_value={
        "message": {"content": content},
        "finish_reason": "stop",
    })

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_empty_response():
    """Helper to build a mock empty response"""
    response = MagicMock()
    response.choices = []
    return response


def _make_none_content_response():
    """Helper to build a mock response with None content"""
    message = MagicMock()
    message.content = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    return response


class TestLiteLLMProviderSync(unittest.TestCase):
    """Tests for LiteLLMProvider synchronous chat_completion"""

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            api_key="test-key",
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            provider="litellm",
        )
        self.messages = [
            LLMMessageTextOnly(role=LLMRole.SYSTEM, content="You are a helpful assistant"),
            LLMMessageTextOnly(role=LLMRole.USER, content="Hello"),
        ]

    @patch("litellm.completion")
    def test_chat_completion(self, mock_completion):
        mock_completion.return_value = _make_litellm_response("Hi there!")
        provider = LiteLLMProvider(self.config)
        result = provider.chat_completion(self.messages)
        self.assertEqual(result, "Hi there!")
        mock_completion.assert_called_once()

    @patch("litellm.completion")
    def test_chat_completion_passes_api_key(self, mock_completion):
        mock_completion.return_value = _make_litellm_response("ok")
        provider = LiteLLMProvider(self.config)
        provider.chat_completion(self.messages)
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "test-key")

    @patch("litellm.completion")
    def test_base_url_not_leaked_to_litellm(self, mock_completion):
        """base_url must NOT be passed as api_base — litellm routes via model prefix
        base_url不应传递给litellm，litellm通过model前缀自动路由"""
        mock_completion.return_value = _make_litellm_response("ok")
        config_with_url = LLMConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            provider="litellm",
        )
        provider = LiteLLMProvider(config_with_url)
        provider.chat_completion(self.messages)
        call_kwargs = mock_completion.call_args[1]
        self.assertNotIn("api_base", call_kwargs)

    @patch("litellm.completion")
    def test_chat_completion_empty_response(self, mock_completion):
        """Empty choices list should return empty string 空响应返回空字符串"""
        mock_completion.return_value = _make_empty_response()
        provider = LiteLLMProvider(self.config)
        result = provider.chat_completion(self.messages)
        self.assertEqual(result, "")

    @patch("litellm.completion")
    def test_chat_completion_none_content(self, mock_completion):
        """None content in message should return empty string"""
        mock_completion.return_value = _make_none_content_response()
        provider = LiteLLMProvider(self.config)
        result = provider.chat_completion(self.messages)
        self.assertEqual(result, "")


class TestLiteLLMProviderAsync(unittest.TestCase):
    """Tests for LiteLLMProvider async_chat_completion"""

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            api_key="test-key",
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            provider="litellm",
        )
        self.messages = [
            LLMMessageTextOnly(role=LLMRole.USER, content="Hello"),
        ]

    @patch("litellm.acompletion", new_callable=AsyncMock)
    def test_async_chat_completion(self, mock_acompletion):
        mock_acompletion.return_value = _make_litellm_response("Async hi!")
        provider = LiteLLMProvider(self.config)
        result = asyncio.run(provider.async_chat_completion(self.messages))
        self.assertEqual(result, "Async hi!")
        mock_acompletion.assert_called_once()

    @patch("litellm.acompletion", new_callable=AsyncMock)
    def test_async_chat_completion_empty_response(self, mock_acompletion):
        mock_acompletion.return_value = _make_empty_response()
        provider = LiteLLMProvider(self.config)
        result = asyncio.run(provider.async_chat_completion(self.messages))
        self.assertEqual(result, "")


class TestLLMProviderFactory(unittest.TestCase):
    """Tests for factory provider selection via provider field"""

    def test_default_provider_is_openai_compatible(self):
        """No provider specified → defaults to openai-compatible → OpenAIProvider"""
        config = LLMConfig(base_url="https://api.openai.com/v1", api_key="k", model="gpt-4o")
        provider = LLMProviderFactory.create_instance(config)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_litellm_provider(self):
        config = LLMConfig(api_key="k", model="bedrock/model", provider="litellm")
        provider = LLMProviderFactory.create_instance(config)
        self.assertIsInstance(provider, LiteLLMProvider)

    def test_volcano_provider(self):
        config = LLMConfig(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key="k", model="m", provider="volcano")
        provider = LLMProviderFactory.create_instance(config)
        self.assertIsInstance(provider, VolcanoProvider)

    def test_openai_compatible_provider(self):
        config = LLMConfig(base_url="https://api.openai.com/v1", api_key="k", model="gpt-4o", provider="openai-compatible")
        provider = LLMProviderFactory.create_instance(config)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_invalid_provider_rejected_by_pydantic(self):
        """Invalid provider value is rejected at LLMConfig construction time"""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            LLMConfig(api_key="k", model="m", provider="unknown_provider")

    def test_factory_passes_streaming_keys(self):
        config = LLMConfig(api_key="k", model="bedrock/model", provider="litellm")
        keys = ["usage"]
        provider = LLMProviderFactory.create_instance(config, streaming_monitored_trunk_keys=keys)
        self.assertIsInstance(provider, LiteLLMProvider)
        self.assertEqual(provider.streaming_monitored_trunk_keys, keys)


class TestLiteLLMProviderIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests with real AWS Bedrock API calls via LiteLLM
    真实调用AWS Bedrock API的集成测试，验证端到端流程"""

    def setUp(self):
        configure_module_logging(level=logging.INFO)
        from f1 import config
        self.llm_config = LLMConfig(**config['aws_example'])
        self.messages = [
            LLMMessageTextOnly(role=LLMRole.USER, content="Reply with exactly: hello world"),
        ]

    def test_sync_chat_completion(self):
        """Test sync call via factory → LiteLLMProvider → Bedrock"""
        provider = LLMProviderFactory.create_instance(self.llm_config)
        self.assertIsInstance(provider, LiteLLMProvider)
        resp = provider.chat_completion(self.messages)
        print(f"[sync] {resp}")
        self.assertTrue(resp)

    async def test_async_chat_completion(self):
        """Test async call via factory → LiteLLMProvider → Bedrock"""
        provider = LLMProviderFactory.create_instance(self.llm_config)
        resp = await provider.async_chat_completion(self.messages)
        print(f"[async] {resp}")
        self.assertTrue(resp)

    async def test_streaming_chat_completion(self):
        """Test streaming call via factory → LiteLLMProvider → Bedrock"""
        provider = LLMProviderFactory.create_instance(self.llm_config)
        chunks = []
        async for chunk in provider.chat_completion_stream(self.messages):
            chunks.append(chunk)
        full_response = "".join(chunks)
        print(f"[stream] {full_response}")
        self.assertTrue(full_response)
        self.assertTrue(len(chunks) > 0)


class TestLLMConfigProviderField(unittest.TestCase):
    """Tests for LLMConfig provider field defaults"""

    def test_provider_defaults_to_openai_compatible(self):
        config = LLMConfig(api_key="k", model="m")
        self.assertEqual(config.provider, LLMProviderEnum.OPENAI_COMPATIBLE)

    def test_base_url_defaults_to_empty(self):
        config = LLMConfig(api_key="k", model="m")
        self.assertEqual(config.base_url, "")

    def test_existing_config_without_provider_field(self):
        """Existing configs without provider field get openai-compatible by default"""
        config = LLMConfig(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="k",
            model="m",
        )
        self.assertEqual(config.provider, LLMProviderEnum.OPENAI_COMPATIBLE)


if __name__ == "__main__":
    unittest.main()
