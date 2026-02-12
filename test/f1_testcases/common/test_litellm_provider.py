import asyncio
import logging
import time
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
    def test_extra_params_passed_to_litellm(self, mock_completion):
        """extra_params from config should be forwarded to litellm call
        config中的extra_params应传递给litellm调用"""
        mock_completion.return_value = _make_litellm_response("ok")
        config = LLMConfig(
            api_key="test-key",
            model="bedrock/anthropic.claude-haiku-4-5-20251001-v1:0",
            provider="litellm",
            extra_params={"aws_region_name": "ap-southeast-1"},
        )
        provider = LiteLLMProvider(config)
        provider.chat_completion(self.messages)
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["aws_region_name"], "ap-southeast-1")

    @patch("litellm.completion")
    def test_extra_params_empty_by_default(self, mock_completion):
        """Without extra_params, no extra keys should appear in litellm call"""
        mock_completion.return_value = _make_litellm_response("ok")
        provider = LiteLLMProvider(self.config)
        provider.chat_completion(self.messages)
        call_kwargs = mock_completion.call_args[1]
        expected_keys = {"model", "messages", "api_key"}
        self.assertEqual(set(call_kwargs.keys()), expected_keys)

    @patch("litellm.completion")
    def test_kwargs_override_extra_params(self, mock_completion):
        """Code-level kwargs should override extra_params from config"""
        mock_completion.return_value = _make_litellm_response("ok")
        config = LLMConfig(
            api_key="test-key",
            model="bedrock/model",
            provider="litellm",
            extra_params={"max_tokens": 100},
        )
        provider = LiteLLMProvider(config)
        provider.chat_completion(self.messages, max_tokens=200)
        call_kwargs = mock_completion.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 200)

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
    def test_extra_params_passed_in_async(self, mock_acompletion):
        """extra_params should also be forwarded in async calls"""
        mock_acompletion.return_value = _make_litellm_response("ok")
        config = LLMConfig(
            api_key="test-key",
            model="bedrock/model",
            provider="litellm",
            extra_params={"aws_region_name": "us-west-2"},
        )
        provider = LiteLLMProvider(config)
        asyncio.run(provider.async_chat_completion(self.messages))
        call_kwargs = mock_acompletion.call_args[1]
        self.assertEqual(call_kwargs["aws_region_name"], "us-west-2")

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

    def test_extra_params_does_not_affect_openai_provider(self):
        """OpenAI provider should construct fine even with extra_params set"""
        config = LLMConfig(
            base_url="https://api.openai.com/v1", api_key="k", model="gpt-4o",
            provider="openai-compatible",
            extra_params={"some_param": "value"},
        )
        provider = LLMProviderFactory.create_instance(config)
        self.assertIsInstance(provider, OpenAIProvider)

    def test_factory_passes_streaming_keys(self):
        config = LLMConfig(api_key="k", model="bedrock/model", provider="litellm")
        keys = ["usage"]
        provider = LLMProviderFactory.create_instance(config, streaming_monitored_trunk_keys=keys)
        self.assertIsInstance(provider, LiteLLMProvider)
        self.assertEqual(provider.streaming_monitored_trunk_keys, keys)


class TestLiteLLMProviderIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests with real AWS Bedrock API calls via LiteLLM
    真实调用AWS Bedrock API的集成测试，验证端到端流程"""

    CONCURRENCY = 100

    def setUp(self):
        configure_module_logging(level=logging.INFO)
        from f1 import config
        self.llm_config = LLMConfig(**config['aws_example'])

    async def test_concurrent_async_completion(self):
        """Test 100 concurrent async calls via factory → LiteLLMProvider → Bedrock
        测试100个并发异步请求，验证extra_params(aws_region_name)端到端生效"""
        provider = LLMProviderFactory.create_instance(self.llm_config)
        self.assertIsInstance(provider, LiteLLMProvider)

        async def single_call(i: int) -> str:
            messages = [LLMMessageTextOnly(role=LLMRole.USER, content=f"Say hello #{i}")]
            return await provider.async_chat_completion(messages)

        start = time.perf_counter()
        tasks = [single_call(i) for i in range(self.CONCURRENCY)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.perf_counter() - start

        successes = [r for r in results if isinstance(r, str) and r]
        errors = [r for r in results if isinstance(r, Exception)]

        print(f"\n[concurrent] {len(successes)}/{self.CONCURRENCY} succeeded in {elapsed:.2f}s")
        if errors:
            print(f"[concurrent] {len(errors)} errors, first: {type(errors[0]).__name__}: {errors[0]}")

        # Allow some rate-limit failures, but majority should succeed
        self.assertGreater(len(successes), self.CONCURRENCY * 0.5,
                           f"Too many failures: {len(errors)}/{self.CONCURRENCY}")


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

    def test_extra_params_defaults_to_empty_dict(self):
        config = LLMConfig(api_key="k", model="m")
        self.assertEqual(config.extra_params, {})

    def test_extra_params_from_yaml_dict(self):
        """Simulate YAML input with extra_params containing multiple keys"""
        config = LLMConfig(
            api_key="k", model="m", provider="litellm",
            extra_params={"aws_region_name": "us-west-2", "timeout": 30},
        )
        self.assertEqual(config.extra_params["aws_region_name"], "us-west-2")
        self.assertEqual(config.extra_params["timeout"], 30)


class TestLiteLLMEnvVarAuth(unittest.TestCase):
    """Tests for LiteLLM provider AWS env var authentication
    LiteLLM provider AWS环境变量认证测试"""

    def test_litellm_provider_no_api_key(self):
        """LiteLLM provider should accept empty api_key
        LiteLLM provider应接受空api_key"""
        config = LLMConfig(model="bedrock/model", provider="litellm")
        self.assertEqual(config.api_key, "")
        self.assertEqual(config.provider, LLMProviderEnum.LITELLM)

    def test_litellm_provider_with_api_key(self):
        """LiteLLM provider should still work with explicit api_key
        LiteLLM provider仍应支持显式api_key"""
        config = LLMConfig(api_key="my-key", model="bedrock/model", provider="litellm")
        self.assertEqual(config.api_key, "my-key")

    def test_openai_provider_requires_api_key(self):
        """OpenAI-compatible provider must have api_key — ValidationError when missing
        OpenAI-compatible provider必须有api_key"""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError) as ctx:
            LLMConfig(model="gpt-4o")
        self.assertIn("api_key is required", str(ctx.exception))

    def test_volcano_provider_requires_api_key(self):
        """Volcano provider must have api_key — ValidationError when missing
        Volcano provider必须有api_key"""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError) as ctx:
            LLMConfig(model="m", provider="volcano")
        self.assertIn("api_key is required", str(ctx.exception))

    @patch("litellm.completion")
    def test_no_api_key_not_passed_to_litellm(self, mock_completion):
        """When api_key is empty, it should NOT be passed to litellm.completion()
        api_key为空时不应传递给litellm.completion()"""
        mock_completion.return_value = _make_litellm_response("ok")
        config = LLMConfig(model="bedrock/model", provider="litellm")
        # Set AWS env vars so _initialize_client doesn't raise
        with patch.dict("os.environ", {
            "AWS_ACCESS_KEY_ID": "fake-key",
            "AWS_SECRET_ACCESS_KEY": "fake-secret",
            "AWS_REGION_NAME": "us-east-1",
        }):
            provider = LiteLLMProvider(config)
        messages = [LLMMessageTextOnly(role=LLMRole.USER, content="Hi")]
        provider.chat_completion(messages)
        call_kwargs = mock_completion.call_args[1]
        self.assertNotIn("api_key", call_kwargs)

    def test_missing_aws_env_vars_raises_error(self):
        """No api_key + no extra_params creds + no env vars → ValueError
        无api_key、无extra_params凭证、无环境变量时应抛出ValueError"""
        config = LLMConfig(model="bedrock/model", provider="litellm")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                LiteLLMProvider(config)
        self.assertIn("Missing env vars", str(ctx.exception))
        self.assertIn("AWS_ACCESS_KEY_ID", str(ctx.exception))

    def test_aws_creds_in_extra_params_no_env_check(self):
        """AWS creds in extra_params should bypass env var check
        extra_params中提供AWS凭证时不检查环境变量"""
        config = LLMConfig(
            model="bedrock/model",
            provider="litellm",
            extra_params={
                "aws_access_key_id": "key-from-params",
                "aws_secret_access_key": "secret-from-params",
            },
        )
        # Should not raise even with no env vars
        with patch.dict("os.environ", {}, clear=True):
            provider = LiteLLMProvider(config)
        self.assertIsInstance(provider, LiteLLMProvider)


if __name__ == "__main__":
    unittest.main()
