"""Tests for provider usage tracking and template method refactor
测试provider usage tracking和模板方法重构

Covers:
- last_usage / accumulated_usage initialization
- _record_usage() extraction and accumulation
- _extract_content() consolidated logic
- Template method orchestration (chat_completion / async_chat_completion)
- Multiple calls accumulate correctly
- Edge cases: empty usage, None usage, empty response
"""
import asyncio
import logging
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

from f1.common.schema import LLMConfig, LLMProviderEnum, LLMMessageTextOnly, LLMRole
from f1.common.llm_provider import (
    LiteLLMProvider, LLMProviderFactory, OpenAIProvider,
)
from f1.common.logging_helper import configure_module_logging


def _make_response_with_usage(content="Hello", prompt_tokens=10, completion_tokens=5, total_tokens=15):
    """Helper to build a mock response with usage data
    构建包含usage数据的mock response"""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    choice.model_dump = MagicMock(return_value={
        "message": {"content": content},
        "finish_reason": "stop",
    })

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    usage.model_dump = MagicMock(return_value={
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    })

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_response_without_usage(content="Hello"):
    """Helper to build a mock response without usage data (usage=None)
    构建不包含usage数据的mock response"""
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
    response.usage = None
    return response


def _make_empty_response_with_usage(prompt_tokens=5, completion_tokens=0, total_tokens=5):
    """Helper to build an empty response (no choices) but with usage data
    构建空响应（无choices）但包含usage的mock response"""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens
    usage.model_dump = MagicMock(return_value={
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    })

    response = MagicMock()
    response.choices = []
    response.usage = usage
    return response


class TestUsageTrackingInit(unittest.TestCase):
    """Tests for usage tracking attribute initialization
    测试usage tracking属性初始化"""

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            api_key="test-key",
            model="bedrock/model",
            provider="litellm",
        )

    def test_last_usage_initialized_to_none(self):
        """last_usage should be None before any call
        调用前last_usage应为None"""
        provider = LiteLLMProvider(self.config)
        self.assertIsNone(provider.last_usage)

    def test_accumulated_usage_initialized_to_zeros(self):
        """accumulated_usage should be a zero dict before any call
        调用前accumulated_usage应为全零字典"""
        provider = LiteLLMProvider(self.config)
        self.assertEqual(provider.accumulated_usage, {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })

    def test_accumulated_usage_safe_to_read_without_guards(self):
        """accumulated_usage is never None — always a dict with known keys
        accumulated_usage永远不为None，始终是包含已知key的字典"""
        provider = LiteLLMProvider(self.config)
        # Should be safely accessible without try/except or None checks
        self.assertIsInstance(provider.accumulated_usage, dict)
        self.assertIn("llm_calls", provider.accumulated_usage)
        self.assertIn("prompt_tokens", provider.accumulated_usage)
        self.assertIn("completion_tokens", provider.accumulated_usage)
        self.assertIn("total_tokens", provider.accumulated_usage)

    def test_openai_provider_has_usage_attributes(self):
        """OpenAI provider should also have usage tracking attributes
        OpenAI provider也应有usage tracking属性"""
        config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4o",
        )
        provider = OpenAIProvider(config)
        self.assertIsNone(provider.last_usage)
        self.assertEqual(provider.accumulated_usage["llm_calls"], 0)


class TestRecordUsage(unittest.TestCase):
    """Tests for _record_usage() method
    测试_record_usage()方法"""

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            api_key="test-key",
            model="bedrock/model",
            provider="litellm",
        )
        self.provider = LiteLLMProvider(self.config)

    def test_record_usage_extracts_token_counts(self):
        """_record_usage should extract prompt/completion/total tokens
        _record_usage应提取prompt/completion/total tokens"""
        response = _make_response_with_usage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        self.provider._record_usage(response)
        self.assertEqual(self.provider.last_usage, {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        })

    def test_record_usage_increments_llm_calls(self):
        """Each _record_usage call increments llm_calls by 1
        每次_record_usage调用应使llm_calls加1"""
        response = _make_response_with_usage()
        self.provider._record_usage(response)
        self.assertEqual(self.provider.accumulated_usage["llm_calls"], 1)
        self.provider._record_usage(response)
        self.assertEqual(self.provider.accumulated_usage["llm_calls"], 2)

    def test_record_usage_accumulates_tokens(self):
        """Multiple calls should accumulate token counts
        多次调用应累加token计数"""
        r1 = _make_response_with_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        r2 = _make_response_with_usage(prompt_tokens=20, completion_tokens=8, total_tokens=28)
        self.provider._record_usage(r1)
        self.provider._record_usage(r2)
        self.assertEqual(self.provider.accumulated_usage["prompt_tokens"], 30)
        self.assertEqual(self.provider.accumulated_usage["completion_tokens"], 13)
        self.assertEqual(self.provider.accumulated_usage["total_tokens"], 43)
        self.assertEqual(self.provider.accumulated_usage["llm_calls"], 2)

    def test_record_usage_none_usage(self):
        """When response.usage is None, last_usage should be None but llm_calls still increments
        当response.usage为None时，last_usage应为None但llm_calls仍递增"""
        response = _make_response_without_usage()
        self.provider._record_usage(response)
        self.assertIsNone(self.provider.last_usage)
        self.assertEqual(self.provider.accumulated_usage["llm_calls"], 1)
        self.assertEqual(self.provider.accumulated_usage["prompt_tokens"], 0)

    def test_record_usage_no_usage_attr(self):
        """When response has no usage attribute at all, should handle gracefully
        当response完全没有usage属性时应优雅处理"""
        response = MagicMock(spec=[])  # no attributes at all
        response.choices = []
        # Manually ensure no 'usage' attr
        self.provider._record_usage(response)
        self.assertIsNone(self.provider.last_usage)
        self.assertEqual(self.provider.accumulated_usage["llm_calls"], 1)

    def test_last_usage_overwritten_each_call(self):
        """last_usage should reflect only the most recent call
        last_usage应只反映最近一次调用"""
        r1 = _make_response_with_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        r2 = _make_response_with_usage(prompt_tokens=99, completion_tokens=88, total_tokens=187)
        self.provider._record_usage(r1)
        self.provider._record_usage(r2)
        self.assertEqual(self.provider.last_usage["prompt_tokens"], 99)
        self.assertEqual(self.provider.last_usage["completion_tokens"], 88)


class TestExtractContent(unittest.TestCase):
    """Tests for _extract_content() method
    测试_extract_content()方法"""

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            api_key="test-key",
            model="bedrock/model",
            provider="litellm",
        )
        self.provider = LiteLLMProvider(self.config)

    def test_extract_content_normal(self):
        """Should extract and strip content from response
        应从response中提取并strip内容"""
        response = _make_response_with_usage(content="  Hello World  ")
        result = self.provider._extract_content(response)
        self.assertEqual(result, "Hello World")

    def test_extract_content_empty_choices(self):
        """Empty choices list returns empty string
        空choices列表返回空字符串"""
        response = _make_empty_response_with_usage()
        result = self.provider._extract_content(response)
        self.assertEqual(result, "")

    def test_extract_content_none_message_content(self):
        """None content in message returns empty string
        消息内容为None时返回空字符串"""
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"
        response = MagicMock()
        response.choices = [choice]
        result = self.provider._extract_content(response)
        self.assertEqual(result, "")

    def test_extract_content_sets_last_trunk_meta_dict(self):
        """_extract_content should set last_trunk_meta_dict from choice
        _extract_content应从choice设置last_trunk_meta_dict"""
        response = _make_response_with_usage(content="Hi")
        self.provider._extract_content(response)
        self.assertEqual(self.provider.last_trunk_meta_dict, {
            "message": {"content": "Hi"},
            "finish_reason": "stop",
        })


class TestTemplateMethodLiteLLMSync(unittest.TestCase):
    """Tests for template method orchestration — LiteLLM sync path
    测试模板方法编排 — LiteLLM同步路径"""

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

    @patch("litellm.completion")
    def test_chat_completion_records_usage(self, mock_completion):
        """chat_completion should record usage from response
        chat_completion应记录response中的usage"""
        mock_completion.return_value = _make_response_with_usage(
            content="Hi!", prompt_tokens=15, completion_tokens=3, total_tokens=18
        )
        provider = LiteLLMProvider(self.config)
        result = provider.chat_completion(self.messages)
        self.assertEqual(result, "Hi!")
        self.assertEqual(provider.last_usage["prompt_tokens"], 15)
        self.assertEqual(provider.last_usage["completion_tokens"], 3)
        self.assertEqual(provider.accumulated_usage["total_tokens"], 18)
        self.assertEqual(provider.accumulated_usage["llm_calls"], 1)

    @patch("litellm.completion")
    def test_multiple_sync_calls_accumulate(self, mock_completion):
        """Multiple sync calls should accumulate usage
        多次同步调用应累加usage"""
        mock_completion.side_effect = [
            _make_response_with_usage(content="r1", prompt_tokens=10, completion_tokens=5, total_tokens=15),
            _make_response_with_usage(content="r2", prompt_tokens=20, completion_tokens=8, total_tokens=28),
        ]
        provider = LiteLLMProvider(self.config)
        provider.chat_completion(self.messages)
        provider.chat_completion(self.messages)
        self.assertEqual(provider.accumulated_usage["llm_calls"], 2)
        self.assertEqual(provider.accumulated_usage["prompt_tokens"], 30)
        self.assertEqual(provider.accumulated_usage["completion_tokens"], 13)
        self.assertEqual(provider.accumulated_usage["total_tokens"], 43)

    @patch("litellm.completion")
    def test_empty_response_still_counts_call(self, mock_completion):
        """Empty response should still increment llm_calls and record usage tokens
        空响应仍应递增llm_calls并记录usage tokens"""
        mock_completion.return_value = _make_empty_response_with_usage(
            prompt_tokens=5, completion_tokens=0, total_tokens=5
        )
        provider = LiteLLMProvider(self.config)
        result = provider.chat_completion(self.messages)
        self.assertEqual(result, "")
        self.assertEqual(provider.accumulated_usage["llm_calls"], 1)
        self.assertEqual(provider.accumulated_usage["prompt_tokens"], 5)

    @patch("litellm.completion")
    def test_no_usage_in_response_graceful(self, mock_completion):
        """Response without usage data should not break tracking
        没有usage数据的response不应破坏tracking"""
        mock_completion.return_value = _make_response_without_usage(content="ok")
        provider = LiteLLMProvider(self.config)
        result = provider.chat_completion(self.messages)
        self.assertEqual(result, "ok")
        self.assertIsNone(provider.last_usage)
        self.assertEqual(provider.accumulated_usage["llm_calls"], 1)
        self.assertEqual(provider.accumulated_usage["prompt_tokens"], 0)


class TestTemplateMethodLiteLLMAsync(unittest.IsolatedAsyncioTestCase):
    """Tests for template method orchestration — LiteLLM async path
    测试模板方法编排 — LiteLLM异步路径"""

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
    async def test_async_chat_completion_records_usage(self, mock_acompletion):
        """async_chat_completion should record usage
        async_chat_completion应记录usage"""
        mock_acompletion.return_value = _make_response_with_usage(
            content="Async hi!", prompt_tokens=12, completion_tokens=4, total_tokens=16
        )
        provider = LiteLLMProvider(self.config)
        result = await provider.async_chat_completion(self.messages)
        self.assertEqual(result, "Async hi!")
        self.assertEqual(provider.last_usage["prompt_tokens"], 12)
        self.assertEqual(provider.accumulated_usage["total_tokens"], 16)
        self.assertEqual(provider.accumulated_usage["llm_calls"], 1)

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_multiple_async_calls_accumulate(self, mock_acompletion):
        """Multiple async calls should accumulate usage
        多次异步调用应累加usage"""
        mock_acompletion.side_effect = [
            _make_response_with_usage(content="r1", prompt_tokens=10, completion_tokens=5, total_tokens=15),
            _make_response_with_usage(content="r2", prompt_tokens=20, completion_tokens=10, total_tokens=30),
        ]
        provider = LiteLLMProvider(self.config)
        await provider.async_chat_completion(self.messages)
        await provider.async_chat_completion(self.messages)
        self.assertEqual(provider.accumulated_usage["llm_calls"], 2)
        self.assertEqual(provider.accumulated_usage["prompt_tokens"], 30)
        self.assertEqual(provider.accumulated_usage["total_tokens"], 45)


class TestTemplateMethodOpenAISync(unittest.TestCase):
    """Tests for template method orchestration — OpenAI provider
    测试模板方法编排 — OpenAI provider"""

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4o",
        )
        self.messages = [
            LLMMessageTextOnly(role=LLMRole.USER, content="Hello"),
        ]

    def test_openai_sync_records_usage(self):
        """OpenAI sync chat_completion should record usage via template method
        OpenAI同步chat_completion应通过模板方法记录usage"""
        provider = OpenAIProvider(self.config)
        mock_response = _make_response_with_usage(
            content="Hi from OpenAI", prompt_tokens=8, completion_tokens=3, total_tokens=11
        )
        provider.sync_client = MagicMock()
        provider.sync_client.chat.completions.create.return_value = mock_response
        result = provider.chat_completion(self.messages)
        self.assertEqual(result, "Hi from OpenAI")
        self.assertEqual(provider.last_usage["prompt_tokens"], 8)
        self.assertEqual(provider.accumulated_usage["total_tokens"], 11)
        self.assertEqual(provider.accumulated_usage["llm_calls"], 1)


class TestTemplateMethodOpenAIAsync(unittest.IsolatedAsyncioTestCase):
    """Tests for template method orchestration — OpenAI async
    测试模板方法编排 — OpenAI异步"""

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            model="gpt-4o",
        )
        self.messages = [
            LLMMessageTextOnly(role=LLMRole.USER, content="Hello"),
        ]

    async def test_openai_async_records_usage(self):
        """OpenAI async_chat_completion should record usage via template method
        OpenAI异步async_chat_completion应通过模板方法记录usage"""
        provider = OpenAIProvider(self.config)
        mock_response = _make_response_with_usage(
            content="Async from OpenAI", prompt_tokens=9, completion_tokens=4, total_tokens=13
        )
        provider.async_client = AsyncMock()
        provider.async_client.chat.completions.create.return_value = mock_response
        result = await provider.async_chat_completion(self.messages)
        self.assertEqual(result, "Async from OpenAI")
        self.assertEqual(provider.last_usage["prompt_tokens"], 9)
        self.assertEqual(provider.accumulated_usage["total_tokens"], 13)


class TestUsageTrackingWithAgentRetry(unittest.IsolatedAsyncioTestCase):
    """Tests that usage accumulates correctly across LLMChatAgent retry loops
    测试在LLMChatAgent重试循环中usage正确累加

    Per the design doc: LLMChatAgent reuses the same provider instance across retries,
    so accumulated_usage naturally captures all retry calls.
    """

    def setUp(self):
        configure_module_logging(level=logging.DEBUG)
        self.config = LLMConfig(
            api_key="test-key",
            model="bedrock/model",
            provider="litellm",
        )

    @patch("litellm.acompletion", new_callable=AsyncMock)
    async def test_retries_accumulate_on_same_provider(self, mock_acompletion):
        """Simulating agent retry: 3 calls on same provider should accumulate all usage
        模拟agent重试：同一provider上3次调用应累加所有usage"""
        mock_acompletion.side_effect = [
            _make_response_with_usage(content="bad1", prompt_tokens=10, completion_tokens=5, total_tokens=15),
            _make_response_with_usage(content="bad2", prompt_tokens=15, completion_tokens=8, total_tokens=23),
            _make_response_with_usage(content="good", prompt_tokens=20, completion_tokens=10, total_tokens=30),
        ]
        provider = LiteLLMProvider(self.config)

        # Simulate 3 retry calls (as LLMChatAgent.async_run would do)
        messages = [LLMMessageTextOnly(role=LLMRole.USER, content="attempt")]
        await provider.async_chat_completion(messages)
        await provider.async_chat_completion(messages)
        result = await provider.async_chat_completion(messages)

        self.assertEqual(result, "good")
        self.assertEqual(provider.accumulated_usage["llm_calls"], 3)
        self.assertEqual(provider.accumulated_usage["prompt_tokens"], 45)
        self.assertEqual(provider.accumulated_usage["completion_tokens"], 23)
        self.assertEqual(provider.accumulated_usage["total_tokens"], 68)
        # last_usage should reflect only the final call
        self.assertEqual(provider.last_usage["prompt_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
