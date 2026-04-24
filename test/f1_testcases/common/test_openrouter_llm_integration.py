import unittest

from f1 import config
from f1.common.llm_provider import LLMProviderFactory, OpenAIProvider
from f1.common.schema import LLMConfig, LLMMessageTextOnly, LLMRole


class TestOpenRouterLLMIntegration(unittest.TestCase):
    """Live smoke test for the openrouter-llm entry in conf/config.yaml."""

    def test_openrouter_llm_chat_completion(self):
        raw_config = config.get("openrouter-llm")
        if not raw_config:
            self.skipTest("openrouter-llm is not configured")

        api_key = raw_config.get("api_key", "")
        if not api_key or api_key.startswith("EXAMPLE") or api_key.startswith("YOUR_"):
            self.skipTest("openrouter-llm api_key is not configured with a real key")

        llm_config = LLMConfig(**raw_config)
        provider = LLMProviderFactory.create_instance(llm_config)
        self.assertIsInstance(provider, OpenAIProvider)

        try:
            response = provider.chat_completion(
                messages=[
                    LLMMessageTextOnly(
                        role=LLMRole.SYSTEM,
                        content="You are a precise test endpoint. Follow the user's requested output exactly.",
                    ),
                    LLMMessageTextOnly(role=LLMRole.USER, content="Reply with exactly: pong"),
                ],
                max_tokens=8,
                temperature=0,
                extra_headers={
                    "HTTP-Referer": "https://github.com/FoundationOne",
                    "X-Title": "FoundationOne OpenRouter Smoke Test",
                },
            )
        except Exception as exc:
            body = getattr(exc, "body", None)
            if isinstance(body, dict):
                error = body.get("error", {})
                detail = error.get("message") or body
            else:
                detail = str(exc)
            raise self.failureException(
                f"openrouter-llm request failed: {type(exc).__name__}: {detail}"
            ) from None

        self.assertEqual(response.strip().lower().strip("\"'` ."), "pong")


if __name__ == "__main__":
    unittest.main()
