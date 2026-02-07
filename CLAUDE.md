# CLAUDE.md — Foundation One (f1)

## Project Overview

Foundation One (f1) is a Python library providing base infrastructure and common utilities for building LLM-powered agents. It wraps the OpenAI-compatible API and Volcano Engine (volcengine) SDK behind a unified provider abstraction, and offers agent classes with built-in conversation loop, validation/retry, and streaming support.

## Tech Stack

- **Language:** Python 3.12 (3.10+ supported)
- **Package manager:** pip (Conda recommended for env isolation)
- **Key dependencies:** openai, volcengine-python-sdk[ark], pydantic v2, PyYAML
- **Build:** setuptools (`setup.py`), editable install via `pip install -e .`

## Project Layout

```
src/f1/                  # Library source (installed as package "f1")
  __init__.py            # Config loading at import time (YAML + env overrides)
  common/
    schema.py            # Pydantic models: LLMConfig, LLMRole, AbstractLLMMessage, LLMMessageTextOnly, LLMMessageVisual
    llm_provider.py      # AbstractLLMProvider, BaseProvider, OpenAIProvider, VolcanoProvider, LLMProviderFactory
    llm_chat_agent.py    # LLMChatAgent — abstract agent with conversation loop + validation/retry
    llm_chat_stream_agent.py  # LLMChatStreamAgent — streaming agent base class
    hello_world_chat_agent.py # Example agents: HelloWorldChatAgent, HelloWorldVisualChatAgent, BaichuanEvidenceChatAgent
    llm_response_helper.py    # JSON extraction and Pydantic parsing from LLM responses
    llm_request_helper.py     # Image-to-base64 conversion utility
    logging_helper.py         # Module-level logging config + execution time decorator
conf/                    # YAML config files (config.yaml is gitignored; config_example.yaml is checked in)
test/                    # Unit tests (unittest, some require live API keys)
build/                   # Build artifacts (gitignored content, may be stale)
```

## Configuration

- f1 loads config **at import time** from YAML.
- Resolution order: `F1_CONFIG` env var → `./conf/config.yaml` (relative to cwd).
- Any YAML leaf can be overridden via env vars: `f1_cfg.<dotted.path>=VALUE`.
- `conf/config.yaml` is gitignored (contains real API keys). Use `conf/config_example.yaml` as a template.

## Common Commands

```bash
# Install in editable mode
pip install -e .

# Run all tests
./run_tests.sh
# or equivalently:
PYTHONPATH=$(pwd)/src:$PYTHONPATH python -m unittest discover -s test -p "test_*.py"

# Build distributables
python setup.py sdist bdist_wheel
```

## Architecture Notes

- **Provider pattern:** `LLMProviderFactory.create_instance(llm_config)` auto-selects `VolcanoProvider` (if base_url matches `ark.*.volces.com`) or `OpenAIProvider`. Both extend `BaseProvider` which implements sync, async, and streaming chat completion via the OpenAI SDK interface.
- **Agent pattern:** Subclass `LLMChatAgent` and implement `build_init_user_msg()` and `validate()`. The base class handles the conversation loop: send message → get response → validate → retry with correction if invalid. `build_system_msg()` is optional (returns `None` by default).
- **Streaming agent:** `LLMChatStreamAgent` takes system messages at init and yields response chunks via `async_run()`. Supports monitored trunk keys for capturing metadata from stream chunks.
- **Message types:** `LLMMessageTextOnly` for text, `LLMMessageVisual` for text + images. Both serialize to OpenAI-compatible dict format via Pydantic's `model_dump()`.

## Code Conventions

- Bilingual comments (Chinese + English) are used throughout — preserve this style.
- Pydantic v2 (`BaseModel`, `model_validate_json`, `model_dump`) is used for all data models.
- Logging uses `logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")` pattern.
- Async-first design: primary agent entry points are `async_run()`.
- Tests use `unittest` (not pytest). Test discovery: `test/f1_testcases/**/test_*.py`.
