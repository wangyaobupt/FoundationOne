# Foundation One (f1)

Inspired by Issac Asimov's Foundation series, Foundation One aims to be the base camp of all our following explorations. It offers base infrastructure, shared services, and common libraries to accelerate development and ensure consistency across our projects.

## Install as a library

- Requires Python 3.12 (Conda recommended)

Conda setup (recommended):

```bash
# Create and activate a clean Conda env with Python 3.12
conda create -n f1 python=3.12 -y
conda activate f1

# Install pinned dependencies from this environment snapshot (pip-friendly pins)
pip install -r requirements.txt

# Install f1 in editable mode
pip install -e .
```

To build distributables locally:

```bash
python setup.py sdist bdist_wheel
```

## Configuration (mandatory)

f1 requires a YAML config file at import time. Importing `f1` resolves the config path as follows:
- If environment variable `F1_CONFIG` is set, use that path.
- Otherwise expect a file at `./conf/config.yaml` (relative to your working directory).
- If neither exists, importing `f1` raises `FileNotFoundError` with instructions.

You may also load a config explicitly if needed in scripts:

```python
from f1 import load_config
cfg = load_config("/absolute/or/relative/path/to/config.yaml")
```

Example YAML (see `conf/config_example.yaml`):

```yaml
chatbot-llm:
  base_url: https://ark.cn-beijing.volces.com/api/v3
  api_key: EXAMPLE_API_KEY
  model: EXAMPLE_MODEL_ID
```

Each config entry maps to an `LLMConfig` object with these fields:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `api_key` | Yes | — | API key for the provider |
| `model` | Yes | — | Model identifier (e.g. `gpt-4o`, `bedrock/anthropic.claude-haiku-4-5-20251001-v1:0`) |
| `base_url` | No | `""` | API endpoint URL. Required for OpenAI-compatible and Volcano providers |
| `provider` | No | `openai-compatible` | Provider selection. See [Providers](#providers) below |

Environment overrides: any YAML leaf can be overridden via environment variables using the pattern `f1_cfg.<dotted.path>`. For example:

```bash
export f1_cfg.chatbot-llm.api_key=REAL_KEY
export f1_cfg.chatbot-llm.model=REAL_MODEL
```

## Providers

The `provider` field in `LLMConfig` controls which backend is used. It accepts values from the `LLMProviderEnum`:

| Provider value | Class | When to use |
|---------------|-------|-------------|
| `openai-compatible` | `OpenAIProvider` | **Default.** Any endpoint that follows the OpenAI API format — OpenAI, Volcano Engine, Baichuan, DeepSeek, local vLLM, etc. |
| `volcano` | `VolcanoProvider` | Volcano Engine endpoints via the dedicated `volcenginesdkarkruntime` SDK |
| `litellm` | `LiteLLMProvider` | AWS Bedrock, Azure, and 100+ other backends via [LiteLLM](https://github.com/BerriAI/litellm) |

### Config examples

**OpenAI-compatible** (default — no `provider` field needed):

```yaml
chatbot-llm:
  base_url: https://ark.cn-beijing.volces.com/api/v3
  api_key: YOUR_API_KEY
  model: YOUR_MODEL_ID
```

**AWS Bedrock via LiteLLM:**

```yaml
aws-example:
  api_key: 'YOUR_AWS_BEDROCK_API_KEY'
  model: bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0
  provider: litellm
```

> Note: Bedrock API keys containing `=` (Base64 padding) must be wrapped in single quotes in YAML.

**Volcano Engine (dedicated SDK):**

```yaml
volcano-example:
  base_url: https://ark.cn-beijing.volces.com/api/v3
  api_key: YOUR_API_KEY
  model: YOUR_MODEL_ID
  provider: volcano
```

### Using a provider in code

```python
from f1 import config as f1_config
from f1.common.schema import LLMConfig
from f1.common.llm_provider import LLMProviderFactory

# The factory reads the 'provider' field and returns the correct provider instance
llm_cfg = LLMConfig(**f1_config["aws-example"])
provider = LLMProviderFactory.create_instance(llm_cfg)

# Sync call
response = provider.chat_completion(messages=[...])

# Async call
response = await provider.async_chat_completion(messages=[...])

# Streaming call
async for chunk in provider.chat_completion_stream(messages=[...]):
    print(chunk, end="")
```

## Quickstart

### Text-only agent

```python
import asyncio
from f1.common.hello_world_chat_agent import HelloWorldChatAgent
from f1.common.schema import LLMConfig
from f1 import config as f1_config  # loaded at import based on F1_CONFIG or ./conf/config.yaml

async def main():
    llm_cfg = LLMConfig(**f1_config["chatbot-llm"])  # keys: base_url, api_key, model
    agent = HelloWorldChatAgent(llm_config=llm_cfg)
    result = await agent.async_run(name="Alice")
    print(result)

asyncio.run(main())
```

### Streaming agent

```python
import asyncio
from f1.common.llm_chat_stream_agent import HelloWorldChatStreamAgent
from f1.common.schema import LLMConfig, LLMMessageTextOnly, LLMRole
from f1 import config as f1_config

async def main():
    llm_cfg = LLMConfig(**f1_config["chatbot-llm"])  # or construct directly
    agent = HelloWorldChatStreamAgent(llm_config=llm_cfg, name="Alice", streaming_monitored_trunk_keys=["finish_reason"]) 
    user_msg = LLMMessageTextOnly(role=LLMRole.USER, content="Tell me a joke about Python.")
    async for chunk in agent.async_run(user_message=user_msg):
        print(chunk, end="", flush=True)
    print("\n-- done --")

asyncio.run(main())
```

### Logging

```python
import logging
from f1.common.logging_helper import configure_module_logging

configure_module_logging(module_names_list=["f1"], level=logging.INFO, log_dir="./log")
```

## Usage
To run the tests, execute the following command:
```bash
./run_tests.sh
```

## Project layout

- src/f1: library code (agents, providers, schemas, helpers)
- conf/: example and local configs
- resources/: sample assets
- test/: example tests (some require valid API credentials)

## Notes

- OpenAI SDK, Volcano Engine SDK, and LiteLLM are all installed via `requirements.txt`.
- If `F1_CONFIG` is set but points to a non-existent file, import will raise with a clear message.
- Keep API keys out of VCS; set them via environment variables or secret managers.
