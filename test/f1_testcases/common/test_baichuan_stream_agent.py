import asyncio
import json
import unittest
from unittest import mock
from typing import AsyncGenerator

from f1 import config
from f1.common.llm_chat_stream_agent import HelloWorldChatStreamAgent, LLMChatStreamAgent
from f1.common.schema import LLMConfig, LLMMessageTextOnly, LLMRole
from f1.common.logging_helper import configure_module_logging


class TestHelloWorldChatStreamAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        configure_module_logging(['f1'])
        # Create mock LLMConfig for testing
        self.llm_config = LLMConfig(**config["baichuan"])

    async def test_behavior(self):
        agent = LLMChatStreamAgent(llm_config=self.llm_config,
                                   system_msg_list=[],
                                   streaming_monitored_trunk_keys=["grounding"])

        user_msg = LLMMessageTextOnly(
            role=LLMRole.USER, content="Hello, agent! 对于diabetes mellitus (disease)患者，使用Fosinopril治疗的适应症是什么，是否存在量表或其他机器学习模型基于患者特征预测是否适用于使用Fosinopril治疗？"
        )
        response_chunks = list()
        async for chunk in agent.async_run(user_msg):
            if not chunk:
                print("EMPTY")
            else:
                print(chunk)
            response_chunks.append(chunk)

        self.assertEqual("".join(response_chunks), agent.messages_history[-1].get_serialized_content())

        trunk_meta_dict = agent.get_streaming_trunk_meta_dict()
        for monitored_key, meta_list in trunk_meta_dict.items():
            print(f"{monitored_key}\t{json.dumps(meta_list, ensure_ascii=False)}")

        grounding_list = trunk_meta_dict.get("grounding")
        self.assertTrue(len(grounding_list) >= 1)
        for evidence in grounding_list[0]['grounding']['evidence']:
            print(json.dumps(evidence, ensure_ascii=False))




if __name__ == "__main__":
    unittest.main()