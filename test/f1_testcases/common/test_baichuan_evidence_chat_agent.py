import asyncio
import json
import logging
import unittest

from f1.common.hello_world_chat_agent import HelloWorldChatAgent, SimpleSchema, HelloWorldVisualChatAgent, \
    BaichuanEvidenceChatAgent, BaichuanResponseModel
from f1.common.llm_chat_agent import LLMChatAgent
from f1.common.llm_request_helper import LLMRequestHelper
from f1.common.schema import LLMConfig
from f1 import config
from f1.common.logging_helper import configure_module_logging


class MyTestCase(unittest.TestCase):
    def setUp(self):
        configure_module_logging(module_names_list=["f1"], level=logging.DEBUG)

    def test_agent(self):
        agent = BaichuanEvidenceChatAgent(llm_config=LLMConfig(**config['baichuan']))
        ret: BaichuanResponseModel = asyncio.run(agent.async_run(prompt="Hello, agent! 对于diabetes mellitus ("
                                                  "disease)患者，使用Fosinopril治疗的适应症是什么，是否存在量表或其他机器学习模型基于患者特征预测是否适用于使用Fosinopril治疗？"))
        print(ret.answer)
        for evidence in ret.evidence_list:
            print(json.dumps(evidence, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
