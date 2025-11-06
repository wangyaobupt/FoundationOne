import asyncio
import logging
import unittest

from f1.common.hello_world_chat_agent import HelloWorldChatAgent, SimpleSchema, HelloWorldVisualChatAgent
from f1.common.llm_request_helper import LLMRequestHelper
from f1.common.schema import LLMConfig
from f1 import config
from f1.common.logging_helper import configure_module_logging


class MyTestCase(unittest.TestCase):
    def setUp(self):
        configure_module_logging(module_names_list=["f1"], level=logging.DEBUG)

    def test_hello_world_agent(self):
        agent = HelloWorldChatAgent(llm_config=LLMConfig(**config['patientSummary-llm']))
        ret = asyncio.run(agent.async_run(name="Alice"))
        self.assertTrue(isinstance(ret, SimpleSchema))
        self.assertEqual(ret.name, "Alice")

    def test_visual_agent_online(self):
        visual_agent = HelloWorldVisualChatAgent(llm_config=LLMConfig(**config["image2meta-llm"]))
        ret = visual_agent.async_run(
                               image_url=("https://ark-project.tos-cn-beijing.volces.com/doc_image"
                                                     "/ark_demo_img_1.png")
        )
        self.assertTrue(ret)
        print(ret)

    def test_visual_agent_base64(self):
        visual_agent = HelloWorldVisualChatAgent(llm_config=LLMConfig(**config["image2meta-llm"]))
        with open("resources/img/ark_demo_img_1.png", 'rb') as f:
            encoded_image = LLMRequestHelper.convert_image_to_base64(image_format='png', image_content=f.read())
        ret = visual_agent.async_run(image_url=encoded_image)
        self.assertTrue(ret)
        print(ret)

if __name__ == '__main__':
    unittest.main()
