import unittest

from f1.common.llm_response_helper import LLMResponseHelper


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_msg_1 = """
        ABCDEFD
        ```json
        {"name": "json_part1"}
        ```
        ABCFDDD
        AAAAAAA
        ```JSON
        {"name": "json_part2"}
        ```
        
        ```Markdown
        # MARKDOWN TEST
        ```
        """
        self.mock_msg_2 = """
        ABCDEFD
        """

    def test_extract_content_from_response(self):
        res = LLMResponseHelper.extract_contents_from_response(self.mock_msg_1)
        self.assertEqual(res, ['{"name": "json_part1"}', '{"name": "json_part2"}'])
        res_markdown = LLMResponseHelper.extract_contents_from_response(self.mock_msg_1, lang="markdown")
        self.assertEqual(res_markdown, ["# MARKDOWN TEST"])
        res_empty = LLMResponseHelper.extract_contents_from_response(self.mock_msg_2)
        self.assertEqual(res_empty, [])


if __name__ == '__main__':
    unittest.main()
