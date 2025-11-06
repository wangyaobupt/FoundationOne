import base64
import logging


class LLMRequestHelper:
    logger = logging.getLogger(__name__)

    @classmethod
    def convert_image_to_base64(cls, image_content: bytes, image_format: str) -> str:
        image_url = base64.b64encode(image_content).decode('utf-8')
        return f"data:image/{image_format};base64,{image_url}"
