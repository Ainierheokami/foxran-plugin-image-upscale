from app.logger import setup_logger

from .backend.tool import ImageHDUpscaleTool


logger = setup_logger(__name__)

register_tool_classes = [ImageHDUpscaleTool]


async def startup() -> None:
    from .backend.config import load_config

    load_config()
    logger.info("Image HD upscale plugin started")


__all__ = ["ImageHDUpscaleTool", "register_tool_classes", "startup"]
