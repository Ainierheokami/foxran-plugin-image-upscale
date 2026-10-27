"""Read the whole image once and work out what its text should say.

The image model redraws one tile at a time and is told to keep the text as
it is, so a wrong but legible character ("做蠣启德跑道前端" for "坐落…") is
redrawn faithfully. A vision model sees the whole picture, reads the text
in context and lists the corrections; every tile prompt then names them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Awaitable, Callable, List, Optional, Tuple

from PIL import Image

from app.logger import setup_logger


logger = setup_logger(__name__)

Ask = Callable[[list], Awaitable[str]]

MAX_EDGE = 2560
MAX_TEXT = 1500

SYSTEM = "你是中文排版校对专家，擅长从图片中准确读出文字，并根据上下文判断哪些字是错别字或乱码。"
PROMPT = (
    "这张图是 AI 生成的，图里的文字可能有错别字、形近字或乱码（笔画对了但字错了，或者根本不是字）。"
    "请完成两件事：\n"
    "1. 按阅读顺序读出图中所有文字，并把错的字按上下文改成正确的；拿不准的保持原样，不要改写措辞、不要补充内容。\n"
    "2. 列出你改动的每一处：图里实际画的是什么（from），应该是什么（to）。from 要照图里的字形写，"
    "尽量取能唯一定位的短片段（2~8 个字），乱码认不出时用 ? 代替认不出的字。\n"
    "只输出 JSON，不要其他内容，格式：\n"
    '{"text": "改正后的全部文字", "fixes": [{"from": "做蠣启德", "to": "坐落启德"}]}\n'
    "没有需要改的字时 fixes 为空数组。"
)


@dataclass
class Proofread:
    text: str = ""
    fixes: List[Tuple[str, str]] = field(default_factory=list)


def _data_url(image: Image.Image) -> str:
    img = image.convert("RGB")
    if max(img.size) > MAX_EDGE:
        ratio = MAX_EDGE / max(img.size)
        img = img.resize((round(img.width * ratio), round(img.height * ratio)), Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_reply(reply: str) -> Proofread:
    """Pull the JSON object out of the reply (models like code fences and preambles)."""
    match = re.search(r"\{.*\}", reply or "", re.S)
    if not match:
        raise ValueError("校对模型没有返回 JSON")
    data = json.loads(match.group(0))
    text = str(data.get("text") or "").strip()
    fixes: List[Tuple[str, str]] = []
    for item in data.get("fixes") or []:
        if not isinstance(item, dict):
            continue
        wrong, right = str(item.get("from") or "").strip(), str(item.get("to") or "").strip()
        if wrong and right and wrong != right and (wrong, right) not in fixes:
            fixes.append((wrong, right))
    return Proofread(text=text[:MAX_TEXT], fixes=fixes[:30])


async def proofread(image: Image.Image, ask: Ask, user_hint: str = "") -> Proofread:
    prompt = PROMPT
    if user_hint:
        prompt += f"\n用户给出的正确文字如下，以它为准：{user_hint}"
    data_url = await asyncio.to_thread(_data_url, image)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    return parse_reply(await ask(messages))


def make_ask(model_name: str = "") -> Ask:
    async def ask(messages: list) -> str:
        from app.openai_client import ask_model

        name = model_name.strip()
        return await ask_model(messages=messages, tags=["vision"], model_names=[name] if name else None)

    return ask


async def try_proofread(image: Image.Image, ask: Ask, user_hint: str = "") -> Optional[Proofread]:
    try:
        result = await proofread(image, ask, user_hint)
    except Exception as e:
        logger.warning(f"[图片放大] 校对图中文字失败，跳过: {e}")
        return None
    logger.info(f"[图片放大] 校对完成，{len(result.fixes)} 处修正: {result.fixes}")
    return result
