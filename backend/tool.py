from __future__ import annotations

import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Type
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, Field

from app.logger import setup_logger
from app.runtime.paths import TMP_DIR
from app.tools.base import BaseTool, ToolInputSchema, ToolResult
from app.tools.commands import ToolCommand

from .config import UpscaleConfig, load_config
from .upscale import UpscaleResult, upscale


logger = setup_logger(__name__)

OUTPUT_DIR = TMP_DIR / "image_hd_upscale"
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


class ImageHDUpscaleInput(ToolInputSchema):
    path: str = Field(
        ...,
        description="要放大的图片：本地路径、完整 http(s) URL 或完整的 url-<uuid> 引用，必须原样传递。",
    )
    scale: Optional[float] = Field(
        None,
        description="放大倍率，1~4，默认取配置（通常为 2）。只想修乱码不放大时传 1。",
    )
    text_hint: str = Field(
        "",
        description="图中文字应有的正确内容（按阅读顺序），用来把乱码修成对的字；不确定就留空。",
    )


async def _read_image_ref(url: str) -> bytes:
    """Read what ``ask_model_for_image_edit`` returned: a local temp file or a URL."""
    if url.startswith(("http://", "https://")):
        import httpx

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            if len(response.content) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("模型返回的图片过大")
            return response.content
    path = Path(url)
    try:
        return path.read_bytes()
    finally:
        # The host writes base64 results to a throwaway temp file.
        if path.parent == Path(tempfile.gettempdir()):
            path.unlink(missing_ok=True)


def _make_redraw(cfg: UpscaleConfig):
    async def redraw(prompt: str, image_bytes: bytes, size: str) -> bytes:
        from app.openai_client import ask_model_for_image_edit

        response = await ask_model_for_image_edit(
            prompt=prompt,
            image_bytes=image_bytes,
            model_name=cfg.model_name.strip() or None,
            size=size,  # type: ignore[arg-type]
        )
        if not response or not response.url:
            raise RuntimeError("图像编辑模型没有返回图片")
        return await _read_image_ref(response.url)

    return redraw


def _resolve_input_path(raw: str, session_ctx: Any) -> str:
    text = str(raw or "").strip()
    if text.replace("\\", "/").startswith("/workspace"):
        from app.tools.file_resolver import resolve_tool_file_path_str

        return resolve_tool_file_path_str(text, session_ctx=session_ctx, must_exist=True)
    return text


async def _load_input(raw: str, session_ctx: Any) -> Image.Image:
    from app.tools.plugins.core_tools.image_tools.utils import path2img

    async with path2img(_resolve_input_path(raw, session_ctx)) as local_path:
        with Image.open(local_path) as img:
            img.load()
            return img.copy()


def save_output(image: Image.Image, cfg: UpscaleConfig) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid4().hex[:8]}"
    fmt = (cfg.output_format or "png").lower().strip(".")
    quality = min(max(int(cfg.jpeg_quality), 50), 100)

    if fmt == "png":
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        limit = cfg.max_output_mb * 1024 * 1024
        if not limit or buffer.tell() <= limit:
            path = OUTPUT_DIR / f"{stem}.png"
            path.write_bytes(buffer.getvalue())
            return path
        fmt = "jpg"
    if fmt == "webp":
        path = OUTPUT_DIR / f"{stem}.webp"
        image.save(path, format="WEBP", quality=quality, method=6)
        return path
    path = OUTPUT_DIR / f"{stem}.jpg"
    image.convert("RGB").save(path, format="JPEG", quality=quality, subsampling=0, optimize=True)
    return path


def describe(result: UpscaleResult, src_size: tuple[int, int], seconds: float) -> str:
    plan = result.plan
    lines = [
        f"放大完成：{src_size[0]}x{src_size[1]} → {plan.out_w}x{plan.out_h}（{result.scale:g} 倍），"
        f"切成 {plan.rows}×{plan.cols} 块，用时 {seconds:.0f} 秒。"
    ]
    if result.scale_reduced:
        lines.append("倍率超出块数/像素上限，已自动降低。")
    if result.failed_tiles:
        lines.append(f"第 {', '.join(result.failed_tiles)} 块重绘失败，用普通插值代替。")
    if result.unaligned_tiles:
        lines.append(f"第 {', '.join(result.unaligned_tiles)} 块重绘后无法和原图对齐，这几块用的是普通插值。")
    heavy = {k: v for k, v in result.patched_tiles.items() if k not in result.unaligned_tiles and v >= 0.05}
    if heavy:
        parts = "，".join(f"{k}（{v:.0%}）" for k, v in heavy.items())
        lines.append(f"模型重绘时挪动或改动了部分内容，这些区域已用原图插值填补：{parts}。")
    return "\n".join(lines)


class ImageHDUpscaleTool(BaseTool):
    """Upscale an image by redrawing overlapping tiles with an image edit model."""

    name: str = "image_hd_upscale"
    description: str = (
        "图片高清放大：把图片带冗余切成若干块，交给 GPT Image 2 等图像编辑模型逐块高清重绘并修复乱码文字，"
        "再对齐拼回高分辨率原图。适合 AI 生成图分辨率低、文字糊或乱码的情况。耗时较长（每块一次模型调用）。"
    )
    usage: str = "用户要把图片放大、变高清、修复图里的乱码文字时使用。"
    manual_tags: List[str] = ["image", "upscale", "放大", "高清", "超分", "修复文字"]
    output_kind = "image"
    result_visibility = "user_visible"
    capabilities: List[str] = ["image.upscale"]
    args_schema: Type[BaseModel] = ImageHDUpscaleInput
    commands = [
        ToolCommand(
            name="image_hd_upscale",
            patterns=["/放大", "/高清", "/upscale"],
            text_arg="text_hint",
            text_required=False,
            image_arg="path",
            image_required=True,
            description="图片高清放大并修复乱码文字（附带图片或引用一条带图片的消息）",
            usage_hint="/放大 [图中文字的正确内容] + 图片",
            start_text="收到，正在分块高清重绘，可能要几分钟～",
            done_text="放大好啦：",
        ),
    ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        status_callback = kwargs.get("status_callback")
        session_ctx = kwargs.get("session_ctx")
        try:
            args = ImageHDUpscaleInput(**{k: v for k, v in kwargs.items() if k in ImageHDUpscaleInput.model_fields})
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"参数错误: {e}")

        cfg = load_config()
        try:
            image = await _load_input(args.path, session_ctx)
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"无法读取图片: {e}")

        async def progress(percent: int, message: str) -> None:
            await self.report_progress(percent, message, status_callback)

        started = time.monotonic()
        try:
            result = await upscale(
                image,
                scale=args.scale or cfg.default_scale,
                cfg=cfg,
                redraw=_make_redraw(cfg),
                text_hint=args.text_hint.strip(),
                progress=progress,
                debug_dir=OUTPUT_DIR / "debug" / time.strftime("%Y%m%d-%H%M%S") if cfg.save_tiles else None,
            )
        except Exception as e:
            logger.error(f"[图片放大] 处理失败: {e}", exc_info=True)
            return ToolResult(tool_name=self.name, success=False, error=f"图片放大失败: {e}")

        if len(result.failed_tiles) == len(result.plan.tiles):
            return ToolResult(tool_name=self.name, success=False, error="所有切块都重绘失败，请检查图像编辑模型配置。")

        path = save_output(result.image, cfg)
        summary = describe(result, image.size, time.monotonic() - started)
        logger.info(f"[图片放大] {summary} 输出: {path}")
        return ToolResult(tool_name=self.name, success=True, data=f"[image,url={path.resolve()}]\n{summary}")
