from __future__ import annotations

import asyncio
import re
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Type
from uuid import uuid4

from PIL import Image
from pydantic import BaseModel, Field

from app.logger import setup_logger
from app.runtime.paths import DATA_DIR, TMP_DIR
from app.tools.base import BaseTool, ToolInputSchema, ToolResult
from app.tools.commands import ToolCommand

from .config import UpscaleConfig, load_config
from .jobs import Job, find_job, parse_labels, save_job, split_retry_text
from .upscale import UpscaleResult, tiles_worth_retrying, upscale


logger = setup_logger(__name__)

OUTPUT_DIR = TMP_DIR / "image_hd_upscale"
JOBS_DIR = DATA_DIR / "image_hd_upscale" / "jobs"
AUTO_RETRY = "auto"
# "/放大 重试 2-3" reads as a retry too, not as text for the model.
RETRY_TEXT = re.compile(r"^(?:重试|重画|重绘|retry)\s*(?P<labels>.*)$", re.IGNORECASE)
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


class ImageHDUpscaleInput(ToolInputSchema):
    path: str = Field(
        "",
        description="要放大的图片：本地路径、完整 http(s) URL 或完整的 url-<uuid> 引用，必须原样传递。"
        "重试时可以不传（用 job_id），或传那次放大输出的结果文件。",
    )
    scale: Optional[float] = Field(
        None,
        description="放大倍率，1~4，默认取配置（通常为 2）。只想修乱码不放大时传 1。",
    )
    text_hint: str = Field(
        "",
        description="图中文字应有的正确内容（按阅读顺序），用来把乱码修成对的字；不确定就留空。重试时留空沿用上次的。",
    )
    job_id: str = Field(
        "",
        description="重试时：那次放大结果里写的任务号（6 位，如 3fa9c1）；留空则用本会话最近一次放大。",
    )
    retry_tiles: str = Field(
        "",
        description="只重绘那次放大里的部分切块，其余块保持不变：填块号（行-列，如 \"2-3\" 或 \"1-2,2-3\"），"
        "填 auto 重绘失败（重绘失败或对不齐而退回插值）的块，填 all 重绘全部块。不重试时留空。",
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


def describe(result: UpscaleResult, src_size: tuple[int, int], seconds: float, *, job_id: str = "", retry: bool = False) -> str:
    plan = result.plan
    if retry:
        lines = [
            f"{f'任务 {job_id}：' if job_id else ''}重绘了第 {', '.join(result.redrawn)} 块（共 {plan.rows}×{plan.cols} 块，其余块保持不变），"
            f"{plan.out_w}x{plan.out_h}，用时 {seconds:.0f} 秒。"
        ]
    else:
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
    heavy = {k: v for k, v in result.patched_tiles.items() if v >= 0.05}
    if heavy:
        parts = "，".join(f"{k}（{v:.0%}）" for k, v in heavy.items())
        lines.append(f"模型重绘时挪动或改动了部分内容，这些区域已用原图插值填补：{parts}。")
    if job_id:
        retryable = tiles_worth_retrying(result.tiles)
        if retryable:
            lines.append(
                f"任务号 {job_id}：发送 /放大重试 {job_id} 只重绘失败的块"
                f"（也可以指定块号，如 /放大重试 {job_id} {retryable[0]}），不用整张重来。"
            )
        else:
            lines.append(f"任务号 {job_id}：想重画某一块可以发送 /放大重试 {job_id} 行-列（如 1-2）。")
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
        ToolCommand(
            name="image_hd_upscale_retry",
            patterns=["/放大重试", "/重试放大", "/upscale-retry"],
            text_arg="retry_tiles",
            text_required=False,
            image_arg="path",
            image_required=False,
            static_args={"retry_tiles": AUTO_RETRY},
            description="只重绘某次放大里失败的块，其余块不动；可写任务号和块号，都不写则重试本会话最近一次放大",
            usage_hint="/放大重试 [任务号] [块号，如 2-3]",
            start_text="收到，正在重绘指定的块～",
            done_text="重绘好啦：",
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
        retry = args.retry_tiles.strip()
        text_hint = args.text_hint.strip()
        matched = RETRY_TEXT.match(text_hint)
        if not retry and matched:
            retry, text_hint = matched.group("labels").strip() or AUTO_RETRY, ""
        if retry or args.job_id.strip():
            job_id, labels = split_retry_text("" if retry.lower() == AUTO_RETRY else retry)
            return await self._retry(
                args.path, args.job_id.strip() or job_id, labels or AUTO_RETRY, text_hint, cfg, session_ctx, status_callback,
            )
        if not args.path.strip():
            return ToolResult(tool_name=self.name, success=False, error="没有收到要放大的图片")
        try:
            image = await _load_input(args.path, session_ctx)
        except Exception as e:
            return ToolResult(tool_name=self.name, success=False, error=f"无法读取图片: {e}")
        return await self._run(image, cfg, session_ctx, status_callback, scale=args.scale or cfg.default_scale, text_hint=text_hint)

    async def _retry(
        self, path: str, job_id: str, retry: str, text_hint: str, cfg: UpscaleConfig, session_ctx: Any, status_callback: Any,
    ) -> ToolResult:
        local = ""
        if path.strip() and not job_id:
            try:
                local = _resolve_input_path(path, session_ctx)
            except Exception:
                local = ""
        session_id = str(getattr(session_ctx, "session_id", "") or "")
        job = find_job(JOBS_DIR, job_id=job_id, path=local, session_id=None if job_id else session_id)
        if job is None:
            if job_id:
                error = f"找不到任务号 {job_id} 的放大记录，请核对放大结果里写的任务号"
            else:
                error = "本会话还没有放大记录，请先用 /放大 处理一张图，或带上任务号（如 /放大重试 3fa9c1 2-3）"
            return ToolResult(tool_name=self.name, success=False, error=error)
        previous = job.tiles()
        if retry.lower() in {AUTO_RETRY, ""}:
            labels = tiles_worth_retrying(previous)
            if not labels:
                return ToolResult(
                    tool_name=self.name, success=False,
                    error=f"任务 {job.id} 的每块都重绘成功了，没有失败的块；想重画某块请写块号"
                    f"（共 {job.plan.rows} 行 {job.plan.cols} 列，如 /放大重试 {job.id} 1-2）",
                )
        elif retry.lower() in {"all", "全部"}:
            labels = sorted(previous)
        else:
            try:
                labels = parse_labels(retry, job.plan)
            except ValueError as e:
                return ToolResult(tool_name=self.name, success=False, error=str(e))
            if not labels:
                return ToolResult(tool_name=self.name, success=False, error="没有指定要重绘的块")
        return await self._run(
            job.source(), cfg, session_ctx, status_callback,
            scale=job.scale, text_hint=text_hint or job.text_hint, job=job, previous=previous, only=labels,
        )

    async def _run(
        self,
        image: Image.Image,
        cfg: UpscaleConfig,
        session_ctx: Any,
        status_callback: Any,
        *,
        scale: float,
        text_hint: str,
        job: Optional[Job] = None,
        previous: Optional[dict] = None,
        only: Optional[List[str]] = None,
    ) -> ToolResult:
        async def progress(percent: int, message: str) -> None:
            await self.report_progress(percent, message, status_callback)

        started = time.monotonic()
        try:
            result = await upscale(
                image,
                scale=scale,
                cfg=cfg,
                redraw=_make_redraw(cfg),
                text_hint=text_hint,
                progress=progress,
                debug_dir=OUTPUT_DIR / "debug" / time.strftime("%Y%m%d-%H%M%S") if cfg.save_tiles else None,
                plan=job.plan if job else None,
                previous=previous,
                only=only,
            )
        except Exception as e:
            logger.error(f"[图片放大] 处理失败: {e}", exc_info=True)
            return ToolResult(tool_name=self.name, success=False, error=f"图片放大失败: {e}")

        if job is None and len(result.failed_tiles) == len(result.plan.tiles):
            return ToolResult(tool_name=self.name, success=False, error="所有切块都重绘失败，请检查图像编辑模型配置。")
        if job is not None and set(result.redrawn) <= set(result.failed_tiles):
            return ToolResult(tool_name=self.name, success=False, error=f"第 {', '.join(result.redrawn)} 块重绘仍然失败，请检查图像编辑模型配置。")

        path = save_output(result.image, cfg)
        saved = None
        try:
            saved = await asyncio.to_thread(
                save_job, JOBS_DIR,
                source=image, result=result, output=path, text_hint=text_hint,
                session_id=str(getattr(session_ctx, "session_id", "") or ""), job=job, keep=cfg.keep_jobs,
            )
        except Exception as e:
            logger.warning(f"[图片放大] 保存放大记录失败，之后无法单独重试切块: {e}")
        summary = describe(
            result, image.size, time.monotonic() - started, job_id=saved.id if saved else "", retry=job is not None,
        )
        logger.info(f"[图片放大] {summary} 输出: {path}")
        return ToolResult(tool_name=self.name, success=True, data=f"[image,url={path.resolve()}]\n{summary}")
