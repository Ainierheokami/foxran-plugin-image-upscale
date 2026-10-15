"""Tile -> redraw -> align -> stitch, with the model call injected.

``redraw`` takes ``(prompt, png_bytes, "WxH")`` and returns image bytes; the
tool wires it to the host's ``ask_model_for_image_edit`` and tests pass a fake.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from io import BytesIO
from typing import Awaitable, Callable, List, Optional

import numpy as np
from PIL import Image

from app.logger import setup_logger

from .config import UpscaleConfig
from .tiling import (
    Tile,
    TilePlan,
    align_to_reference,
    match_low_frequency,
    plan_tiles,
    stitch,
    to_array,
    to_image,
)


logger = setup_logger(__name__)

Redraw = Callable[[str, bytes, str], Awaitable[bytes]]
Progress = Callable[[int, str], Awaitable[None]]

BASE_PROMPT = (
    "这是一张大图切下来的局部（{where}）。请把它重绘成清晰锐利的高清版本："
    "构图、每个元素的位置和大小、颜色、字体样式都必须与原图完全一致，"
    "不要增删任何内容，不要移动、缩放、裁切或补边，不要加边框或水印；"
    "把模糊、发虚的细节画清楚，把乱码、糊掉或笔画错误的文字修正成清晰正确的文字。"
    "画面边缘被截断的元素保持截断，不要补全。"
)


@dataclass
class UpscaleResult:
    image: Image.Image
    plan: TilePlan
    scale: float
    scale_reduced: bool = False
    failed_tiles: List[str] = field(default_factory=list)
    unaligned_tiles: List[str] = field(default_factory=list)


def _where(plan: TilePlan, tile: Tile) -> str:
    if plan.rows == 1 and plan.cols == 1:
        return "整张图"
    vertical = "" if plan.rows == 1 else ("上" if tile.row == 0 else "下" if tile.row == plan.rows - 1 else "中")
    horizontal = "" if plan.cols == 1 else ("左" if tile.col == 0 else "右" if tile.col == plan.cols - 1 else "中")
    spot = (vertical + horizontal).replace("中中", "中") or "中"
    return f"共 {plan.rows} 行 {plan.cols} 列，这是第 {tile.row + 1} 行第 {tile.col + 1} 列，位于整图{spot}部"


def build_prompt(plan: TilePlan, tile: Tile, text_hint: str, extra: str) -> str:
    parts = [BASE_PROMPT.format(where=_where(plan, tile))]
    if text_hint:
        parts.append(
            f"整张图里文字的正确内容如下，仅用于修正本块中出现的文字，本块没有的文字不要画进来：{text_hint}"
        )
    if extra:
        parts.append(extra)
    return "".join(parts)


def choose_scale(src_w: int, src_h: int, requested: float, cfg: UpscaleConfig) -> tuple[float, TilePlan, bool]:
    """Largest scale ≤ requested that fits ``max_tiles`` and ``max_output_pixels``."""
    scale = min(max(requested, 1.0), max(cfg.max_scale, 1.0))
    reduced = scale < requested
    max_by_pixels = math.sqrt(max(cfg.max_output_pixels, 1) / (src_w * src_h))
    if scale > max_by_pixels:
        scale, reduced = max(1.0, max_by_pixels), True
    while True:
        plan = plan_tiles(src_w, src_h, scale, cfg.tile_sizes, cfg.overlap_ratio)
        if len(plan.tiles) <= max(cfg.max_tiles, 1) or scale <= 1.0:
            return scale, plan, reduced
        scale, reduced = max(1.0, round(scale - 0.25, 2)), True


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def upscale(
    image: Image.Image,
    *,
    scale: float,
    cfg: UpscaleConfig,
    redraw: Redraw,
    text_hint: str = "",
    progress: Optional[Progress] = None,
) -> UpscaleResult:
    async def report(percent: int, message: str) -> None:
        if progress:
            try:
                await progress(percent, message)
            except Exception:
                pass

    image.load()
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    src = image.convert("RGB")
    used_scale, plan, reduced = choose_scale(src.width, src.height, scale, cfg)
    reference = src.resize((plan.out_w, plan.out_h), Image.LANCZOS)
    ref_arr = to_array(reference)
    total = len(plan.tiles)
    await report(5, f"{src.width}x{src.height} → {plan.out_w}x{plan.out_h}，切成 {plan.rows}×{plan.cols} 块")

    semaphore = asyncio.Semaphore(max(1, cfg.max_concurrency))
    done = 0
    failed: List[str] = []
    unaligned: List[str] = []

    async def run(tile: Tile) -> np.ndarray:
        nonlocal done
        label = f"{tile.row + 1}-{tile.col + 1}"
        ref_tile = ref_arr[tile.y:tile.y + tile.h, tile.x:tile.x + tile.w]
        # The model gets the interpolated tile at its request size, so it only
        # has to sharpen a canvas whose layout already matches the output.
        source = reference.crop((tile.x, tile.y, tile.x + tile.w, tile.y + tile.h))
        if (tile.request_w, tile.request_h) != source.size:
            source = source.resize((tile.request_w, tile.request_h), Image.LANCZOS)
        payload = _png(source)
        prompt = build_prompt(plan, tile, text_hint, cfg.prompt)

        result: Optional[np.ndarray] = None
        last_error: Optional[Exception] = None
        async with semaphore:
            for attempt in range(max(0, cfg.retries) + 1):
                try:
                    data = await redraw(prompt, payload, tile.request_size)
                    with Image.open(BytesIO(data)) as out:
                        out = out.convert("RGB")
                        if out.size != (tile.w, tile.h):
                            out = out.resize((tile.w, tile.h), Image.LANCZOS)
                        result = to_array(out)
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"[图片放大] 切块 {label} 第 {attempt + 1} 次重绘失败: {e}")

        if result is None:
            if not cfg.fallback_on_failure:
                raise RuntimeError(f"切块 {label} 重绘失败: {last_error}")
            failed.append(label)
            result = ref_tile.copy()
        else:
            result = await asyncio.to_thread(_postprocess, result, ref_tile, cfg, label, unaligned)
        done += 1
        await report(5 + int(85 * done / total), f"已完成 {done}/{total} 块")
        return result

    tiles = await asyncio.gather(*(run(tile) for tile in plan.tiles))
    await report(92, "正在拼接")
    merged = await asyncio.to_thread(stitch, plan, tiles, cfg.seam_feather)
    output = to_image(merged)
    if alpha is not None:
        output = output.convert("RGBA")
        output.putalpha(alpha.resize(output.size, Image.LANCZOS))
    return UpscaleResult(
        image=output,
        plan=plan,
        scale=used_scale,
        scale_reduced=reduced,
        failed_tiles=sorted(failed),
        unaligned_tiles=sorted(unaligned),
    )


def _postprocess(tile: np.ndarray, ref_tile: np.ndarray, cfg: UpscaleConfig, label: str, unaligned: List[str]) -> np.ndarray:
    if cfg.align:
        tile, ok = align_to_reference(tile, ref_tile)
        if not ok:
            unaligned.append(label)
    return match_low_frequency(tile, ref_tile, cfg.color_correction)
