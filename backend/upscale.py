"""Tile -> redraw -> align -> stitch, with the model call injected.

``redraw`` takes ``(prompt, png_bytes, "WxH")`` and returns image bytes; the
tool wires it to the host's ``ask_model_for_image_edit`` and tests pass a fake.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Awaitable, Callable, Collection, Dict, List, Optional

import numpy as np
from PIL import Image

from app.logger import setup_logger

from .config import UpscaleConfig
from .tiling import (
    Tile,
    TilePlan,
    align_to_reference,
    anchor_to_source,
    deviation_mask,
    match_colour,
    plan_tiles,
    stitch,
    to_array,
    to_image,
)


logger = setup_logger(__name__)

Redraw = Callable[[str, bytes, str], Awaitable[bytes]]
Progress = Callable[[int, str], Awaitable[None]]

BASE_PROMPT = (
    "这是一张大图切下来的局部（{where}），画面里可能同时有好几块互不相关的内容，四边都可能是被截断的。"
    "请原样重绘整个画面，输出的画面范围必须和输入完全相同：输入里的每一部分，包括四周被截断的地图、文字、图片和卡片，"
    "都要画在原来的位置、保持原来的大小，不要只挑其中某个卡片、图片或主体放大铺满画面。"
    "在此基础上把它画成清晰锐利的高清版本："
    "构图、每个元素的位置和大小、颜色、字体样式都必须与原图完全一致，"
    "不要增删任何内容，不要移动、缩放、裁切或补边，不要加边框或水印；"
    "把模糊、发虚的细节画清楚，把乱码、糊掉或笔画错误的文字修正成清晰正确的文字。"
    "画面边缘被截断的元素保持截断，不要补全。"
)


FAILED = "failed"
UNALIGNED = "unaligned"
DEVIATED = "deviated"
REDRAWN = "redrawn"


@dataclass
class TileOutcome:
    """What ended up in one tile; kept so a later run can redraw only some tiles."""

    image: np.ndarray = field(repr=False)
    status: str = REDRAWN
    """``redrawn``; ``deviated`` (aligned, but more than ``max_deviation`` patched from the original);
    ``unaligned`` (interpolated: could not be registered) or ``failed`` (interpolated: no image)."""
    patched: float = 0.0
    """Share of the tile restored from the interpolated original."""


@dataclass
class UpscaleResult:
    image: Image.Image
    plan: TilePlan
    scale: float
    scale_reduced: bool = False
    tiles: Dict[str, TileOutcome] = field(default_factory=dict, repr=False)
    redrawn: List[str] = field(default_factory=list)
    """Tiles sent to the model in this run (all of them unless it was a partial retry)."""

    def _labels(self, status: str) -> List[str]:
        return [label for label, tile in self.tiles.items() if tile.status == status]

    @property
    def failed_tiles(self) -> List[str]:
        return self._labels(FAILED)

    @property
    def unaligned_tiles(self) -> List[str]:
        return self._labels(UNALIGNED)

    @property
    def patched_tiles(self) -> Dict[str, float]:
        return {
            label: tile.patched for label, tile in self.tiles.items()
            if tile.status in (REDRAWN, DEVIATED) and tile.patched > 0.005
        }


def tile_label(tile: Tile) -> str:
    return f"{tile.row + 1}-{tile.col + 1}"


def _where(plan: TilePlan, tile: Tile) -> str:
    if plan.rows == 1 and plan.cols == 1:
        return "整张图"
    vertical = "" if plan.rows == 1 else ("上" if tile.row == 0 else "下" if tile.row == plan.rows - 1 else "中")
    horizontal = "" if plan.cols == 1 else ("左" if tile.col == 0 else "右" if tile.col == plan.cols - 1 else "中")
    spot = (vertical + horizontal).replace("中中", "中") or "中"
    return f"共 {plan.rows} 行 {plan.cols} 列，这是第 {tile.row + 1} 行第 {tile.col + 1} 列，位于整图{spot}部"


RETRY_PROMPT = (
    "注意：上一次重绘的画面和输入对不上（例如只画了其中一部分并放大，或者挪动了内容）。"
    "这次务必逐一对照输入：画面范围完全相同，每个元素的大小和位置与输入一致，四周边缘的内容一个都不能少。"
)


def build_prompt(plan: TilePlan, tile: Tile, text_hint: str, extra: str, *, retry: bool = False) -> str:
    parts = [BASE_PROMPT.format(where=_where(plan, tile))]
    if retry:
        parts.append(RETRY_PROMPT)
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
    debug_dir: Optional[Path] = None,
    plan: Optional[TilePlan] = None,
    previous: Optional[Dict[str, TileOutcome]] = None,
    only: Optional[Collection[str]] = None,
) -> UpscaleResult:
    """Upscale ``image``; with ``plan``/``previous``/``only``, redraw just the tiles in ``only``.

    A partial retry reuses the earlier run's plan and keeps every other tile
    exactly as it came out, then stitches everything again.
    """
    async def report(percent: int, message: str) -> None:
        if progress:
            try:
                await progress(percent, message)
            except Exception:
                pass

    image.load()
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    src = image.convert("RGB")
    if plan is None:
        used_scale, plan, reduced = choose_scale(src.width, src.height, scale, cfg)
    else:
        used_scale, reduced = scale, False
        if (plan.src_w, plan.src_h) != src.size:
            raise ValueError("原图尺寸和切块方案不一致")
    reference = src.resize((plan.out_w, plan.out_h), Image.LANCZOS)
    ref_arr = to_array(reference)
    previous = previous or {}
    targets = [
        tile for tile in plan.tiles
        if only is None or tile_label(tile) in only or tile_label(tile) not in previous
    ]
    total = len(targets)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        reference.save(debug_dir / "reference.png")
    if total == len(plan.tiles):
        await report(5, f"{src.width}x{src.height} → {plan.out_w}x{plan.out_h}，切成 {plan.rows}×{plan.cols} 块")
    else:
        await report(5, f"重绘第 {', '.join(tile_label(t) for t in targets)} 块（共 {plan.rows}×{plan.cols} 块）")

    semaphore = asyncio.Semaphore(max(1, cfg.max_concurrency))
    done = 0
    scale_factor = (plan.scale_x + plan.scale_y) / 2

    async def run(tile: Tile) -> TileOutcome:
        nonlocal done
        label = tile_label(tile)
        ref_tile = ref_arr[tile.y:tile.y + tile.h, tile.x:tile.x + tile.w]
        # The model gets the interpolated tile at its request size, so it only
        # has to sharpen a canvas whose layout already matches the output.
        source = reference.crop((tile.x, tile.y, tile.x + tile.w, tile.y + tile.h))
        if (tile.request_w, tile.request_h) != source.size:
            source = source.resize((tile.request_w, tile.request_h), Image.LANCZOS)
        payload = _png(source)
        # Redrawing a tile that failed before: say so in the prompt from the first attempt.
        earlier = previous.get(label)
        failed_before = earlier is not None and earlier.status != REDRAWN

        best: Optional[TileCandidate] = None
        last_error: Optional[Exception] = None
        async with semaphore:
            for attempt in range(max(0, cfg.retries) + 1):
                prompt = build_prompt(plan, tile, text_hint, cfg.prompt, retry=failed_before or attempt > 0)
                try:
                    data = await redraw(prompt, payload, tile.request_size)
                    with Image.open(BytesIO(data)) as out:
                        out = out.convert("RGB")
                        if out.size != (tile.w, tile.h):
                            out = out.resize((tile.w, tile.h), Image.LANCZOS)
                        raw = to_array(out)
                except Exception as e:
                    last_error = e
                    logger.warning(f"[图片放大] 切块 {label} 第 {attempt + 1} 次重绘失败: {e}")
                    continue
                candidate = await asyncio.to_thread(_postprocess, raw, ref_tile, cfg, scale_factor)
                if debug_dir is not None:
                    await asyncio.to_thread(_save_debug, debug_dir, f"{label}_try{attempt + 1}", raw, candidate)
                if best is None or candidate.score < best.score:
                    best = candidate
                if candidate.aligned and candidate.deviation <= cfg.max_deviation:
                    break
                logger.info(
                    f"[图片放大] 切块 {label} 第 {attempt + 1} 次重绘与原图偏差 {candidate.deviation:.0%}"
                    f"{'' if candidate.aligned else '，无法对齐'}"
                )

        if best is None:
            if not cfg.fallback_on_failure:
                raise RuntimeError(f"切块 {label} 重绘失败: {last_error}")
            outcome = TileOutcome(image=ref_tile.copy(), status=FAILED)
        elif not best.aligned:
            outcome = TileOutcome(image=best.image, status=UNALIGNED, patched=best.patched)
        else:
            status = REDRAWN if best.deviation <= cfg.max_deviation else DEVIATED
            outcome = TileOutcome(image=best.image, status=status, patched=best.patched)
        done += 1
        await report(5 + int(85 * done / max(total, 1)), f"已完成 {done}/{total} 块")
        return outcome

    fresh = dict(zip((tile_label(t) for t in targets), await asyncio.gather(*(run(t) for t in targets))))
    outcomes = {tile_label(t): fresh.get(tile_label(t)) or previous[tile_label(t)] for t in plan.tiles}
    await report(92, "正在拼接")
    merged = await asyncio.to_thread(stitch, plan, [o.image for o in outcomes.values()], cfg.seam_feather, ref_arr)
    merged = await asyncio.to_thread(anchor_to_source, merged, to_array(src), cfg.color_correction)
    output = to_image(merged)
    if alpha is not None:
        output = output.convert("RGBA")
        output.putalpha(alpha.resize(output.size, Image.LANCZOS))
    return UpscaleResult(
        image=output,
        plan=plan,
        scale=used_scale,
        scale_reduced=reduced,
        tiles=outcomes,
        redrawn=sorted(fresh),
    )


def tiles_worth_retrying(result_tiles: Dict[str, TileOutcome]) -> List[str]:
    """Tiles a retry without explicit labels redraws: the failed ones (no image, not aligned, or deviating too much)."""
    return sorted(label for label, tile in result_tiles.items() if tile.status != REDRAWN)


@dataclass
class TileCandidate:
    image: np.ndarray
    aligned: bool
    deviation: float
    """Share of the tile whose layout differs from the original after alignment."""
    patched: float
    weight: np.ndarray = field(repr=False, default=None)

    @property
    def score(self) -> float:
        return self.deviation + (0 if self.aligned else 1.0)


def _save_debug(directory: Path, stem: str, raw: np.ndarray, candidate: "TileCandidate") -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        to_image(raw).save(directory / f"{stem}_model.png")
        to_image(candidate.image).save(directory / f"{stem}_fixed.png")
        Image.fromarray((np.clip(candidate.weight, 0, 1) * 255).astype(np.uint8), "L").save(directory / f"{stem}_patched.png")
    except Exception as e:
        logger.debug(f"[图片放大] 保存调试图失败: {e}")


def _postprocess(tile: np.ndarray, ref_tile: np.ndarray, cfg: UpscaleConfig, scale: float) -> TileCandidate:
    """Register, colour-match and patch one redrawn tile against the reference."""
    aligned = True
    if cfg.align:
        tile, aligned = align_to_reference(tile, ref_tile, local=cfg.local_flow)
    tile = match_colour(tile, ref_tile, cfg.color_correction)
    weight, deviation = deviation_mask(tile, ref_tile, scale, cfg.layout_threshold)
    if not aligned:
        # A tile that could not be registered is not trusted anywhere.
        weight = np.ones(tile.shape[:2], np.float32)
    patched = float(weight.mean())
    if patched > 0:
        tile = tile * (1 - weight[..., None]) + ref_tile * weight[..., None]
    return TileCandidate(image=tile, aligned=aligned, deviation=deviation if aligned else 1.0, patched=patched, weight=weight)
