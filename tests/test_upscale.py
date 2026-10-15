"""Run from the host repo: ``.venv/bin/python -m pytest <plugin>/tests``."""

import asyncio
import importlib.util
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def _load_package():
    name = "image_hd_upscale_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_ROOT / "__init__.py", submodule_search_locations=[str(PLUGIN_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pkg = _load_package()
tiling = sys.modules[f"{pkg.__name__}.backend.tiling"]
upscale_mod = sys.modules[f"{pkg.__name__}.backend.upscale"]
UpscaleConfig = sys.modules[f"{pkg.__name__}.backend.config"].UpscaleConfig


def _poster(w=900, h=640) -> Image.Image:
    img = Image.new("RGB", (w, h), (240, 228, 210))
    draw = ImageDraw.Draw(img)
    for i in range(0, w, 60):
        draw.rectangle([i, 0, i + 28, h], fill=(200 - i % 120, 120, 90 + i % 100))
    for j in range(0, h, 45):
        draw.text((20, j + 8), f"ROW {j:04d} sweet tile text", fill=(10, 10, 10))
    draw.ellipse([w // 3, h // 4, w // 3 + 260, h // 4 + 260], outline=(30, 60, 160), width=9)
    return img


def test_plan_covers_output_with_overlap():
    plan = tiling.plan_tiles(900, 640, 2.0, ["1024x1024", "1536x1024", "1024x1536"], 0.15)
    assert (plan.out_w, plan.out_h) == (1800, 1280)
    covered = np.zeros((plan.out_h, plan.out_w), bool)
    for t in plan.tiles:
        assert t.x + t.w <= plan.out_w and t.y + t.h <= plan.out_h
        assert t.request_w % 16 == 0 and t.request_h % 16 == 0
        covered[t.y:t.y + t.h, t.x:t.x + t.w] = True
    assert covered.all()
    # 1024x1536 shrinks to the 1280 output height: 1 row x 2 columns beats 2x2.
    assert (plan.rows, plan.cols) == (1, 2)
    xs = sorted({t.x for t in plan.tiles})
    assert plan.tile_w - (xs[1] - xs[0]) >= plan.overlap


def test_small_image_is_single_tile_with_valid_request():
    plan = tiling.plan_tiles(400, 300, 2.0, ["1024x1024"], 0.15)
    assert len(plan.tiles) == 1
    tile = plan.tiles[0]
    assert (tile.w, tile.h) == (800, 600)
    assert tile.request_w * tile.request_h >= tiling.MIN_REQUEST_PIXELS


def test_choose_scale_respects_tile_budget():
    cfg = UpscaleConfig(max_tiles=4)
    scale, plan, reduced = upscale_mod.choose_scale(1000, 1000, 4.0, cfg)
    assert reduced and len(plan.tiles) <= 4 and scale < 4.0


def _fake_model(seed=0):
    """Mimic GPT Image drift: small zoom/shift, colour cast, sharper detail."""
    rng = np.random.default_rng(seed)

    async def redraw(prompt: str, data: bytes, size: str) -> bytes:
        assert "乱码" in prompt
        w, h = tiling.parse_size(size)
        with Image.open(BytesIO(data)) as im:
            arr = np.asarray(im.convert("RGB"), np.float32)
        assert arr.shape[:2] == (h, w)
        zoom = 1 + rng.uniform(-0.01, 0.01)
        dx, dy = rng.uniform(-6, 6, size=2)
        m = np.float32([[zoom, 0, dx], [0, zoom, dy]])
        arr = cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        arr = arr * rng.uniform(0.9, 1.1, size=3) + rng.uniform(-12, 12, size=3)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    return redraw


def _seam_jump(arr: np.ndarray, plan) -> float:
    """Mean absolute step across each tile boundary line vs. the image's typical step."""
    gray = arr.astype(np.float32).mean(axis=2)
    steps = np.abs(np.diff(gray, axis=1))
    boundaries = sorted({t.x for t in plan.tiles if t.x} | {t.x + t.w for t in plan.tiles if t.x + t.w < plan.out_w})
    at = np.mean([steps[:, b - 1].mean() for b in boundaries]) if boundaries else 0.0
    return at / (steps.mean() + 1e-6)


def test_end_to_end_alignment_and_colour():
    src = _poster()
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], max_concurrency=4)
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=_fake_model()))
    plan = result.plan
    assert len(plan.tiles) > 1
    assert result.image.size == (plan.out_w, plan.out_h)
    assert not result.failed_tiles
    assert not result.unaligned_tiles

    out = np.asarray(result.image, np.float32)
    ref = np.asarray(src.resize(result.image.size, Image.LANCZOS), np.float32)
    # Shifts and colour casts are undone: output tracks the reference closely.
    assert np.abs(out - ref).mean() < 6.0
    # No visible step where tiles meet.
    assert _seam_jump(out, plan) < 2.0


def test_failed_tile_falls_back_to_interpolation():
    calls = {"n": 0}

    async def flaky(prompt, data, size):
        calls["n"] += 1
        raise RuntimeError("boom")

    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=1)
    result = asyncio.run(upscale_mod.upscale(_poster(), scale=2.0, cfg=cfg, redraw=flaky))
    assert len(result.failed_tiles) == len(result.plan.tiles)
    assert calls["n"] == 2 * len(result.plan.tiles)

    cfg_strict = cfg.model_copy(update={"fallback_on_failure": False})
    with pytest.raises(RuntimeError):
        asyncio.run(upscale_mod.upscale(_poster(), scale=2.0, cfg=cfg_strict, redraw=flaky))


def test_alpha_is_preserved():
    src = _poster(500, 400).convert("RGBA")
    src.putalpha(128)
    cfg = UpscaleConfig(tile_sizes=["1024x1024"])
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=_fake_model(1)))
    assert result.image.mode == "RGBA"
    assert abs(np.asarray(result.image)[..., 3].mean() - 128) < 2


def _moving_model(shift_block: bool):
    """Large global drift (8% shift, 6% zoom); optionally moves a block of content."""
    async def redraw(prompt: str, data: bytes, size: str) -> bytes:
        w, h = tiling.parse_size(size)
        with Image.open(BytesIO(data)) as im:
            arr = np.asarray(im.convert("RGB"), np.float32)
        m = np.float32([[1.06, 0, 0.08 * w], [0, 0.95, -0.05 * h]])
        arr = cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        if shift_block:
            # The model "re-lays out" a band: content moves down by 15% of the tile.
            y0, y1, dy = int(h * 0.4), int(h * 0.7), int(h * 0.15)
            arr[y0 + dy:y1 + dy] = arr[y0:y1].copy()
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    return redraw


def test_large_drift_is_registered():
    src = _poster()
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0)
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=_moving_model(False)))
    out = np.asarray(result.image, np.float32)
    ref = np.asarray(src.resize(result.image.size, Image.LANCZOS), np.float32)
    assert not result.unaligned_tiles
    assert np.abs(out - ref).mean() < 8.0


def test_moved_content_is_patched_from_the_original():
    src = _poster()
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0)
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=_moving_model(True)))
    assert result.patched_tiles
    # Judged at source resolution, the layout matches the original again.
    down = np.asarray(result.image.resize(src.size, Image.LANCZOS), np.float32)
    err = np.abs(cv2.GaussianBlur(down, (0, 0), 1) - cv2.GaussianBlur(np.asarray(src, np.float32), (0, 0), 1)).mean(axis=2)
    assert (err > 25).mean() < 0.02


def test_unregistrable_tile_falls_back_to_the_original():
    async def noise(prompt, data, size):
        w, h = tiling.parse_size(size)
        arr = np.random.default_rng(3).integers(0, 255, (h, w, 3), dtype=np.uint8)
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    src = _poster(500, 400)
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0)
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=noise))
    ref = np.asarray(src.resize(result.image.size, Image.LANCZOS), np.float32)
    assert result.unaligned_tiles
    assert np.abs(np.asarray(result.image, np.float32) - ref).mean() < 4.0


def test_aligned_redraw_with_corrected_text_is_kept_as_drawn():
    """Regression (real GPT Image run): the model kept the canvas in place and
    fixed garbled glyphs. Patching from the original ghosted the new digits and
    optical flow bent the corrected characters; the redraw must pass through."""
    src = _poster()
    panel = (240, 228, 210)
    draw_src = ImageDraw.Draw(src)
    # Text on a flat panel, as on the poster's cards; the source text is garbled.
    draw_src.rectangle([10, 10, 170, src.height - 10], fill=panel)
    for y in range(20, src.height - 20, 45):
        draw_src.text((16, y), "3%0O kj-45#O", fill=(10, 10, 10))

    async def fix_text(prompt, data, size):
        with Image.open(BytesIO(data)) as im:
            out = im.convert("RGB")
        if prompt.split("这是", 1)[-1].split("，", 1)[0].endswith("第 1 列"):
            draw = ImageDraw.Draw(out)
            # Same lines on the same panel, correct glyphs: only the characters change.
            for y in range(40, out.height - 40, 90):
                draw.rectangle([30, y, 300, y + 26], fill=panel)
                draw.text((34, y + 6), "3500 - 4500 OK", fill=(10, 10, 10))
        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    redrawn = {}

    async def recording(prompt, data, size):
        result = await fix_text(prompt, data, size)
        redrawn[prompt] = result
        return result

    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0)
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=recording))
    assert not result.patched_tiles and not result.unaligned_tiles
    out = np.asarray(result.image, np.float32)
    tile = result.plan.tiles[0]
    first = next(data for prompt, data in redrawn.items() if "第 1 行第 1 列" in prompt)
    model = np.asarray(Image.open(BytesIO(first)).convert("RGB"), np.float32)
    # Where only this tile covers the image, the output is the model's
    # drawing: identical detail (no ghost, no bending), colour may be adjusted.
    second_row = min(t.y for t in result.plan.tiles if t.row == 1)
    inner = (slice(20, second_row - 10), slice(20, 320))
    detail = lambda a: a - cv2.GaussianBlur(a, (0, 0), 8)
    ghost = np.abs(detail(out)[inner] - detail(model)[inner])
    assert ghost.mean() < 0.05 and ghost.max() < 3.0
    assert np.abs(out[inner] - model[inner]).mean() < 10.0
