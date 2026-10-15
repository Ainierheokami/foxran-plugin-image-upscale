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
