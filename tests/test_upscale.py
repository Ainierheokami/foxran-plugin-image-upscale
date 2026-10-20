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

    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=1, refine=False)
    result = asyncio.run(upscale_mod.upscale(_poster(), scale=2.0, cfg=cfg, redraw=flaky))
    assert len(result.failed_tiles) == len(result.plan.tiles)
    assert calls["n"] == 2 * len(result.plan.tiles)

    # With the local recut on, its pieces fail too; the first pass stays and the summary says so.
    calls["n"] = 0
    recut = asyncio.run(upscale_mod.upscale(_poster(), scale=2.0, cfg=cfg.model_copy(update={"refine": True}), redraw=flaky))
    assert recut.refine_failed > 0 and not recut.patches
    assert calls["n"] == 2 * (len(recut.plan.tiles) + recut.refine_failed)
    assert "局部重绘失败" in tool_mod.describe(recut, (900, 640), 1.0)

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


# --- retrying single tiles ------------------------------------------------------------

jobs_mod = sys.modules[f"{pkg.__name__}.backend.jobs"]
tool_mod = sys.modules[f"{pkg.__name__}.backend.tool"]


def _bad_at(label: str):
    """A model that returns noise for one tile (so it cannot be aligned) and a clean redraw elsewhere."""
    row, col = label.split("-")
    good = _fake_model(5)
    calls = []

    async def redraw(prompt: str, data: bytes, size: str) -> bytes:
        here = f"第 {row} 行第 {col} 列" in prompt
        calls.append(here)
        if here:
            w, h = tiling.parse_size(size)
            arr = np.random.default_rng(len(calls)).integers(0, 255, (h, w, 3), dtype=np.uint8)
            buf = BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()
        return await good(prompt, data, size)

    redraw.calls = calls
    return redraw


def test_retry_redraws_only_the_named_tiles():
    src = _poster()
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0, refine=False)
    first = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=_bad_at("1-2")))
    assert first.unaligned_tiles == ["1-2"]
    assert upscale_mod.tiles_worth_retrying(first.tiles) == ["1-2"]

    calls = []

    async def fixed(prompt, data, size):
        calls.append(prompt)
        return await _fake_model(7)(prompt, data, size)

    again = asyncio.run(upscale_mod.upscale(
        src, scale=first.scale, cfg=cfg, redraw=fixed,
        plan=first.plan, previous=first.tiles, only=["1-2"],
    ))
    assert len(calls) == 1 and "第 1 行第 2 列" in calls[0]
    # It failed last time, so the prompt says so from the first attempt.
    assert upscale_mod.RETRY_PROMPT in calls[0]
    assert again.redrawn == ["1-2"] and not again.unaligned_tiles
    for label, tile in first.tiles.items():
        if label != "1-2":
            assert again.tiles[label] is tile


def test_jobs_are_found_only_by_id_or_exact_output(tmp_path):
    src = _poster(500, 400)
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0)
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=_fake_model(2)))
    out = tmp_path / "out.png"
    result.image.save(out)
    root = tmp_path / "jobs"
    a = jobs_mod.save_job(root, source=src, result=result, output=out, text_hint="hi", session_id="s1")
    b = jobs_mod.save_job(root, source=src, result=result, output=tmp_path / "other.png", text_hint="", session_id="s1")

    assert jobs_mod.find_job(root, job_id=a.id).id == a.id
    assert jobs_mod.find_job(root, job_id=a.id.upper()).id == a.id
    assert jobs_mod.find_job(root, path=str(out)).id == a.id
    # Without an id or file: only the newest run of the same chat, never another chat's.
    assert jobs_mod.find_job(root) is None
    assert jobs_mod.find_job(root, session_id="s1").id == b.id
    assert jobs_mod.find_job(root, session_id="s2") is None
    assert jobs_mod.find_job(root, job_id="ffffff" if "ffffff" not in {a.id, b.id} else "eeeeee") is None
    assert jobs_mod.find_job(root, job_id="../etc", session_id="s1") is None

    loaded = jobs_mod.find_job(root, job_id=a.id)
    assert loaded.text_hint == "hi" and loaded.plan == result.plan and loaded.source().size == src.size
    tiles = loaded.tiles()
    assert set(tiles) == set(result.tiles)
    for label in tiles:
        # Tiles are stored as 8-bit PNG: only rounding differs.
        assert np.abs(tiles[label].image - result.tiles[label].image).mean() < 1.0

    assert jobs_mod.split_retry_text("3FA9C1 2-3，1-1") == ("3fa9c1", "2-3,1-1")
    assert jobs_mod.split_retry_text("2-3 3fa9c1") == ("3fa9c1", "2-3")
    assert jobs_mod.split_retry_text("2-3") == ("", "2-3")
    assert jobs_mod.parse_labels("1-1,1_1", result.plan) == ["1-1"]
    with pytest.raises(ValueError):
        jobs_mod.parse_labels("9-9", result.plan)


def test_tool_retry_by_id_or_latest_of_the_chat(tmp_path, monkeypatch):
    src = _poster()
    monkeypatch.setattr(tool_mod, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(tool_mod, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(tool_mod, "load_config", lambda: UpscaleConfig(tile_sizes=["1024x1024"], retries=0, refine=False))

    async def load(_raw, _ctx):
        return src.copy()

    monkeypatch.setattr(tool_mod, "_load_input", load)

    async def no_vision(_messages):
        raise RuntimeError("没有视觉模型")

    # Proofreading failing must not stop the upscale.
    monkeypatch.setattr(tool_mod, "_make_ask", lambda _cfg: no_vision)
    model = {"redraw": _bad_at("1-2")}
    monkeypatch.setattr(tool_mod, "_make_redraw", lambda _cfg: lambda *a: model["redraw"](*a))
    tool = tool_mod.ImageHDUpscaleTool()
    ctx = type("Ctx", (), {"session_id": "s1"})()

    first = asyncio.run(tool.execute(path="x.png", session_ctx=ctx))
    assert first.success, first.error
    match = __import__("re").search(r"任务号 ([0-9a-f]{6})", first.data)
    assert match and "1-2" in first.data
    job_id = match.group(1)

    # A bare retry in another chat finds nothing; an unknown id is reported as such.
    other = asyncio.run(tool.execute(retry_tiles="auto", session_ctx=type("Ctx", (), {"session_id": "s2"})()))
    assert not other.success and "还没有放大记录" in other.error
    unknown = asyncio.run(tool.execute(retry_tiles="abcdef 1-2" if job_id != "abcdef" else "abcdee 1-2", session_ctx=ctx))
    assert not unknown.success and "找不到任务号" in unknown.error

    calls = []

    async def fixed(prompt, data, size):
        calls.append(prompt)
        return await _fake_model(9)(prompt, data, size)

    model["redraw"] = fixed
    retried = asyncio.run(tool.execute(retry_tiles=f"{job_id}", session_ctx=ctx))
    assert retried.success, retried.error
    assert len(calls) == 1 and "重绘了第 1-2 块" in retried.data
    assert jobs_mod.find_job(tmp_path / "jobs", job_id=job_id).statuses["1-2"]["status"] == "redrawn"
    # "/放大 重试 <id> <块>" goes the same way; a named tile is redrawn even when it was fine.
    named = asyncio.run(tool.execute(text_hint=f"重试 {job_id} 1-1", session_ctx=ctx))
    assert named.success and len(calls) == 2 and "第 1 行第 1 列" in calls[-1]
    assert f"任务 {job_id}" in named.data

    # A newer upscale in the chat: a bare retry takes it and redraws its failed tiles.
    model["redraw"] = _bad_at("1-1")
    second = asyncio.run(tool.execute(path="y.png", session_ctx=ctx))
    second_id = __import__("re").search(r"任务号 ([0-9a-f]{6})", second.data).group(1)
    model["redraw"] = fixed
    bare = asyncio.run(tool.execute(retry_tiles="auto", session_ctx=ctx))
    assert bare.success and f"任务 {second_id}" in bare.data and "第 1 行第 1 列" in calls[-1] and len(calls) == 3
    # Tiles only, no id: the newest run as well.
    tiles_only = asyncio.run(tool.execute(retry_tiles="1-2", session_ctx=ctx))
    assert tiles_only.success and f"任务 {second_id}" in tiles_only.data and "第 1 行第 2 列" in calls[-1]
    nothing_failed = asyncio.run(tool.execute(retry_tiles="auto", session_ctx=ctx))
    assert not nothing_failed.success and "没有失败的块" in nothing_failed.error
    # Records stay (keep_jobs defaults to 0 = keep forever).
    assert len(jobs_mod.list_jobs(tmp_path / "jobs")) == 2


def test_automatic_retry_uses_the_stronger_prompt():
    model = _bad_at("1-1")
    prompts = []

    async def recording(prompt, data, size):
        prompts.append(prompt)
        return await model(prompt, data, size)

    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=1, refine=False)
    asyncio.run(upscale_mod.upscale(_poster(), scale=2.0, cfg=cfg, redraw=recording))
    bad = [p for p in prompts if "第 1 行第 1 列" in p]
    assert len(bad) == 2
    assert upscale_mod.RETRY_PROMPT not in bad[0] and upscale_mod.RETRY_PROMPT in bad[1]
    assert all(upscale_mod.RETRY_PROMPT not in p for p in prompts if "第 1 行第 1 列" not in p)


proof_mod = sys.modules[f"{pkg.__name__}.backend.proofread"]


def test_proofread_reply_is_parsed_leniently():
    reply = """好的，结果如下：
```json
{"text": "坐落启德跑道前端，临海而建", "fixes": [{"from": "做蠣启德", "to": "坐落启德"}, {"from": "同", "to": "同"}, "x"]}
```"""
    result = proof_mod.parse_reply(reply)
    assert result.text.startswith("坐落") and result.fixes == [("做蠣启德", "坐落启德")]
    with pytest.raises(ValueError):
        proof_mod.parse_reply("看不清")


def test_corrections_reach_every_tile_and_survive_a_retry(tmp_path, monkeypatch):
    src = _poster()
    monkeypatch.setattr(tool_mod, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(tool_mod, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(tool_mod, "load_config", lambda: UpscaleConfig(tile_sizes=["1024x1024"], retries=0))

    async def load(_raw, _ctx):
        return src.copy()

    monkeypatch.setattr(tool_mod, "_load_input", load)
    asked = []

    async def vision(messages):
        asked.append(messages)
        return '{"text": "坐落启德跑道前端", "fixes": [{"from": "做蠣启德", "to": "坐落启德"}]}'

    monkeypatch.setattr(tool_mod, "_make_ask", lambda _cfg: vision)
    prompts = []
    model = _fake_model(4)

    async def recording(prompt, data, size):
        prompts.append(prompt)
        return await model(prompt, data, size)

    monkeypatch.setattr(tool_mod, "_make_redraw", lambda _cfg: recording)
    tool = tool_mod.ImageHDUpscaleTool()
    ctx = type("Ctx", (), {"session_id": "s1"})()

    first = asyncio.run(tool.execute(path="x.png", text_hint="坐落", session_ctx=ctx))
    assert first.success, first.error
    assert len(asked) == 1
    user_text = asked[0][1]["content"][0]["text"]
    assert "以它为准：坐落" in user_text
    assert asked[0][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert prompts and all("「做蠣启德」→「坐落启德」" in p for p in prompts)
    assert "做蠣启德→坐落启德" in first.data

    job_id = __import__("re").search(r"任务号 ([0-9a-f]{6})", first.data).group(1)
    prompts.clear()
    again = asyncio.run(tool.execute(retry_tiles=f"{job_id} 1-1", session_ctx=ctx))
    assert again.success, again.error
    # No second proofread; the saved corrections are reused.
    assert len(asked) == 1 and len(prompts) == 1 and "「做蠣启德」→「坐落启德」" in prompts[0]


# --- settings API ------------------------------------------------------------------------

api_mod = sys.modules[f"{pkg.__name__}.backend.api"]
config_mod = sys.modules[f"{pkg.__name__}.backend.config"]


def test_settings_api_validates_and_keeps_comments(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = tmp_path / "image_hd_upscale.yaml"
    path.write_text(config_mod.TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(api_mod, "CONFIG_PATH", path)
    monkeypatch.setattr(config_mod, "CONFIG_PATH", path)
    monkeypatch.setattr(config_mod, "_cache", None)
    app = FastAPI()
    app.include_router(api_mod.router)
    app.dependency_overrides[api_mod.require_auth] = lambda: True
    client = TestClient(app)

    got = client.get("/api/v1/plugins/image-hd-upscale/config").json()
    assert got["config"]["proofread"] is True and "错别字" in got["descriptions"]["proofread"]

    cfg = dict(got["config"], proofread=False, proofread_model="qwen-vl", tile_sizes=["1536x1024"], retries=2)
    saved = client.post("/api/v1/plugins/image-hd-upscale/config", json={"config": cfg})
    assert saved.status_code == 200, saved.text
    assert saved.json()["config"]["proofread_model"] == "qwen-vl"
    text = path.read_text(encoding="utf-8")
    assert "# 放大前先让视觉模型通读整张图" in text and "proofread: false" in text and "retries: 2" in text

    for bad, word in [
        (dict(cfg, overlap_ratio=0.9), "overlap_ratio"),
        (dict(cfg, tile_sizes=["abc"]), "abc"),
        (dict(cfg, default_scale=5, max_scale=4), "default_scale"),
        (dict(cfg, output_format="gif"), "output_format"),
        (dict(cfg, mystery=1), "mystery"),
        (dict(cfg, refine_tile_sizes=["128x128"]), "128x128"),
        (dict(cfg, refine_max_tiles=100), "refine_max_tiles"),
    ]:
        rejected = client.post("/api/v1/plugins/image-hd-upscale/config", json={"config": bad})
        assert rejected.status_code == 400 and word in rejected.json()["detail"]
    assert "retries: 2" in path.read_text(encoding="utf-8")


# ------------------------------------------------------------ local recut

refine_mod = sys.modules[f"{pkg.__name__}.backend.refine"]
EDGE = 0.12  # share of the request width the grid model garbles at a tile's left and right edges


def _sharpen(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 1.6 - cv2.GaussianBlur(arr, (0, 0), 2.0) * 0.6, 0, 255)


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _edge_garbling_model():
    """Sharpens everything; on grid tiles it also redraws the content near the left and right
    edges wrongly (mirrored), the way a cut line through text leaves both neighbours wrong.
    Pieces recut around such a place (their prompt names a range of the whole image) come out clean."""
    prompts = []

    async def redraw(prompt: str, data: bytes, size: str) -> bytes:
        prompts.append(prompt)
        with Image.open(BytesIO(data)) as im:
            arr = _sharpen(np.asarray(im.convert("RGB"), np.float32))
        if "整图横向" not in prompt:
            band = int(arr.shape[1] * EDGE)
            arr[:, :band] = arr[::-1, :band]
            arr[:, -band:] = arr[::-1, -band:]
        return _png_bytes(arr)

    redraw.prompts = prompts
    return redraw


def test_boxes_merge_and_cuts_move_onto_quiet_lines():
    assert refine_mod.merge_boxes([(0, 0, 10, 10), (15, 0, 30, 10), (100, 100, 110, 110)], gap=8) == [
        (0, 0, 30, 10), (100, 100, 110, 110),
    ]
    # A text column at x 40..60 on a flat page: an edge at x=50 moves off it onto flat ground.
    page = np.full((200, 200, 3), 240, np.float32)
    page[20:180:6, 40:60] = 0
    energy = refine_mod.edge_energy(page)
    x0, y0, x1, y1 = refine_mod.snap_to_quiet((50, 0, 150, 200), energy, search=32, band=6)
    assert x0 + 6 < 40 and energy[:, x0:x0 + 6].mean() < 1.0
    # The other edges were on flat ground or the border already.
    assert (y0, y1) == (0, 200) and energy[:, x1 - 6:x1].mean() < 1.0
    # A small area gets one piece centred on it, with context; a big one its own grid.
    sizes = ["768x768", "1024x768", "768x1024"]
    (piece,) = refine_mod.plan_pieces((900, 500, 1000, 540), 1800, 1280, sizes, 0.3, 64)
    assert piece[2] - piece[0] >= 768 and piece[3] - piece[1] >= 768
    assert piece[0] < 900 and piece[2] > 1000 and piece[1] < 500 and piece[3] > 540
    grid = refine_mod.plan_pieces((0, 0, 1800, 1280), 1800, 1280, sizes, 0.3, 64)
    assert len(grid) > 1
    covered = np.zeros((1280, 1800), bool)
    for bx0, by0, bx1, by1 in grid:
        covered[by0:by1, bx0:bx1] = True
    assert covered.all()


def _edge_error(result, src, plan) -> float:
    """Distance from the sharpened reference inside the garbled bands of the grid tiles."""
    ref = _sharpen(np.asarray(src.resize(result.image.size, Image.LANCZOS), np.float32))
    out = np.asarray(result.image, np.float32)
    mask = np.zeros(out.shape[:2], bool)
    for t in plan.tiles:
        band = int(t.w * EDGE)
        mask[t.y:t.y + t.h, t.x:t.x + band] = True
        mask[t.y:t.y + t.h, t.x + t.w - band:t.x + t.w] = True
    return float(np.abs(out - ref).mean(axis=2)[mask].mean())


def test_areas_cut_by_the_grid_are_recut_locally():
    src = _poster()
    base = UpscaleConfig(tile_sizes=["1024x1024"], retries=0, refine_max_tiles=16)
    plain = asyncio.run(upscale_mod.upscale(
        src, scale=2.0, cfg=base.model_copy(update={"refine": False}), redraw=_edge_garbling_model(),
    ))
    assert plain.patched_tiles and not plain.patches

    model = _edge_garbling_model()
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=base, redraw=model))
    pieces = [p for p in model.prompts if "整图横向" in p]
    assert result.patches and len(pieces) <= 16
    # The recut pieces are told the area failed before.
    assert all(upscale_mod.RETRY_PROMPT in p for p in pieces)
    plan = result.plan
    seams = {t.x for t in plan.tiles if t.x} | {t.x + t.w for t in plan.tiles if t.x + t.w < plan.out_w}
    # Pieces are cut elsewhere: each one spans a first-pass cut line instead of ending on it.
    assert any(p.box[0] < x < p.box[2] for p in result.patches for x in seams)
    # Where the first pass fell back to the blurry original, the sharp redraw now shows.
    assert _edge_error(result, src, plan) < 0.7 * _edge_error(plain, src, plan)
    # And the rest of the image did not get worse.
    ref = np.asarray(src.resize(result.image.size, Image.LANCZOS), np.float32)
    assert np.abs(np.asarray(result.image, np.float32) - ref).mean() < np.abs(np.asarray(plain.image, np.float32) - ref).mean() + 1.0
    assert _seam_jump(np.asarray(result.image, np.float32), plan) < 2.0


def test_recut_budget_and_switch():
    src = _poster()
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0, refine_max_tiles=1)
    model = _edge_garbling_model()
    result = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=model))
    assert sum("整图横向" in p for p in model.prompts) <= 1
    assert result.refine_skipped > 0
    assert "局部重绘上限" in tool_mod.describe(result, src.size, 1.0)

    off = _edge_garbling_model()
    asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg.model_copy(update={"refine": False}), redraw=off))
    assert not any("整图横向" in p for p in off.prompts)


def test_recut_pieces_survive_a_retry_of_other_tiles(tmp_path):
    src = _poster()
    cfg = UpscaleConfig(tile_sizes=["1024x1024"], retries=0, refine_max_tiles=16)
    first = asyncio.run(upscale_mod.upscale(src, scale=2.0, cfg=cfg, redraw=_edge_garbling_model()))
    out = tmp_path / "out.png"
    first.image.save(out)
    job = jobs_mod.save_job(tmp_path / "jobs", source=src, result=first, output=out, text_hint="", session_id="s")
    loaded = jobs_mod.find_job(tmp_path / "jobs", job_id=job.id)
    patches = loaded.patches()
    assert [p.box for p in patches] == [p.box for p in first.patches]
    assert all(np.abs(a.alpha - b.alpha).max() < 0.01 for a, b in zip(patches, first.patches))
    tiles = loaded.tiles()
    assert any(t.regions for t in tiles.values())

    # Redraw a tile that some recut piece does not touch.
    for tile in first.plan.tiles:
        rect = (tile.x, tile.y, tile.x + tile.w, tile.y + tile.h)
        away = [p for p in patches if not refine_mod.intersect(p.box, rect)]
        if away:
            break
    assert away, "the test needs a recut piece away from some tile"
    label = upscale_mod.tile_label(tile)
    model = _edge_garbling_model()
    again = asyncio.run(upscale_mod.upscale(
        loaded.source(), scale=loaded.scale, cfg=cfg, redraw=model,
        plan=loaded.plan, previous=tiles, only=[label], previous_patches=patches,
    ))
    assert again.redrawn == [label]
    # Pieces away from the redrawn tile come back as they were; only areas touching it are recut.
    kept = [p for p in again.patches if any(p is q for q in away)]
    assert len(kept) == len(away)
    assert all(refine_mod.intersect(p.box, rect) for p in again.patches if not any(p is q for q in away))
    grid_calls = [p for p in model.prompts if "整图横向" not in p]
    assert len(grid_calls) == 1
