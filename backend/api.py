"""Settings API for the plugin's WebUI editor.

Like the host's generic visual editor it keeps the file's comments (host
4.5.3+), and on top of that range-checks every value before saving.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from app.api.endpoints.auth import require_auth
from app.logger import setup_logger

from .config import CONFIG_PATH, TEMPLATE_PATH, UpscaleConfig, load_config
from .tiling import parse_size


logger = setup_logger(__name__)

router = APIRouter()
PREFIX = "/api/v1/plugins/image-hd-upscale"

RANGES: Dict[str, tuple] = {
    "default_scale": (1.0, 8.0),
    "max_scale": (1.0, 8.0),
    "overlap_ratio": (0.02, 0.45),
    "max_tiles": (1, 64),
    "max_output_pixels": (1_000_000, 200_000_000),
    "max_concurrency": (1, 16),
    "retries": (0, 5),
    "layout_threshold": (0.0, 100.0),
    "max_deviation": (0.0, 1.0),
    "color_correction": (0.0, 1.0),
    "seam_feather": (0.0, 64.0),
    "jpeg_quality": (50, 100),
    "max_output_mb": (0.0, 500.0),
    "keep_jobs": (0, 100_000),
}
FORMATS = ("png", "jpg", "webp")


class ConfigUpdate(BaseModel):
    config: Dict[str, Any]


def validate(raw: Dict[str, Any]) -> UpscaleConfig:
    """Parse and range-check a config from the editor; raises ValueError with a readable message."""
    unknown = sorted(set(raw) - set(UpscaleConfig.model_fields))
    if unknown:
        raise ValueError(f"未知配置项: {', '.join(unknown)}")
    try:
        cfg = UpscaleConfig(**raw)
    except ValidationError as e:
        first = e.errors()[0]
        raise ValueError(f"{'.'.join(str(p) for p in first.get('loc', ()))}: {first.get('msg', '无效')}") from e
    problems: List[str] = []
    for name, (low, high) in RANGES.items():
        value = getattr(cfg, name)
        if not low <= value <= high:
            problems.append(f"{name} 应在 {low}~{high} 之间")
    if cfg.default_scale > cfg.max_scale:
        problems.append("default_scale 不能大于 max_scale")
    if not cfg.tile_sizes:
        problems.append("tile_sizes 至少要有一种切块尺寸")
    for size in cfg.tile_sizes:
        try:
            w, h = parse_size(size)
        except Exception:
            problems.append(f"切块尺寸 {size} 格式不对，应写成 宽x高")
            continue
        if not (256 <= w <= 3840 and 256 <= h <= 3840):
            problems.append(f"切块尺寸 {size} 的边长应在 256~3840 之间")
    if cfg.output_format.lower().strip(".") not in FORMATS:
        problems.append(f"output_format 只能是 {' / '.join(FORMATS)}")
    if problems:
        raise ValueError("；".join(problems))
    return cfg


def save_config(cfg: UpscaleConfig) -> None:
    """Write ``cfg`` into the config file, keeping its comments and key order."""
    from app.utils.yaml_utils import dump_yaml_preserving_comments

    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    dump_yaml_preserving_comments(CONFIG_PATH, cfg.model_dump())


def _payload(cfg: UpscaleConfig) -> Dict[str, Any]:
    return {
        "config": cfg.model_dump(),
        "defaults": UpscaleConfig().model_dump(),
        "descriptions": {name: field.description or "" for name, field in UpscaleConfig.model_fields.items()},
        "ranges": {name: list(bounds) for name, bounds in RANGES.items()},
        "file_name": CONFIG_PATH.name,
    }


@router.get(f"{PREFIX}/config")
async def get_config(_: bool = Depends(require_auth)):
    return _payload(load_config())


@router.post(f"{PREFIX}/config")
async def update_config(data: ConfigUpdate, _: bool = Depends(require_auth)):
    try:
        cfg = validate(data.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        save_config(cfg)
    except Exception as e:
        logger.error(f"[图片放大] 保存配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存配置失败: {e}")
    return _payload(load_config())
