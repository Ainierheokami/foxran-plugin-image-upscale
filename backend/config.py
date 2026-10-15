"""The plugin's settings, stored in the host config directory as ``image_hd_upscale.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from app.logger import setup_logger
from app.runtime.paths import CONFIG_DIR


logger = setup_logger(__name__)

CONFIG_FILENAME = "image_hd_upscale.yaml"
CONFIG_PATH = CONFIG_DIR / CONFIG_FILENAME
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "config_template.yaml"


class UpscaleConfig(BaseModel):
    model_name: str = Field("", description="指定 models.yaml 里的图像编辑模型名（如 gpt-image-2）；留空按 image_edit 标签轮询。")
    default_scale: float = Field(2.0, description="未指定倍率时的放大倍率。")
    max_scale: float = Field(4.0, description="允许的最大放大倍率。")
    tile_sizes: List[str] = Field(
        default_factory=lambda: ["1024x1024", "1536x1024", "1024x1536"],
        description="候选切块（输出）尺寸，按块数最少优先挑一个。",
    )
    overlap_ratio: float = Field(0.15, description="相邻切块的冗余比例（相对切块短边）。")
    max_tiles: int = Field(16, description="单张图最多切多少块；超出时自动降低倍率。")
    max_output_pixels: int = Field(40_000_000, description="输出总像素上限；超出时自动降低倍率。")
    max_concurrency: int = Field(3, description="同时请求模型的切块数。")
    retries: int = Field(1, description="单块失败后的重试次数。")
    align: bool = Field(True, description="是否把模型重绘的切块对齐回原图位置。")
    color_correction: float = Field(0.8, description="低频校色强度 0~1；0 关闭。")
    seam_feather: float = Field(6.0, description="拼缝羽化半径（像素）。")
    fallback_on_failure: bool = Field(True, description="某块重试仍失败时用普通插值顶上，而不是整张失败。")
    prompt: str = Field("", description="附加在每块提示词末尾的自定义要求。")
    output_format: str = Field("png", description="输出格式：png / jpg / webp。")
    jpeg_quality: int = Field(95, description="jpg / webp 质量。")
    max_output_mb: float = Field(20.0, description="PNG 超过此大小（MB）时改存为 jpg；0 不限制。")


_cache: tuple[float, UpscaleConfig] | None = None


def load_config() -> UpscaleConfig:
    global _cache
    from app.utils.yaml_utils import load_yaml_config_with_template

    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = -1.0
    if _cache and _cache[0] == mtime:
        return _cache[1]
    try:
        raw = load_yaml_config_with_template(CONFIG_PATH, TEMPLATE_PATH, UpscaleConfig().model_dump())
        cfg = UpscaleConfig(**raw)
    except Exception as e:
        logger.warning(f"[图片放大] 读取配置失败，使用默认值: {e}")
        cfg = UpscaleConfig()
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        pass
    _cache = (mtime, cfg)
    return cfg
