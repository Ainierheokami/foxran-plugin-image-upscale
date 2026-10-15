"""Tile planning and stitching for model-based upscaling.

Everything here is pure (numpy / OpenCV / Pillow) so it can be tested without
the host. Coordinates in a :class:`TilePlan` are in *output* pixels; the model
receives the matching source region pre-resized to the tile's request size, so
each tile is "redraw this canvas sharper" rather than "invent a bigger image".

Pipeline per tile: model output -> resize to the tile box -> align to the
interpolated reference (small affine) -> low-frequency colour correction.
Neighbouring tiles are then joined along a minimum-difference seam inside
their overlap, so text that two tiles redrew slightly differently is cut
around instead of cross-faded into a ghost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


SIZE_MULTIPLE = 16
# GPT Image 2 rejects requests below this many pixels; smaller tiles are
# requested at the minimum and resized back.
MIN_REQUEST_PIXELS = 655_360
MAX_REQUEST_EDGE = 3840


def parse_size(text: str) -> Tuple[int, int]:
    clean = str(text).lower().replace("×", "x").strip()
    w, h = clean.split("x", 1)
    width, height = int(w), int(h)
    if width <= 0 or height <= 0:
        raise ValueError(f"无效尺寸: {text}")
    return width, height


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    request_w: int
    request_h: int

    @property
    def request_size(self) -> str:
        return f"{self.request_w}x{self.request_h}"


@dataclass
class TilePlan:
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    tile_w: int
    tile_h: int
    overlap: int
    rows: int
    cols: int
    tiles: List[Tile] = field(default_factory=list)

    @property
    def scale_x(self) -> float:
        return self.out_w / self.src_w

    @property
    def scale_y(self) -> float:
        return self.out_h / self.src_h

    def source_box(self, tile: Tile) -> Tuple[float, float, float, float]:
        """The tile's region in source pixels (fractional)."""
        return (
            tile.x / self.scale_x,
            tile.y / self.scale_y,
            (tile.x + tile.w) / self.scale_x,
            (tile.y + tile.h) / self.scale_y,
        )


def _axis_positions(length: int, tile: int, overlap: int) -> Tuple[int, List[int]]:
    """Evenly spread tiles of ``tile`` px over ``length`` with at least ``overlap``."""
    if length <= tile:
        return length, [0]
    step = tile - overlap
    count = max(2, math.ceil((length - overlap) / step))
    span = length - tile
    return tile, [round(i * span / (count - 1)) for i in range(count)]


def _request_dims(w: int, h: int) -> Tuple[int, int]:
    """Size to ask the model for: ``w x h`` snapped to 16px and API limits."""
    rw = max(SIZE_MULTIPLE, round(w / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    rh = max(SIZE_MULTIPLE, round(h / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    if rw * rh < MIN_REQUEST_PIXELS:
        factor = math.sqrt(MIN_REQUEST_PIXELS / (rw * rh))
        rw = math.ceil(rw * factor / SIZE_MULTIPLE) * SIZE_MULTIPLE
        rh = math.ceil(rh * factor / SIZE_MULTIPLE) * SIZE_MULTIPLE
    if max(rw, rh) > MAX_REQUEST_EDGE:
        factor = MAX_REQUEST_EDGE / max(rw, rh)
        rw = max(SIZE_MULTIPLE, math.floor(rw * factor / SIZE_MULTIPLE) * SIZE_MULTIPLE)
        rh = max(SIZE_MULTIPLE, math.floor(rh * factor / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    return rw, rh


def plan_tiles(
    src_w: int,
    src_h: int,
    scale: float,
    tile_sizes: Sequence[str],
    overlap_ratio: float,
) -> TilePlan:
    """Pick the sweet-spot tile size that covers the output with the fewest tiles.

    Ties prefer the plan whose tightest overlap is widest (easier seams), then
    the one asking the model for fewer pixels.
    """
    if src_w <= 0 or src_h <= 0:
        raise ValueError("图片尺寸无效")
    out_w = max(1, round(src_w * scale))
    out_h = max(1, round(src_h * scale))
    overlap_ratio = min(max(overlap_ratio, 0.02), 0.45)

    best: Optional[Tuple[tuple, TilePlan]] = None
    for size in tile_sizes:
        tw, th = parse_size(size)
        overlap = max(SIZE_MULTIPLE, round(min(tw, th) * overlap_ratio))
        real_w, xs = _axis_positions(out_w, tw, overlap)
        real_h, ys = _axis_positions(out_h, th, overlap)
        rw, rh = _request_dims(real_w, real_h)
        tiles = [
            Tile(row=r, col=c, x=x, y=y, w=real_w, h=real_h, request_w=rw, request_h=rh)
            for r, y in enumerate(ys)
            for c, x in enumerate(xs)
        ]
        overlaps = [real_w - (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
        overlaps += [real_h - (ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
        min_overlap = min(overlaps) if overlaps else 0
        plan = TilePlan(
            src_w=src_w, src_h=src_h, out_w=out_w, out_h=out_h,
            tile_w=real_w, tile_h=real_h, overlap=overlap,
            rows=len(ys), cols=len(xs), tiles=tiles,
        )
        key = (len(tiles), -min_overlap, rw * rh * len(tiles))
        if best is None or key < best[0]:
            best = (key, plan)
    if best is None:
        raise ValueError("没有配置可用的切块尺寸")
    return best[1]


# ---------------------------------------------------------------- per tile


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8), "RGB")


def align_to_reference(
    tile: np.ndarray,
    reference: np.ndarray,
    *,
    max_shift_ratio: float = 0.05,
    max_scale_dev: float = 0.06,
) -> Tuple[np.ndarray, bool]:
    """Undo the small shift/zoom image models add when redrawing a tile.

    Estimates an affine warp on blurred, downscaled greyscale copies with ECC
    and applies it only if it stays close to identity; anything larger means
    the model changed the content, and warping would make it worse. Returns
    ``(tile, ok)``; ``ok`` is False when a textured tile could not be aligned.
    """
    h, w = reference.shape[:2]
    factor = min(1.0, 640 / max(h, w))
    small = (max(8, round(w * factor)), max(8, round(h * factor)))

    def prep(arr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, small, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(gray, (0, 0), 1.5) / 255.0

    ref_small = prep(reference)
    # Nearly blank tiles have nothing to lock onto, and nothing to misplace.
    if float(ref_small.std()) < 0.02:
        return tile, True
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            ref_small, prep(tile), warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), None, 5,
        )
    except cv2.error:
        return tile, False

    linear = warp[:, :2]
    shift = warp[:, 2] / factor
    if (
        np.abs(linear - np.eye(2)).max() > max_scale_dev
        or abs(shift[0]) > w * max_shift_ratio
        or abs(shift[1]) > h * max_shift_ratio
    ):
        return tile, False
    full = np.hstack([linear, shift.reshape(2, 1)]).astype(np.float32)
    aligned = cv2.warpAffine(
        tile, full, (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, True


def match_low_frequency(tile: np.ndarray, reference: np.ndarray, strength: float) -> np.ndarray:
    """Pull the tile's broad colour/brightness back to the reference.

    Only the heavily blurred component is replaced, so sharpened edges and
    redrawn text stay intact while tile-to-tile colour drift disappears.
    """
    if strength <= 0:
        return tile
    sigma = max(4.0, max(tile.shape[:2]) / 24)
    delta = cv2.GaussianBlur(reference, (0, 0), sigma) - cv2.GaussianBlur(tile, (0, 0), sigma)
    return tile + delta * float(min(strength, 1.0))


# ---------------------------------------------------------------- stitching


def _seam_mask(a: np.ndarray, b: np.ndarray, feather: float) -> np.ndarray:
    """Weights for ``b`` over a left/right overlap (``a`` left, ``b`` right).

    Finds the top-to-bottom path where the two tiles agree most (dynamic
    programming, as in image quilting), keeps the seam off the overlap edges,
    then softens it by ``feather`` px.
    """
    h, ov = a.shape[:2]
    if ov <= 2:
        return np.ones((h, ov), np.float32) * (np.arange(ov) >= ov / 2)
    diff = np.abs(a - b).sum(axis=2)
    diff = cv2.blur(diff, (5, 5))
    margin = min(ov // 2 - 1, max(2, int(math.ceil(feather * 2))))
    if margin > 0:
        diff[:, :margin] += 1e6
        diff[:, ov - margin:] += 1e6

    cost = diff.copy()
    back = np.zeros((h, ov), np.int8)
    for i in range(1, h):
        prev = cost[i - 1]
        left = np.concatenate(([np.inf], prev[:-1]))
        right = np.concatenate((prev[1:], [np.inf]))
        stacked = np.stack([left, prev, right])
        choice = np.argmin(stacked, axis=0)
        cost[i] += stacked[choice, np.arange(ov)]
        back[i] = choice - 1

    seam = np.empty(h, np.int64)
    seam[-1] = int(np.argmin(cost[-1]))
    for i in range(h - 1, 0, -1):
        seam[i - 1] = seam[i] + back[i, seam[i]]

    mask = (np.arange(ov)[None, :] > seam[:, None]).astype(np.float32)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        mask[:, :1] = 0.0
        mask[:, -1:] = 1.0
    return mask


def _stitch_horizontal(parts: Sequence[np.ndarray], xs: Sequence[int], width: int, feather: float) -> np.ndarray:
    h = parts[0].shape[0]
    canvas = np.zeros((h, width, parts[0].shape[2]), np.float32)
    end = 0
    for part, x in zip(parts, xs):
        pw = part.shape[1]
        ov = max(0, end - x)
        if ov:
            mask = _seam_mask(canvas[:, x:end], part[:, :ov], feather)[..., None]
            canvas[:, x:end] = canvas[:, x:end] * (1 - mask) + part[:, :ov] * mask
        canvas[:, x + ov:x + pw] = part[:, ov:]
        end = x + pw
    return canvas


def stitch(plan: TilePlan, tiles: Sequence[np.ndarray], feather: float = 6.0) -> np.ndarray:
    """Join tiles (in ``plan.tiles`` order) into the full output image."""
    by_pos = {(t.row, t.col): arr for t, arr in zip(plan.tiles, tiles)}
    xs = sorted({t.x for t in plan.tiles})
    ys = sorted({t.y for t in plan.tiles})
    strips = [
        _stitch_horizontal([by_pos[(r, c)] for c in range(plan.cols)], xs, plan.out_w, feather)
        for r in range(plan.rows)
    ]
    # Rows are joined the same way on transposed strips.
    joined = _stitch_horizontal(
        [s.transpose(1, 0, 2) for s in strips], ys, plan.out_h, feather
    )
    return joined.transpose(1, 0, 2)
