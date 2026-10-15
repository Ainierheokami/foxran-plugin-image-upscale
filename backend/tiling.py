"""Tile planning and stitching for model-based upscaling.

Everything here is pure (numpy / OpenCV / Pillow) so it can be tested without
the host. Coordinates in a :class:`TilePlan` are in *output* pixels; the model
receives the matching source region pre-resized to the tile's request size, so
each tile is "redraw this canvas sharper" rather than "invent a bigger image".

Pipeline per tile: model output -> resize to the tile box -> register to the
interpolated reference (SIFT/ECC affine, then smoothed optical flow) ->
per-tile colour fit -> regions whose layout still deviates fall
back to the reference. Neighbouring tiles are joined along a
minimum-difference seam inside their overlap, so text that two tiles redrew
slightly differently is cut around instead of cross-faded into a ghost; the
stitched image is finally anchored to the source's colours.
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


def _small_size(h: int, w: int, limit: int) -> Tuple[float, Tuple[int, int]]:
    factor = min(1.0, limit / max(h, w))
    return factor, (max(8, round(w * factor)), max(8, round(h * factor)))


def _gray_u8(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _plausible(warp: np.ndarray, w: int, h: int, max_shift_ratio: float, max_zoom_dev: float) -> bool:
    linear = warp[:, :2].astype(np.float64)
    singular = np.linalg.svd(linear, compute_uv=False)
    if singular.min() < 1 - max_zoom_dev or singular.max() > 1 + max_zoom_dev:
        return False
    # Where the tile centre lands tells the shift independent of zoom.
    centre = np.array([w / 2, h / 2])
    moved = linear @ centre + warp[:, 2] - centre
    return abs(moved[0]) <= w * max_shift_ratio and abs(moved[1]) <= h * max_shift_ratio


def _feature_warp(ref_g: np.ndarray, tile_g: np.ndarray) -> Optional[np.ndarray]:
    """Affine ref->tile from SIFT matches; robust to large shifts and zoom."""
    try:
        sift = cv2.SIFT_create(nfeatures=4000)
    except (AttributeError, cv2.error):
        return None
    kp_r, des_r = sift.detectAndCompute(ref_g, None)
    kp_t, des_t = sift.detectAndCompute(tile_g, None)
    if des_r is None or des_t is None or len(kp_r) < 8 or len(kp_t) < 8:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_r, des_t, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < 12:
        return None
    src = np.float32([kp_r[m.queryIdx].pt for m in good])
    dst = np.float32([kp_t[m.trainIdx].pt for m in good])
    warp, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000)
    if warp is None or inliers is None or int(inliers.sum()) < 10:
        return None
    return warp.astype(np.float32)


def _ecc_warp(ref_g: np.ndarray, tile_g: np.ndarray, init: np.ndarray) -> Optional[np.ndarray]:
    blur = lambda g: cv2.GaussianBlur(g.astype(np.float32) / 255.0, (0, 0), 1.5)
    warp = init.astype(np.float32).copy()
    try:
        _, warp = cv2.findTransformECC(
            blur(ref_g), blur(tile_g), warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5,
        )
    except cv2.error:
        return None
    return warp


def _local_flow(aligned: np.ndarray, reference: np.ndarray, max_ratio: float, min_median_px: float = 3.0) -> np.ndarray:
    """Correct smooth local drift (elements nudged by a few pixels) with optical flow.

    Applied only when the tile as a whole has drifted (median displacement
    of at least ``min_median_px``). On a tile that is in place, the flow is
    near zero almost everywhere and spikes only where the model rewrote
    glyphs or on flat areas; warping by that would bend the corrected text.
    The flow is blurred heavily so it moves whole elements, not strokes.
    """
    h, w = reference.shape[:2]
    factor, small = _small_size(h, w, 768)
    ref_g = _gray_u8(reference, small)
    tile_g = _gray_u8(aligned, small)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(ref_g, tile_g, None)
    sigma = max(small) / 40
    flow = cv2.GaussianBlur(flow, (0, 0), sigma)
    limit = max_ratio * max(small)
    magnitude = np.linalg.norm(flow, axis=2, keepdims=True)
    flow = flow * np.minimum(1.0, limit / (magnitude + 1e-6))
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / factor
    if float(np.median(np.linalg.norm(flow, axis=2))) < min_median_px:
        return aligned
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(aligned, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def align_to_reference(
    tile: np.ndarray,
    reference: np.ndarray,
    *,
    max_shift_ratio: float = 0.2,
    max_zoom_dev: float = 0.2,
    local: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Undo the shift, zoom and local drift image models add when redrawing a tile.

    A global affine comes from SIFT matches (refined by ECC), falling back to
    ECC alone on tiles without features; optical flow then corrects smooth
    local drift. Returns ``(tile, ok)``; ``ok`` is False when a textured tile
    could not be registered, and the tile comes back unwarped.
    """
    h, w = reference.shape[:2]
    factor, small = _small_size(h, w, 1024)
    ref_g = _gray_u8(reference, small)
    # Nearly blank tiles have nothing to lock onto, and nothing to misplace.
    if float(ref_g.std()) < 5.0:
        return tile, True
    tile_g = _gray_u8(tile, small)

    sw, sh = small
    warp = _feature_warp(ref_g, tile_g)
    if warp is not None and _plausible(warp, sw, sh, max_shift_ratio, max_zoom_dev):
        refined = _ecc_warp(ref_g, tile_g, warp)
        if refined is not None and _plausible(refined, sw, sh, max_shift_ratio, max_zoom_dev):
            warp = refined
    else:
        warp = _ecc_warp(ref_g, tile_g, np.eye(2, 3, dtype=np.float32))
    if warp is None or not _plausible(warp, sw, sh, max_shift_ratio, max_zoom_dev):
        return tile, False

    full = warp.copy()
    full[:, 2] /= factor
    if _negligible(full, w, h):
        # GPT Image usually keeps the canvas; resampling a tile that is
        # already in place only softens it.
        return tile, True
    aligned = cv2.warpAffine(
        tile, full, (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    if local:
        aligned = _local_flow(aligned, reference, max_ratio=0.03)
    return aligned, True


def _negligible(warp: np.ndarray, w: int, h: int, tolerance_px: float = 1.0) -> bool:
    """Whether the warp moves every corner of the tile by less than ``tolerance_px``."""
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64)
    moved = corners @ warp[:, :2].T.astype(np.float64) + warp[:, 2].astype(np.float64)
    return float(np.abs(moved - corners).max()) < tolerance_px


def deviation_mask(
    tile: np.ndarray, reference: np.ndarray, scale: float, threshold: float, min_area: float = 0.01,
) -> Tuple[np.ndarray, float]:
    """Where the redrawn tile no longer matches the layout of the original.

    Compares structure, not intensity: local zero-mean normalised correlation
    on coarse greyscale copies (below glyph detail). A brighter sky or richer
    colours keep a high correlation; content that moved, vanished or appeared
    does not. ``threshold`` (0~100) is how low the correlation may drop, in
    percent, before a region counts. Only regions covering at least
    ``min_area`` of the tile are returned, as a weight for the reference
    (1 = use the reference), with the share of the tile that deviates.
    """
    h, w = reference.shape[:2]
    factor = 1.0 / max(2.0, 2.0 * scale)
    size = (max(16, round(w * factor)), max(16, round(h * factor)))
    a = cv2.resize(cv2.cvtColor(np.clip(tile, 0, 255).astype(np.float32), cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(cv2.cvtColor(np.clip(reference, 0, 255).astype(np.float32), cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
    sigma = 6.0
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    ma, mb = blur(a), blur(b)
    va = np.maximum(blur(a * a) - ma * ma, 0)
    vb = np.maximum(blur(b * b) - mb * mb, 0)
    cov = blur(a * b) - ma * mb
    textured_a, textured_b = va > 25.0, vb > 25.0  # local std above 5 grey levels
    ncc = cov / np.sqrt(np.maximum(va * vb, 1e-6))
    min_corr = 1.0 - float(threshold) / 100.0
    bad = np.where(
        textured_a & textured_b,
        ncc < min_corr,
        # Texture on one side only: something appeared or vanished. Flat on
        # both sides is fine whatever the shade.
        textured_a != textured_b,
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bad, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area * bad.size
    bad = keep[labels].astype(np.uint8)
    ratio = float(bad.mean())
    if not ratio:
        return np.zeros((h, w), np.float32), 0.0
    mask = cv2.resize(bad.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    grow = max(3, int(round(4 * scale)) | 1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    # A narrow edge: a wide blend would overlay two versions of the content.
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, scale))
    return np.clip(mask * 2.0, 0, 1), ratio


def anchor_to_source(output: np.ndarray, source: np.ndarray, strength: float, iterations: int = 2) -> np.ndarray:
    """Make the result agree with the original once scaled back down.

    Back-projects the heavily smoothed difference between the source and the
    downscaled result. Only broad colour and brightness move, which removes
    tile-to-tile colour steps, including on flat backgrounds where seams have
    nothing else to hide behind.
    """
    if strength <= 0:
        return output
    sh, sw = source.shape[:2]
    oh, ow = output.shape[:2]
    result = output.astype(np.float32)
    for _ in range(max(1, iterations)):
        down = cv2.resize(result, (sw, sh), interpolation=cv2.INTER_AREA)
        # Smoothed well past glyph size: only colour and brightness move, so
        # the source's garbled strokes are not painted back over fixed text.
        residual = cv2.GaussianBlur(source - down, (0, 0), 16.0)
        result = result + cv2.resize(residual, (ow, oh), interpolation=cv2.INTER_CUBIC) * float(min(strength, 1.0))
    return result


def match_colour(tile: np.ndarray, reference: np.ndarray, strength: float) -> np.ndarray:
    """Undo a colour cast the model put on the whole tile.

    Fits one gain and offset per channel (reference ~ gain * tile + offset)
    on blurred copies, ignoring pixels where the content differs (redrawn
    text, fixed details), and applies it to the tile. A global fit cannot
    paint the original's content back the way a per-pixel low-frequency
    correction does.
    """
    if strength <= 0:
        return tile
    t = cv2.GaussianBlur(tile, (0, 0), 3.0).reshape(-1, 3)
    r = cv2.GaussianBlur(reference, (0, 0), 3.0).reshape(-1, 3)
    step = max(1, len(t) // 200_000)
    t, r = t[::step].astype(np.float64), r[::step].astype(np.float64)
    keep = np.ones(len(t), bool)
    gains, offsets = np.ones(3), np.zeros(3)
    for _ in range(3):
        for c in range(3):
            x, y = t[keep, c], r[keep, c]
            if x.size < 100 or x.std() < 1.0:
                gains[c], offsets[c] = 1.0, float(np.median(y - x)) if x.size else 0.0
                continue
            gains[c], offsets[c] = np.polyfit(x, y, 1)
        residual = np.abs(r - (t * gains + offsets)).mean(axis=1)
        keep = residual <= max(4.0, np.percentile(residual, 70))
    gains = np.clip(gains, 0.8, 1.25)
    offsets = np.clip(offsets, -40, 40)
    corrected = tile * gains + offsets
    k = float(min(strength, 1.0))
    return tile * (1 - k) + corrected * k


# ---------------------------------------------------------------- stitching


def _seam_mask(a: np.ndarray, b: np.ndarray, feather: float, ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Weights for ``b`` over a left/right overlap (``a`` left, ``b`` right).

    Finds the top-to-bottom path where the two tiles agree most (dynamic
    programming, as in image quilting), keeps the seam off the overlap edges,
    then softens it by ``feather`` px.
    """
    h, ov = a.shape[:2]
    if ov <= 2:
        return np.ones((h, ov), np.float32) * (np.arange(ov) >= ov / 2)
    diff = np.abs(a - b).sum(axis=2)
    if ref is not None:
        # Prefer cutting where both tiles also agree with the original layout.
        diff += 0.5 * (np.abs(a - ref).sum(axis=2) + np.abs(b - ref).sum(axis=2))
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


def _stitch_horizontal(
    parts: Sequence[np.ndarray], xs: Sequence[int], width: int, feather: float, ref: Optional[np.ndarray] = None,
) -> np.ndarray:
    h = parts[0].shape[0]
    canvas = np.zeros((h, width, parts[0].shape[2]), np.float32)
    end = 0
    for part, x in zip(parts, xs):
        pw = part.shape[1]
        ov = max(0, end - x)
        if ov:
            ref_ov = ref[:, x:end] if ref is not None else None
            mask = _seam_mask(canvas[:, x:end], part[:, :ov], feather, ref_ov)[..., None]
            canvas[:, x:end] = canvas[:, x:end] * (1 - mask) + part[:, :ov] * mask
        canvas[:, x + ov:x + pw] = part[:, ov:]
        end = x + pw
    return canvas


def stitch(
    plan: TilePlan, tiles: Sequence[np.ndarray], feather: float = 6.0, reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Join tiles (in ``plan.tiles`` order) into the full output image.

    With ``reference`` (the interpolated original) seams avoid places where a
    tile departs from the original layout.
    """
    by_pos = {(t.row, t.col): arr for t, arr in zip(plan.tiles, tiles)}
    xs = sorted({t.x for t in plan.tiles})
    ys = sorted({t.y for t in plan.tiles})
    row_ys = {t.row: t.y for t in plan.tiles}
    strips = [
        _stitch_horizontal(
            [by_pos[(r, c)] for c in range(plan.cols)], xs, plan.out_w, feather,
            reference[row_ys[r]:row_ys[r] + plan.tile_h] if reference is not None else None,
        )
        for r in range(plan.rows)
    ]
    # Rows are joined the same way on transposed strips.
    joined = _stitch_horizontal(
        [s.transpose(1, 0, 2) for s in strips], ys, plan.out_h, feather,
        reference.transpose(1, 0, 2) if reference is not None else None,
    )
    return joined.transpose(1, 0, 2)
