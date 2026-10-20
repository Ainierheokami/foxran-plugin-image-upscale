"""Second pass over the places the first tile grid got wrong.

A tile boundary that runs through a line of text (or any detail) leaves both
neighbours redrawing half of it, and redrawing the same tiles again gives
the model the same cut. After the first pass the error areas are collected:
regions a tile filled from the interpolated original (content moved, went
missing, could not be registered, or the request failed) and places where
two neighbours disagree inside their overlap. Each area gets pieces cut
locally around it, so the content sits whole inside one request, and every
piece edge is moved onto a quiet row or column of the original so the new
cut does not split something else. Pieces are blended in over a narrow band
at those quiet edges.

Boxes are ``(x0, y0, x1, y1)`` in output pixels, end-exclusive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .tiling import TilePlan, deviation_mask, parse_size, plan_tiles


Box = Tuple[int, int, int, int]


@dataclass
class Patch:
    """One locally recut piece as it was blended into the output."""

    box: Box
    error: Box
    """The error area the piece was cut for."""
    image: np.ndarray = field(repr=False)
    alpha: np.ndarray = field(repr=False)


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersect(a: Box, b: Box) -> Optional[Box]:
    box = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return box if box[0] < box[2] and box[1] < box[3] else None


def grow(box: Box, margin: int, width: int, height: int) -> Box:
    return (max(0, box[0] - margin), max(0, box[1] - margin), min(width, box[2] + margin), min(height, box[3] + margin))


def mask_boxes(weight: np.ndarray, threshold: float = 0.5) -> List[Box]:
    """Bounding boxes of the connected regions where ``weight`` reaches ``threshold``."""
    binary = (weight >= threshold).astype(np.uint8)
    if not binary.any():
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [
        (int(x), int(y), int(x + w), int(y + h))
        for x, y, w, h, _ in stats[1:count]
    ]


def merge_boxes(boxes: Sequence[Box], gap: int) -> List[Box]:
    """Union boxes that overlap or lie within ``gap`` px of each other, until none do."""
    merged = [tuple(b) for b in boxes if box_area(b) > 0]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if a[0] <= b[2] + gap and b[0] <= a[2] + gap and a[1] <= b[3] + gap and b[1] <= a[3] + gap:
                    merged[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def contains(outer: Box, inner: Box) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def group_for_pieces(errors: Sequence[Box], tile_sizes: Sequence[str], margin: int) -> List[Box]:
    """Merge error areas whose union, with its margin, still fits one piece.

    Nearby areas then share one request centred on both instead of each
    getting its own, heavily overlapping piece.
    """
    sizes = [parse_size(s) for s in tile_sizes]

    def fits(box: Box) -> bool:
        w, h = box[2] - box[0] + 2 * margin, box[3] - box[1] + 2 * margin
        return any(w <= tw and h <= th for tw, th in sizes)

    groups = list(errors)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                union = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                if fits(union):
                    groups[i] = union
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    return groups


def seam_conflicts(
    plan: TilePlan, images: Dict[Tuple[int, int], np.ndarray], skip: Sequence[Tuple[int, int]], scale: float, threshold: float,
) -> List[Box]:
    """Areas inside each overlap where the two neighbouring tiles drew different content.

    Both may be close enough to the blurry original to pass on their own and
    still disagree with each other, typically where the boundary cut a line
    of text in two. Compared like :func:`deviation_mask`, on structure only.
    """
    tiles = {(t.row, t.col): t for t in plan.tiles}
    boxes: List[Box] = []
    for (row, col), tile in tiles.items():
        if (row, col) in skip:
            continue
        for other_pos in ((row, col + 1), (row + 1, col)):
            other = tiles.get(other_pos)
            if other is None or other_pos in skip:
                continue
            overlap = intersect(
                (tile.x, tile.y, tile.x + tile.w, tile.y + tile.h),
                (other.x, other.y, other.x + other.w, other.y + other.h),
            )
            if overlap is None:
                continue
            x0, y0, x1, y1 = overlap
            a = images[(row, col)][y0 - tile.y:y1 - tile.y, x0 - tile.x:x1 - tile.x]
            b = images[other_pos][y0 - other.y:y1 - other.y, x0 - other.x:x1 - other.x]
            weight, ratio = deviation_mask(a, b, scale, threshold)
            if ratio:
                boxes += [(x0 + bx0, y0 + by0, x0 + bx1, y0 + by1) for bx0, by0, bx1, by1 in mask_boxes(weight)]
    return boxes


def edge_energy(reference: np.ndarray) -> np.ndarray:
    """Gradient magnitude of the original: high on text and detail, low on flat areas."""
    gray = cv2.cvtColor(np.clip(reference, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def _window_means(profile: np.ndarray, band: int) -> np.ndarray:
    """``out[i]`` is the mean of ``profile[i:i + band]``."""
    sums = np.concatenate(([0.0], np.cumsum(profile, dtype=np.float64)))
    return (sums[band:] - sums[:-band]) / band


def _quiet(profile: np.ndarray, start: int, stop: int, at: int, band: int) -> int:
    """The window start in ``[start, stop]`` with the least energy, mildly preferring ``at``."""
    means = _window_means(profile, band)
    stop = min(stop, len(means) - 1)
    if stop < start:
        return at
    cost = means[start:stop + 1]
    distance = np.abs(np.arange(start, stop + 1) - at) / max(1, stop - start)
    return start + int(np.argmin(cost + distance * 0.1 * (float(profile.mean()) + 1e-6)))


def snap_to_quiet(box: Box, energy: np.ndarray, search: int, band: int) -> Box:
    """Move each edge of ``box`` outwards by up to ``search`` px onto the quietest line.

    The blend band at an edge then sits on flat background instead of
    cutting through the strokes the first grid already split. Edges on the
    image border stay where they are.
    """
    height, width = energy.shape
    x0, y0, x1, y1 = box
    band = max(1, min(band, x1 - x0, y1 - y0))
    cols = energy[y0:y1].mean(axis=0)
    rows = energy[:, x0:x1].mean(axis=1)
    if x0 > 0:
        x0 = _quiet(cols, max(0, x0 - search), x0, x0, band)
    if x1 < width:
        x1 = _quiet(cols, x1 - band, min(width, x1 + search) - band, x1 - band, band) + band
    if y0 > 0:
        y0 = _quiet(rows, max(0, y0 - search), y0, y0, band)
    if y1 < height:
        y1 = _quiet(rows, y1 - band, min(height, y1 + search) - band, y1 - band, band) + band
    return x0, y0, x1, y1


def _fit(box: Box, w: int, h: int, width: int, height: int) -> Box:
    """A ``w`` x ``h`` box centred on ``box``, kept inside the image."""
    w, h = min(w, width), min(h, height)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0 = int(round(min(max(cx - w / 2, 0), width - w)))
    y0 = int(round(min(max(cy - h / 2, 0), height - h)))
    return x0, y0, x0 + w, y0 + h


def plan_pieces(
    error: Box, width: int, height: int, tile_sizes: Sequence[str], overlap_ratio: float, margin: int,
) -> List[Box]:
    """Pieces to redraw for one error area (before edge snapping).

    An area that fits one piece gets a single piece centred on it, at least
    as large as the smallest configured size so the model has context. A
    larger one (a whole failed tile) is covered by a grid of its own, cut at
    a different size and overlap than the first pass.
    """
    sizes = [parse_size(s) for s in tile_sizes]
    area = grow(error, margin, width, height)
    aw, ah = area[2] - area[0], area[3] - area[1]
    if any(aw <= tw and ah <= th for tw, th in sizes):
        min_w = min(tw for tw, _ in sizes)
        min_h = min(th for _, th in sizes)
        return [_fit(area, max(aw, min_w), max(ah, min_h), width, height)]
    sub = plan_tiles(aw, ah, 1.0, tile_sizes, overlap_ratio)
    return [(area[0] + t.x, area[1] + t.y, area[0] + t.x + t.w, area[1] + t.y + t.h) for t in sub.tiles]


def piece_alpha(
    piece: Box, error: Box, bad: Optional[np.ndarray], width: int, height: int, band: int, feather: float,
) -> np.ndarray:
    """How much of a redrawn piece goes into the output.

    Ramps from 0 to 1 over ``band`` px at every edge not on the image border.
    Inside the error area the piece is taken as it is (where it deviated
    itself it already holds the interpolated original, which beats a broken
    first pass); around it, only where the piece itself matches the original.
    """
    x0, y0, x1, y1 = piece
    h, w = y1 - y0, x1 - x0

    def ramp(n: int, open_start: bool, open_end: bool) -> np.ndarray:
        pos = np.arange(n, dtype=np.float32) + 0.5
        r = np.ones(n, np.float32)
        if open_start:
            r = np.minimum(r, pos / band)
        if open_end:
            r = np.minimum(r, (n - pos) / band)
        return np.clip(r, 0.0, 1.0)

    alpha = ramp(h, y0 > 0, y1 < height)[:, None] * ramp(w, x0 > 0, x1 < width)[None, :]
    if bad is None:
        return alpha
    inside = np.zeros((h, w), np.float32)
    hit = intersect(piece, error)
    if hit is not None:
        inside[hit[1] - y0:hit[3] - y0, hit[0] - x0:hit[2] - x0] = 1.0
        if feather > 0:
            inside = cv2.GaussianBlur(inside, (0, 0), feather)
    trust = np.maximum(inside, 1.0 - np.clip(bad, 0.0, 1.0))
    return alpha * trust


def apply_patch(canvas: np.ndarray, patch: Patch) -> np.ndarray:
    x0, y0, x1, y1 = patch.box
    a = patch.alpha[..., None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a) + patch.image * a
    return canvas
