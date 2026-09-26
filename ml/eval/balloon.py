"""Clip an erase region to its balloon and fill it flat with the balloon's colour.

Mirrors extension/src/pipeline/inpaint/balloon.ts (see there for the steps);
extension/scripts/check-balloon.mjs checks that the two agree.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

TONE_BINS = 64
TONE_TOLERANCE = 24      # plate.ts
FIELD_COVERAGE = 0.73    # plate.ts's PLATE_TONE_COVERAGE
COLOR_TOLERANCE = 40
INK_LUMA = 128
GUARD = 0.05
PAD_FRACTION = 0.5
PAD_MIN = 8
INTERIOR_MAX = 3.0
LOBE_MIN_SHARE = 0.15
LOBE_INSET = 0.06
LOBE_INSET_MIN = 2.0
LOBE_MIN_SIDE = 8.0
SECOND_LOBE = 0.35


def luma(rgb: np.ndarray) -> np.ndarray:
    """plate.ts's toLuma: BT.601, truncated into a byte."""
    rgb = rgb.astype(np.int64)
    return (rgb[..., 0] * 299 + rgb[..., 1] * 587 + rgb[..., 2] * 114) // 1000


def dominant_ground(pixels: np.ndarray) -> tuple[float, np.ndarray | None]:
    """plate.ts's dominantTones + sampleField: (two-tone coverage, ground rgb)."""
    values = luma(pixels)
    if values.size == 0:
        return 1.0, None
    claimed = np.zeros(values.size, bool)
    tones = []
    for _ in range(2):
        if claimed.all():
            break
        histogram = np.bincount((values[~claimed] * TONE_BINS) >> 8, minlength=TONE_BINS)
        peak = (int(np.argmax(histogram)) + 0.5) * 256 / TONE_BINS
        take = ~claimed & (np.abs(values - peak) <= TONE_TOLERANCE)
        count = int(take.sum())
        if count == 0:
            break
        claimed |= take
        tones.append((count / values.size, pixels[take].astype(np.int64).sum(0) / count))
    if not tones:
        return 1.0, None
    coverage = 0.0
    ground = tones[0]
    for tone in tones:
        coverage += tone[0]
        if tone[0] > ground[0]:
            ground = tone
    return coverage, ground[1]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Unset pixels not 4-connected to the window's border become set."""
    background, _ = ndimage.label(~mask)
    border = np.unique(np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]]))
    return mask | (~mask & ~np.isin(background, border[border > 0]))


def largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x, y, w, h) of the largest all-set axis-aligned rectangle, the first found on a tie.

    Written loop by loop so the TypeScript side picks the same one.
    """
    rows, columns = mask.shape
    heights = [0] * columns
    best = (0, 0, 0, 0, 0)
    for row in range(rows):
        line = mask[row]
        for column in range(columns):
            heights[column] = heights[column] + 1 if line[column] else 0
        stack: list[tuple[int, int]] = []
        for column in range(columns + 1):
            current = heights[column] if column < columns else 0
            start = column
            while stack and stack[-1][1] >= current:
                begin, height = stack.pop()
                area = height * (column - begin)
                if area > best[0]:
                    best = (area, begin, row - height + 1, column - begin, height)
                start = begin
            stack.append((start, current))
    return best[1], best[2], best[3], best[4]


def window_of(base: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1): the region's set pixels, padded and clipped to the page."""
    ys, xs = np.nonzero(base)
    if ys.size == 0:
        return None
    height, width = base.shape
    top, bottom, left, right = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    pad = math.floor(max(PAD_MIN, PAD_FRACTION * max(bottom - top, right - left)))
    return max(0, top - pad), min(height, bottom + pad), max(0, left - pad), min(width, right + pad)


def clip_to_balloon(rgb: np.ndarray, base: np.ndarray, seed: np.ndarray):
    """One region: rgb (H, W, 3) uint8, base and seed page-sized bool masks.

    Returns (mask, colour, lobes), or None when the region should go to Telea.
    """
    bounds = window_of(base)
    if bounds is None:
        return None
    y0, y1, x0, x1 = bounds
    window = rgb[y0:y1, x0:x1]
    b = base[y0:y1, x0:x1]
    s = seed[y0:y1, x0:x1] & b

    coverage, ground = dominant_ground(window[s] if s.any() else window[b])
    if ground is None or coverage < FIELD_COVERAGE:
        return None
    field = (np.abs(window.astype(np.float64) - ground) <= COLOR_TOLERANCE).all(-1)
    components, _ = ndimage.label(field)
    reached = np.unique(components[s & field])
    reached = reached[reached > 0]
    if reached.size == 0:
        return None
    balloon = fill_holes(np.isin(components, reached))

    ink = luma(window) < INK_LUMA
    ink_total = int((ink & b).sum())
    if ink_total:
        shapes, _ = ndimage.label(ink)
        leaving = np.unique(shapes[ink & ~b])
        enclosed = ink & b & ~np.isin(shapes, leaving[leaving > 0])
        if (enclosed & ~balloon).sum() / ink_total > GUARD:
            return None

    paint = b & balloon
    closed = []
    limit = INTERIOR_MAX * int(b.sum())
    for label in reached:
        part = fill_holes(components == label)
        edge = part[0].any() or part[-1].any() or part[:, 0].any() or part[:, -1].any()
        if not edge and int(part.sum()) <= limit:
            closed.append(part)
            paint |= part

    lobes = []
    closed_area = sum(int(part.sum()) for part in closed)
    for part in closed:
        if int(part.sum()) < LOBE_MIN_SHARE * closed_area:
            continue
        ys, xs = np.nonzero(part)
        top, left = int(ys.min()), int(xs.min())
        crop = part[top:int(ys.max()) + 1, left:int(xs.max()) + 1].copy()
        rectangles = [largest_rectangle(crop)]
        x, y, w, h = rectangles[0]
        crop[y:y + h, x:x + w] = False
        second = largest_rectangle(crop)
        if second[2] * second[3] >= SECOND_LOBE * w * h:
            rectangles.append(second)
        for x, y, w, h in rectangles:
            inset = max(LOBE_INSET_MIN, LOBE_INSET * min(w, h))
            if min(w, h) - 2 * inset < LOBE_MIN_SIDE:
                continue
            lobes.append((x0 + left + x + inset, y0 + top + y + inset, w - 2 * inset, h - 2 * inset))

    mask = np.zeros_like(base)
    mask[y0:y1, x0:x1] = paint
    colour = tuple(int(math.floor(c + 0.5)) for c in ground)
    return mask, colour, lobes
