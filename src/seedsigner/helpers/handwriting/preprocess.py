"""
    Word image normalization.

    Turns the crop of one handwritten line (grayscale, any size, possibly with
    ruled or printed lines and bits of the neighboring lines) into the input of
    the word recognizer: an ink map of fixed height, 0 for paper and 255 for ink.

    The same function is used to build the training data, so any change here
    requires retraining the model.
"""
import numpy as np
from PIL import Image

INPUT_HEIGHT = 32
MAX_INPUT_WIDTH = 256

# Lines are reduced to at most this height before processing
WORKING_HEIGHT = 64



def box_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """
        Mean over a (2r+1) x (2r+1) window, edges replicated (integral image).
    """
    k = 2 * radius + 1
    p = np.pad(a, radius + 1, mode="edge").astype(np.float64)
    s = p.cumsum(0).cumsum(1)
    total = s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    return (total / (k * k))[:a.shape[0], :a.shape[1]].astype(np.float32)



def remove_long_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    """
        Removes horizontal runs of ink longer than `min_length`: ruled lines and
        printed lines, even slightly tilted (each row then holds a long piece).
        Letter strokes are much shorter.
    """
    starts = mask.copy()
    starts[:, 1:] &= ~mask[:, :-1]
    run_ids = np.cumsum(starts.ravel()).reshape(mask.shape) * mask
    lengths = np.bincount(run_ids.ravel())
    lengths[0] = 0
    return mask & ~(lengths >= min_length)[run_ids]



def text_rows(mask: np.ndarray) -> tuple[int, int] | None:
    """
        Rows of the main line of text: the heaviest block of consecutive inked
        rows, extended while ink continues (ascenders, descenders). Pieces of the
        neighboring lines above and below are left out.
    """
    profile = mask.sum(1)
    if profile.max() == 0:
        return None
    active = np.append(profile > max(2, 0.08 * profile.max()), False)
    blocks, start = [], None
    for y, on in enumerate(active):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if blocks and start - blocks[-1][1] <= 2:
                blocks[-1][1] = y
            else:
                blocks.append([start, y])
            start = None
    if not blocks:
        return None
    y0, y1 = max(blocks, key=lambda b: profile[b[0]:b[1]].sum())
    while y0 > 0 and profile[y0 - 1] > 0:
        y0 -= 1
    while y1 < len(profile) and profile[y1] > 0:
        y1 += 1
    return y0, y1



def word_columns(mask: np.ndarray, height: int) -> np.ndarray | None:
    """
        Inked columns of the word: groups of columns separated by gaps wider than
        the line height; groups holding less than 3% of the ink (stray marks,
        remains of a printed line) are dropped. None if no group is left (only
        scattered specks).
    """
    ink_per_column = mask.sum(0)
    columns = np.nonzero(ink_per_column)[0]
    groups = np.split(columns, np.nonzero(np.diff(columns) > height)[0] + 1)
    total = ink_per_column.sum()
    kept = [g for g in groups if ink_per_column[g].sum() >= 0.03 * total]
    return np.concatenate(kept) if kept else None



def normalize_word(line: Image.Image) -> np.ndarray | None:
    """
        Returns the ink map (uint8, INPUT_HEIGHT x width, width <= MAX_INPUT_WIDTH)
        of the word written on `line`, or None if no ink is found.
    """
    found = locate_word(line)
    return None if found is None else found[0]



def locate_word(line: Image.Image) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """
        normalize_word, also returning the box (left, top, right, bottom) of the
        word in `line` coordinates.
    """
    line = line.convert("L")
    factor = 1
    if line.height > WORKING_HEIGHT:
        factor = -(-line.height // WORKING_HEIGHT)
        line = line.reduce(factor)
    a = np.asarray(line, np.float32)

    # Ink relative to the local paper brightness: insensitive to uneven lighting
    background = box_mean(a, max(4, line.height // 4))
    ink = np.clip((background - a) / np.maximum(0.5 * background, 1), 0, 1)

    # Ink pixels: threshold adapted to the contrast of the darkest strokes, so
    # that faint pencil and dark marker are both kept
    strongest = np.percentile(ink, 99.5)
    if strongest < 0.08:
        return None
    threshold = max(0.06, 0.35 * strongest)
    mask = ink > threshold

    # Ruled and printed lines, found on a more permissive mask where even a
    # faint line stays continuous, then removed with one pixel of margin
    faint = ink > 0.5 * threshold
    long_runs = faint & ~remove_long_runs(faint, max(24, line.width // 4))
    long_runs[1:] |= long_runs[:-1]
    long_runs[:-1] |= long_runs[1:]
    mask &= ~long_runs

    rows = text_rows(mask)
    if rows is None:
        return None
    y0, y1 = rows
    height = y1 - y0
    columns = word_columns(mask[y0:y1], height)
    if columns is None:
        return None
    margin_y, margin_x = int(0.15 * height), int(0.2 * height)
    y0, y1 = max(0, y0 - margin_y), min(a.shape[0], y1 + margin_y + 1)
    x0, x1 = max(0, columns[0] - margin_x), min(a.shape[1], columns[-1] + margin_x + 1)

    word = (ink * mask)[y0:y1, x0:x1]
    width = int(round(word.shape[1] * INPUT_HEIGHT / word.shape[0]))
    width = min(max(width, 8), MAX_INPUT_WIDTH)
    resized = Image.fromarray((word * 255).astype(np.uint8)).resize((width, INPUT_HEIGHT), Image.BILINEAR)
    return np.asarray(resized), (x0 * factor, y0 * factor, x1 * factor, y1 * factor)



def unread_ink(line: Image.Image, share: float = 0.14) -> float:
    """
        Width, in heights of the text, of the faint ink of `line` left out of
        the word that normalize_word gives to the recognizer: strokes on the
        rows of the word, beyond its first or last column, lighter than its
        threshold but darker than `share` of the darkest strokes once averaged
        over 3 x 3 pixels (faded or broken letters). 0.0 if there is none or no
        word is found.
    """
    line = line.convert("L")
    factor = 1
    if line.height > WORKING_HEIGHT:
        factor = -(-line.height // WORKING_HEIGHT)
        line = line.reduce(factor)
    a = np.asarray(line, np.float32)
    background = box_mean(a, max(4, line.height // 4))
    ink = np.clip((background - a) / np.maximum(0.5 * background, 1), 0, 1)
    strongest = np.percentile(ink, 99.5)
    if strongest < 0.08:
        return 0.0
    threshold = max(0.06, 0.35 * strongest)
    faint_threshold = max(0.03, share * strongest)
    faint = (box_mean(ink, 1) > faint_threshold) & (ink > 0.7 * faint_threshold)
    lines = faint | (ink > 0.5 * threshold)
    long_runs = lines & ~remove_long_runs(lines, max(24, line.width // 4))
    long_runs[1:] |= long_runs[:-1]
    long_runs[:-1] |= long_runs[1:]
    mask = (ink > threshold) & ~long_runs
    faint &= ~long_runs
    rows = text_rows(mask)
    if rows is None:
        return 0.0
    y0, y1 = rows
    height = y1 - y0
    columns = word_columns(mask[y0:y1], height)
    pale = word_columns(faint[y0:y1], height)
    if columns is None or pale is None:
        return 0.0
    return max(0, columns[0] - pale[0], pale[-1] - columns[-1]) / max(height, 1)



# Printed or ruled lines in a word crop (erase_lines)
LINE_SLOPES = np.linspace(-0.18, 0.18, 37)  # slopes tried, about +-10 degrees
LINE_COVER = 0.6              # share of the columns a line crosses
LINE_RUN = 0.35               # share of the width of its longest unbroken stretch
MAX_LINES = 3


def find_lines(faint: np.ndarray, max_thickness: int) -> list[tuple[float, float]]:
    """
        Straight thin lines of `faint` (ink mask) crossing most of its width,
        as (slope, row at the middle column), longest first; at most MAX_LINES.
    """
    h, w = faint.shape
    ys, xs = np.nonzero(faint)
    if len(xs) == 0:
        return []
    found, taken = [], np.zeros(len(xs), bool)
    for _ in range(MAX_LINES):
        best = None
        for s in LINE_SLOPES:
            r = np.round(ys - s * (xs - w / 2)).astype(int) + w
            hits = np.zeros((h + 2 * w + 2, w), bool)
            keep = ~taken
            hits[r[keep], xs[keep]] = True
            near = hits.copy()
            near[1:] |= hits[:-1]
            near[:-1] |= hits[1:]
            cover = near.sum(1)
            k = int(np.argmax(cover))
            if best is None or cover[k] > best[0]:
                best = (cover[k], s, k, near[k])
        cover, s, k, row = best
        if cover < LINE_COVER * w:
            break
        breaks = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
        if (breaks[1::2] - breaks[::2]).max() < LINE_RUN * w:
            break
        y = k - w
        # a line is thin: most of the columns it crosses are paper a little above and below it
        cols = np.flatnonzero(row)
        side = []
        for d in (-max_thickness - 2, max_thickness + 2):
            yy = np.round(y + d + s * (cols - w / 2)).astype(int)
            inside = (yy >= 0) & (yy < h)
            side.append(np.where(inside, faint[np.clip(yy, 0, h - 1), cols], False))
        if (side[0] & side[1]).mean() > 0.3:
            break
        found.append((float(s), float(y)))
        taken |= np.abs(ys - s * (xs - w / 2) - y) <= max_thickness
    return found


def erase_lines(line: Image.Image) -> Image.Image:
    """
        `line` (grayscale word crop) reduced as in normalize_word, with the
        straight printed or ruled lines crossing it (find_lines) replaced by the
        paper, column by column; strokes crossing a line from above to below
        are kept. Returned unchanged if no line is found.
    """
    line = line.convert("L")
    if line.height > WORKING_HEIGHT:
        line = line.reduce(-(-line.height // WORKING_HEIGHT))
    a = np.asarray(line, np.float32)
    h, w = a.shape
    if w < 24:
        return line
    background = box_mean(a, max(4, h // 4))
    ink = np.clip((background - a) / np.maximum(0.5 * background, 1), 0, 1)
    strongest = np.percentile(ink, 99.5)
    if strongest < 0.08:
        return line
    faint = ink > 0.5 * max(0.06, 0.35 * strongest)
    max_thickness = max(2, int(round(0.08 * h)))
    lines = find_lines(faint, max_thickness)
    if not lines:
        return line
    a = a.copy()
    for s, y in lines:
        centers = np.round(y + s * (np.arange(w) - w / 2)).astype(int)
        runs = []
        for x, c in enumerate(centers):
            c = next((c + d for d in (0, -1, 1) if 0 <= c + d < h and faint[c + d, x]), None)
            if c is None:
                runs.append(None)
                continue
            u, d = c, c
            while u > 0 and faint[u - 1, x]:
                u -= 1
            while d < h - 1 and faint[d + 1, x]:
                d += 1
            runs.append((c, u, d))
        lengths = [d - u + 1 for r in runs if r for _, u, d in [r]]
        thick = int(np.median([n for n in lengths if n <= 2 * max_thickness] or [max_thickness]))
        half = thick // 2 + 1
        for x, r in enumerate(runs):
            if r is None:
                continue
            c, u, d = r
            top, bottom = max(0, c - half), min(h - 1, c + half)
            if d - u + 1 <= thick + 2:
                top, bottom = u, d
            elif u < top and d > bottom:
                continue  # a stroke crosses the line
            above, below = max(0, top - 2), min(h - 1, bottom + 2)
            paper = max(a[above, x], a[below, x], float(background[top:bottom + 1, x].max()))
            a[top:bottom + 1, x] = paper
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


LEVEL_MIN_DRIFT = 1.5         # a line drifting by fewer rows over the width is left as it is


def level_lines(line: Image.Image) -> Image.Image:
    """
        `line` (grayscale word crop) sheared vertically so that the straight
        printed or ruled lines crossing it (find_lines) become horizontal, as
        normalize_word expects them; unchanged if there is none or if they
        are already nearly horizontal.
    """
    line = line.convert("L")
    small = line.reduce(-(-line.height // WORKING_HEIGHT)) if line.height > WORKING_HEIGHT else line
    a = np.asarray(small, np.float32)
    h, w = a.shape
    if w < 24:
        return line
    background = box_mean(a, max(4, h // 4))
    ink = np.clip((background - a) / np.maximum(0.5 * background, 1), 0, 1)
    strongest = np.percentile(ink, 99.5)
    if strongest < 0.08:
        return line
    found = find_lines(ink > 0.5 * max(0.06, 0.35 * strongest), max(2, int(round(0.08 * h))))
    if not found:
        return line
    s = found[0][0]
    if abs(s) * w < LEVEL_MIN_DRIFT:
        return line
    W, H = line.size
    paper = int(np.percentile(np.asarray(line), 75))
    # output (x, y) takes the input at (x, y + s (x - W / 2))
    return line.transform((W, H), Image.AFFINE, (1, 0, 0, s, 1, -s * W / 2), Image.BILINEAR, fillcolor=paper)
