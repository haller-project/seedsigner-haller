"""
    Locating the handwritten words of a seed on a photo.

    Two supports are recognized:
    * the official SeedSigner seed cards (12 words, or 24 words in two columns):
      the light inner area of the card is found, the card is rectified onto its
      known geometry and each word is cut along its printed line;
    * a free list written on any sheet (blank or ruled paper, one or two
      columns, numbered or not, 12 to 24 words): the page is straightened, the
      lines of text are found and the leading numbers are cut off.

    Detection runs on the photo reduced by half; the word crops are taken from
    the full resolution photo. The result is a WordLayout whose crops are only
    computed when asked for.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from . import native
from .preprocess import remove_long_runs, text_rows

SEED_LENGTHS = (12, 15, 18, 21, 24)

# Geometry of the official cards, in the coordinates of the card front rendered
# at 400 dpi (trading_card_21x21_w12words.pdf and trading_card_25x25_w24words.pdf
# from docs/seed_qr/printable_templates, front cropped to 1004 x 1401 pixels).
# Measured once on the blank renders:
# * CARD_LINES: (left, right, y) of each printed word line, in reading order;
# * CARD_CORNERS: corners of the light inner area of the card, found by
#   bright_area_corners on the blank render pasted on a dark background;
# * NUMBER_ENDS: right edge of the printed word number of each line (last
#   column darker than 128 within 90 pixels left of the line).
CARD_SIZE = (1004, 1401)
LINE_SPACING = 76.6
CARD_LINES = {
    "card12": [(180 if r < 9 else 221, 812, int(367 + LINE_SPACING * r)) for r in range(12)],
    "card24": ([(153 if r < 9 else 188, 474, int(367 + LINE_SPACING * r)) for r in range(12)]
               + [(621, 912, int(367 + LINE_SPACING * r)) for r in range(12)]),
}
CARD_CORNERS = {
    "card12": [(52.5, 82.9641944874943), (937.5, 77.05095647691337), (937.5, 1337.5), (52.5, 1337.5)],
    "card24": [(33.216765690521754, 57.66947577836612), (1002.5, 49.886513539717896),
               (1002.5, 1369.846406478308), (32.04473104554154, 1373.8910192421704)],
}
NUMBER_ENDS = {
    "card12": [154, 155, 154, 157, 153, 156, 156, 156, 156, 186, 186, 186],
    "card24": [118, 145, 147, 145, 149, 152, 152, 152, 152, 187, 187, 187]
              + [595, 620, 620, 620, 620, 620, 620, 620, 620, 620, 620, 620],
}

# Card search, on the half photo
CORNERS_TARGET_SIZE = 384    # side of the reduced image used to find the card corners
MIN_CARD_TILT = 4            # degrees: a card turned more than this is straightened before its corners are found
MIN_CARD_AREA = 0.03         # share of the photo
MIN_SURE_CARD_AREA = 0.08    # a smaller card is read with every word uncertain
CANDIDATE_THRESHOLDS = (0.85, 0.92, 1.0)
LINES_CHECK_SCALE = 0.35     # scale of the rectification used to look for the printed lines
MIN_CARD_LINES = 2 / 12      # share of printed lines found above which the photo is a card
MIN_CARD24_TRY = 0.5         # a 12-word card finding all its lines is kept unless the 24-word one finds this share
LINE_DARKNESS = 6            # darkness of a printed line (gray levels below the local mean)
FAINT_LINE_DARKNESS = 3      # the same for a blurred photo, if no candidate finds MIN_CARD24_TRY of its lines

# Registration of a card on its printed lines (register_card), on the rectification at LINES_CHECK_SCALE
REGISTRATION_MARGINS = (0.15, 3)  # room around the card where its lines are looked for: card width share, line spacings
STRIPS = 6                   # narrow columns along each line where its row is measured
MIN_STRIPS = 4               # a printed line crosses at least this many of them
MAX_ROW_SLOPE = 0.25         # slope of the lines on the first rectification
MIN_LINE_SHARE = 0.5         # a printed line is dark along this share of its right part
MIN_ROWS = 4                 # printed lines needed to register a card
MIN_CUT_ROWS = 6             # and to tell that it is cut off by the photo
MIN_LEFT_COLUMN_LINES = 3    # a 12-word card with this many lines of a left column is a 24-word card
TEXT_DARKNESS = 0.35         # a row whose 10th percentile of darkness is below this share of the median row: text

# Word crops on a card, in line spacings around the printed line: (top, bottom)
CARD_BAND = (0.95, 0.3)
CARD_TALL_BAND = (1.1, 0.2)
NUMBER_MARGIN = 6

# Free lists, on the half photo
LIST_REDUCTION = 2           # further reduction of the half photo for the ink mask
MIN_PAGE_ANGLE = 0.3         # degrees
MAX_PAGE_ANGLE = 30
LIST_MARGINS = (0.6, 0.35)   # horizontal, vertical margins of a line box, in line heights
FRAME_BORDER = 6             # pixels of the ink mask left out along the frame
SHEET_EDGE_DROP = 0.7        # a background darker than this share within ~14 pixels: edge of the sheet
PAPER_EROSION = 6            # pixels of paper left out along the edges of the sheet
EDGE_BAND = 10               # ink this close to the frame border: the list goes on beyond the photo
COLUMN_CORRIDORS = (1.5, 1.0)  # empty width between two columns, in line heights; the second if the first fails
WIDEN_STEP = 0.4             # a line box with ink within this share of the text height beyond a side is widened by it,
MAX_WIDENING = 4.0           # up to this many text heights
UNREAD_LINE = (1.5, 0.3)     # ink left out of the line boxes, at least this many line heights wide and this share
                             # of the ink of a median line: the number of lines is uncertain

# Expected duration of the steps of the search, in seconds on a Pi Zero, reported
# to `step(cost)` as they are done for a steady progress bar. SEARCH_COST is the
# whole search on a list (8 card masks tried in vain); the card is faster.
CARD_MASK_COST = 0.12
CARD_LINES_COST = 0.15
RECTIFY_COSTS = (0.6, 1.0)   # transform, then printed lines erased (all of them)
INK_MASK_COST = 0.85
PAGE_ANGLE_COST = 0.45
ROTATION_COST = 0.6
LINES_COST = 0.3
NUMBER_COST = 0.085          # per line
SEARCH_COST = 8 * CARD_MASK_COST + CARD_LINES_COST + INK_MASK_COST + PAGE_ANGLE_COST + ROTATION_COST + LINES_COST \
    + 12 * NUMBER_COST



class WordsNotFound(Exception):
    """
        No card was found on the photo, and no list with a valid number of lines.
        `at_edge`: the ink of the list reaches the edge of the photo, so lines are
        probably cut off by the frame.
        `line_count`: number of lines found in the list, if any.
        `card`: the words are those of a card, part of which is out of the photo.
    """
    def __init__(self, message: str, at_edge: bool = False, line_count: int = 0, card: bool = False):
        super().__init__(message)
        self.at_edge = at_edge
        self.line_count = line_count
        self.card = card



class WordLayout:
    """
        Words found on a photo.

        kind: "card12", "card24" or "list"
        count: number of words
        orders: possible reading orders (lists of word indices); the first one
            reads column by column, a second one row by row is given for two
            columns of equal length
        variants: number of crop variants per word (crop(k, variant))
        alternative: None, or a function returning the layout to read if this
            one gives no valid seed (a card found on part of its printed lines
            may be a ruled sheet); it raises WordsNotFound if there is none
        complete: False if text is left out of the crops, so that the number
            of words is uncertain, or if the card is too small in the photo:
            no word is then certain unless confirmed
    """
    def __init__(self, kind: str, count: int, orders: list[list[int]], variants: int, crop, outline,
                 alternative=None, complete: bool = True):
        self.kind = kind
        self.count = count
        self.orders = orders
        self.variants = variants
        self._crop = crop
        self._outline = outline
        self.alternative = alternative
        self.complete = complete


    def crop(self, k: int, variant: int = 0) -> Image.Image | None:
        """
            Grayscale crop of word k. Variant 0 is the first reading; on cards,
            variant 1 is a taller band and variant 2 a band widened to the
            printed number (None when the number leaves no room).
        """
        return self._crop(k, variant)


    def outline(self, k: int) -> list[tuple[float, float]]:
        """
            Corners of crop k (variant 0) on the photo, clockwise from the top
            left.
        """
        return self._outline(k)



def no_step(cost: float):
    """
        Default `step` function: the progress is not reported.
    """



def find_words(image: Image.Image, step=no_step) -> WordLayout:
    """
        Words of the card or list on `image` (upright grayscale photo, full
        resolution). Raises WordsNotFound if no card is found and no list with a
        valid number of lines either. `step(cost)` is called as the search goes
        (see SEARCH_COST).
    """
    return locate(image, image.reduce(2), step)



def locate(image: Image.Image, half: Image.Image, step=no_step) -> WordLayout:
    """
        find_words, with `half` the photo reduced by half (for example decoded
        at half size from a JPEG file).
    """
    quad, found, kind, threshold = find_card(half, step)
    if found > MIN_CARD_LINES:
        if found < 1.0 or on_frame(quad, half.size):
            # card found on part of its lines, or on the frame of the photo: registered on its printed lines
            registration = register_card(half, quad, kind, threshold)
            step(CARD_LINES_COST)
            if kind == "card12":
                # the right column of a 24-word card partly out of the photo looks like a 12-word card
                wide = right_column_card24(registration.quad if registration else quad)
                if left_column_lines(half, wide, threshold) >= MIN_LEFT_COLUMN_LINES:
                    kind, quad = "card24", wide
                    registration = register_card(half, quad, kind, threshold)
                step(CARD_LINES_COST)
            if registration:
                if registration.cut and registration.rows >= MIN_CUT_ROWS:
                    raise WordsNotFound("card cut off by the photo", at_edge=True, card=True)
                quad = registration.quad
        scale = image.width / half.width
        layout = card_layout(image, [(x * scale, y * scale) for x, y in quad], kind, step, threshold)
        if found < 1.0:
            layout.alternative = lambda: list_layout(image, half, step)
        # writing this small is read with no certain word
        layout.complete = bool(quad_area(quad) >= MIN_SURE_CARD_AREA * half.width * half.height)
        return layout
    return list_layout(image, half, step)



# ---------- image tools ----------

def local_mean(a: np.ndarray, radius: int) -> np.ndarray:
    """
        Mean over a (2r+1) x (2r+1) window, by Pillow's box blur on 8 bits.
    """
    image = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.BoxBlur(radius)), np.float32)


def local_max(a: np.ndarray, radius: int) -> np.ndarray:
    """
        Maximum over a (2r+1) x (2r+1) window, row then column (edges
        replicated).
    """
    for axis in (0, 1):
        p = np.pad(a, [(radius, radius) if k == axis else (0, 0) for k in range(2)], mode="edge")
        n = a.shape[axis]
        a = p[:n].copy() if axis == 0 else p[:, :n].copy()
        for k in range(1, 2 * radius + 1):
            np.maximum(a, p[k:k + n] if axis == 0 else p[:, k:k + n], out=a)
    return a


def otsu(values: np.ndarray) -> int:
    """
        Otsu threshold of 8-bit values.
    """
    hist = np.bincount(np.clip(values, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(float)
    total, cum, mcum = hist.sum(), hist.cumsum(), (hist * np.arange(256)).cumsum()
    w0 = cum / total
    m0 = mcum / np.maximum(cum, 1)
    m1 = (mcum[-1] - mcum) / np.maximum(total - cum, 1)
    return int(np.argmax(w0 * (1 - w0) * (m0 - m1) ** 2))


def remove_long_strokes(mask: np.ndarray, fraction: float = 0.25) -> np.ndarray:
    """
        Removes horizontal runs of ink longer than `fraction` of the width (and
        at least 40 pixels): ruled and printed lines, even slightly tilted.
    """
    return remove_long_runs(mask, max(40, int(fraction * mask.shape[1])))


def homography(src: list, dst: list) -> np.ndarray:
    """
        Pillow PERSPECTIVE coefficients mapping the points dst (output) to the
        points src (input); least squares fit beyond 4 points.
    """
    a, b = [], []
    for (u, v), (x, y) in zip(dst, src):
        a.append([u, v, 1, 0, 0, 0, -x * u, -x * v])
        b.append(x)
        a.append([0, 0, 0, u, v, 1, -y * u, -y * v])
        b.append(y)
    if len(src) > 4:
        return np.linalg.lstsq(np.array(a, float), np.array(b, float), rcond=None)[0]
    return np.linalg.solve(np.array(a, float), np.array(b, float))


def apply_homography(coefficients, points: list) -> list[tuple[float, float]]:
    """
        Images of `points` by Pillow PERSPECTIVE coefficients.
    """
    a, b, c, d, e, f, g, h = coefficients
    return [((a * u + b * v + c) / (g * u + h * v + 1), (d * u + e * v + f) / (g * u + h * v + 1)) for u, v in points]



# ---------- official cards ----------

def fit_line(t: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """
        Least squares line v = a t + b, outliers removed iteratively.
    """
    t, v = np.asarray(t, np.float64), np.asarray(v, np.float64)
    kept = np.ones(len(t), bool)
    for _ in range(4):
        tk, vk = t[kept], v[kept]
        if len(tk) < 2:
            raise ValueError("fewer than 2 points")
        dt = tk - tk.mean()
        den = float(dt @ dt)
        a = float(dt @ (vk - vk.mean())) / den if den > 0 else 0.0
        b = vk.mean() - a * tk.mean()
        residuals = np.abs(v - (a * t + b))
        kept = residuals < max(2.0, 2.5 * np.median(residuals[kept]))
    return a, b


def border_connected(mask: np.ndarray) -> np.ndarray:
    """
        Pixels of `mask` connected to the image border through pixels of the
        mask (4-connectivity), by the C library.
    """
    m = np.ascontiguousarray(mask, np.uint8)
    reached, work = np.empty_like(m), np.empty(m.size, np.int32)
    native.load().hw_border_connected(m.ctypes.data, m.shape[0], m.shape[1], reached.ctypes.data, work.ctypes.data)
    return reached.astype(bool)


def inner_area(mask: np.ndarray) -> np.ndarray:
    """
        Part of `mask` not connected to the image border (inside of the card or
        sheet, whatever the color of the table), or the whole mask if that part
        is too small.
    """
    inner = mask & ~border_connected(mask)
    return inner if inner.sum() > 0.02 * mask.size else mask


def area_corners(mask: np.ndarray, reduction: int) -> list | None:
    """
        Corners (top left, top right, bottom right, bottom left) of the area
        `mask`, as intersections of lines fitted on its four edges, in the
        coordinates of the image `reduction` times larger.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    cols = np.arange(int(x0 + 0.15 * (x1 - x0)), int(x1 - 0.15 * (x1 - x0)))
    rows = np.arange(int(y0 + 0.15 * (y1 - y0)), int(y1 - 0.15 * (y1 - y0)))
    if len(cols) < 5 or len(rows) < 5:
        return None
    mc, mr = mask[:, cols], mask[rows]
    kc, kr = mc.any(0), mr.any(1)
    top = np.argmax(mc, 0)[kc]
    bottom = (mask.shape[0] - 1 - np.argmax(mc[::-1], 0))[kc]
    left = np.argmax(mr, 1)[kr]
    right = (mask.shape[1] - 1 - np.argmax(mr[:, ::-1], 1))[kr]
    cols, rows = cols[kc], rows[kr]
    at, bt = fit_line(cols, top)
    ab, bb = fit_line(cols, bottom)
    al, bl = fit_line(rows, left)
    ar, br = fit_line(rows, right)

    def intersection(a1, b1, a2, b2):
        y = (a1 * b2 + b1) / (1 - a1 * a2)
        return a2 * y + b2, y

    corners = [intersection(at, bt, al, bl), intersection(at, bt, ar, br),
               intersection(ab, bb, ar, br), intersection(ab, bb, al, bl)]
    return [((x + 0.5) * reduction, (y + 0.5) * reduction) for x, y in corners]


def mask_corners(mask: np.ndarray, reduction: int) -> list:
    """
        Quadrilaterals of the area `mask`: area_corners, and if the main axis
        of the area is tilted by more than MIN_CARD_TILT degrees (a card turned
        in the photo), area_corners of the straightened area.
    """
    try:
        quads = [area_corners(mask, reduction)]
    except ValueError:
        quads = [None]
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return quads
    x, y = xs - xs.mean(), ys - ys.mean()
    # tilt of the main axis from the vertical, or from the horizontal
    tilt = math.degrees(0.5 * math.atan2(2 * float((x * y).mean()), float((x * x).mean() - (y * y).mean())))
    tilt = (tilt + 45) % 90 - 45
    if abs(tilt) >= MIN_CARD_TILT:
        try:
            quads.append(turned_corners(mask, reduction, tilt))
        except ValueError:
            pass
    return quads


def turned_corners(mask: np.ndarray, reduction: int, tilt: float) -> list | None:
    """
        area_corners of `mask` turned by `tilt` degrees, in the coordinates of
        `mask`.
    """
    h, w = mask.shape
    t = math.radians(tilt)
    cos, sin = math.cos(t), math.sin(t)
    size = int(abs(w * cos) + abs(h * sin)) + 2, int(abs(w * sin) + abs(h * cos)) + 2
    # straightened pixel (u, v) <- mask pixel (cos t (u - cu) - sin t (v - cv) + cx, sin t (u - cu) + cos t (v - cv) + cy)
    cu, cv, cx, cy = size[0] / 2, size[1] / 2, w / 2, h / 2
    coefficients = (cos, -sin, cx - cos * cu + sin * cv, sin, cos, cy - sin * cu - cos * cv)
    straight = Image.fromarray(mask.astype(np.uint8) * 255).transform(size, Image.AFFINE, coefficients, Image.NEAREST)
    corners = area_corners(np.asarray(straight) > 127, 1)
    if corners is None:
        return None
    a, b, c, d, e, f = coefficients
    return [((a * (u - 0.5) + b * (v - 0.5) + c + 0.5) * reduction, (d * (u - 0.5) + e * (v - 0.5) + f + 0.5) * reduction)
            for u, v in corners]


def card_ratio() -> float:
    """
        Width / height of the inner area of the 12-word card.
    """
    p = np.array(CARD_CORNERS["card12"])
    return ((np.hypot(*(p[0] - p[1])) + np.hypot(*(p[3] - p[2])))
            / (np.hypot(*(p[0] - p[3])) + np.hypot(*(p[1] - p[2]))))


def quad_area(quad: list) -> float:
    """
        Area of a quadrilateral.
    """
    p = np.array(quad)
    return 0.5 * abs(sum(p[i, 0] * p[(i + 1) % 4, 1] - p[(i + 1) % 4, 0] * p[i, 1] for i in range(4)))


def plausible(quad: list | None, size: tuple[int, int], ratio: float) -> bool:
    """
        Large enough quadrilateral with the proportions of the card.
    """
    if quad is None:
        return False
    p = np.array(quad)
    side = lambda i, j: np.hypot(*(p[i] - p[j]))
    width = (side(0, 1) + side(3, 2)) / 2
    height = (side(0, 3) + side(1, 2)) / 2
    if quad_area(quad) < MIN_CARD_AREA * size[0] * size[1] or height <= 0:
        return False
    return abs(width / height / ratio - 1) < 0.15


def card_candidates(photo: Image.Image, step=no_step):
    """
        Plausible quadrilaterals for the light inner area of the card, best
        guess first: lighting-corrected threshold, then global Otsu threshold,
        then other thresholds of the lighting-corrected image, alone or combined
        with the Otsu threshold (a dark table that the lighting correction makes
        look light). Duplicates (all corners within 2 % of the photo size of an
        earlier candidate) are skipped. Candidates are computed on demand.
    """
    ratio = card_ratio()
    tolerance = 0.02 * max(photo.size)
    returned = []

    def new(quad):
        if not plausible(quad, photo.size, ratio):
            return False
        if any(max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(quad, q)) < tolerance for q in returned):
            return False
        returned.append(quad)
        return True

    reduction = max(1, -(-max(photo.size) // CORNERS_TARGET_SIZE))
    small = np.asarray(photo.reduce(reduction), np.float32)
    normalized = np.clip(small / np.maximum(local_mean(small, max(6, min(small.shape) // 10)), 1) * 128, 0, 255)
    light = small > otsu(small)

    # first candidate: as found on the blank card for CARD_CORNERS
    quads = []
    for mask in (normalized > 0.92 * 128, light):
        quads = mask_corners(inner_area(mask), reduction)
        step(CARD_MASK_COST)
        if any(plausible(quad, photo.size, ratio) for quad in quads):
            break
    for k in [None] + list(CANDIDATE_THRESHOLDS):
        if k is not None:
            quads = [quad for mask in (normalized > k * 128, (normalized > k * 128) & light)
                     for quad in mask_corners(inner_area(mask), reduction)]
            step(2 * CARD_MASK_COST)
        for quad in quads:
            if new(quad):
                yield quad


def row_percentile_max(zone: np.ndarray) -> float:
    """
        Maximum over the rows of the 30th percentile of each row.
    """
    if zone.size == 0:
        return 0.0
    k = int(0.3 * (zone.shape[1] - 1))
    return float(np.partition(zone, k, axis=1)[:, k].max())


def count_card_lines(photo: Image.Image, quad: list, kind: str, threshold: float = LINE_DARKNESS) -> int:
    """
        Number of printed lines of card `kind` found where expected once `quad`
        is rectified: a line is dark along the right half of the word line, and
        light after its end and in the gap between the columns (the lines of a
        ruled sheet cross the whole page).
    """
    f = LINES_CHECK_SCALE
    w, h = CARD_SIZE
    try:
        coefficients = homography(quad, [(x * f, y * f) for x, y in CARD_CORNERS[kind]])
    except np.linalg.LinAlgError:
        return 0
    a = np.asarray(photo.transform((int(w * f), int(h * f)), Image.PERSPECTIVE, coefficients, Image.BILINEAR),
                   np.float32)
    darkness = local_mean(a, max(4, int(round(15 * f)))) - a
    spacing = LINE_SPACING * f
    lines = [(x0 * f, x1 * f, y * f) for x0, x1, y in CARD_LINES[kind]]
    width = w * f
    gap = None
    if any(x0 >= width / 2 for x0, _, _ in lines):
        gap = (max(x1 for x0, x1, _ in lines if x0 < width / 2), min(x0 for x0, _, _ in lines if x0 >= width / 2))
    found = 0
    for x0, x1, y in lines:
        ya, yb = max(0, int(y - 0.45 * spacing)), int(y + 0.45 * spacing)
        dark = lambda xa, xb: row_percentile_max(darkness[ya:yb, int(xa):int(xb)]) > threshold
        line = dark(x0 + 0.5 * (x1 - x0), x1 - 0.05 * (x1 - x0))
        line = line and not dark(x1 + 0.04 * width, min(width - 0.03 * width, x1 + 0.12 * width))
        if line and gap:
            line = not dark(gap[0] + 0.2 * (gap[1] - gap[0]), gap[1] - 0.2 * (gap[1] - gap[0]))
        found += line
    return found


def find_card(photo: Image.Image, step=no_step) -> tuple[list | None, float, str, float]:
    """
        (quadrilateral, share of printed lines found, card kind, line darkness
        threshold) of the best card candidate. The 24-word card wins as soon as
        it finds 75 % of its lines (its left column also fills the positions of
        the 12-word card), over any 12-word candidate. Stops at the first
        candidate with all its lines, unless it is a 12-word card on which the
        24-word card is plausible. If no candidate finds MIN_CARD24_TRY of its
        lines, the lines are counted again with FAINT_LINE_DARKNESS (printed
        lines spread out by a blurred photo).
    """
    candidates = []

    def remembered(quads):
        for quad in quads:
            candidates.append(quad)
            yield quad

    def best_of(quads, threshold):
        best = (None, 0.0, "card12", threshold)
        for quad in quads:
            share = {"card12": count_card_lines(photo, quad, "card12", threshold) / 12, "card24": 0.0}
            if share["card12"] > MIN_CARD_LINES:
                share["card24"] = count_card_lines(photo, quad, "card24", threshold) / 24
            step(CARD_LINES_COST)
            kind = "card24" if share["card24"] >= 0.75 else "card12"
            if (kind == "card24", share[kind]) > (best[2] == "card24", best[1]):
                best = (quad, share[kind], kind, threshold)
            if share[kind] >= 1.0 and (kind == "card24" or share["card24"] < MIN_CARD24_TRY):
                break
        return best

    best = best_of(remembered(card_candidates(photo, step)), LINE_DARKNESS)
    if best[1] < MIN_CARD24_TRY and candidates:
        faint = best_of(candidates, FAINT_LINE_DARKNESS)
        if faint[1] >= MIN_CARD24_TRY:
            best = faint
    return best




class CardRegistration:
    """
        Result of register_card.
        quad: corrected quadrilateral of the inner area of the card
        cut: a word line of the card falls outside the photo
        rows: number of printed rows found
    """
    def __init__(self, quad: list, cut: bool, rows: int):
        self.quad = quad
        self.cut = cut
        self.rows = rows



def card_rows(darkness: np.ndarray, spans: list, spacing: float, threshold: float) -> tuple[float, list]:
    """
        Printed rows of the rectified card `darkness`: in STRIPS narrow columns
        along each span (x0, x1), rows darker than `threshold`; grouped along
        the slope that aligns them best (within MAX_ROW_SLOPE), then a line
        fitted on each group, kept if it crosses at least MIN_STRIPS columns.
        Returns (xc, [(y at xc, slope, [columns])]), rows top to bottom.
    """
    xs, ys = [], []
    for x0, x1 in spans:
        half = max(2, int(0.025 * (x1 - x0)))
        for x in np.linspace(x0, x1, STRIPS):
            zone = darkness[:, int(x) - half:int(x) + half + 1]
            k = int(0.3 * (zone.shape[1] - 1))
            p = np.partition(zone, k, axis=1)[:, k]
            peaks = np.nonzero((p[1:-1] > threshold) & (p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]))[0] + 1
            xs += [float(x)] * len(peaks)
            ys += peaks.tolist()
    if len(xs) < MIN_STRIPS:
        return 0.0, []
    x, y = np.array(xs), np.array(ys, float)
    xc = float(x.mean())
    width = max(3, int(0.15 * spacing)) | 1

    def histogram(slope):
        c = np.round(y - slope * (x - xc)).astype(int)
        lo = int(c.min())
        h = np.convolve(np.bincount(c - lo).astype(float), np.ones(width), mode="same")
        return h, lo

    slope = max(np.arange(-MAX_ROW_SLOPE, MAX_ROW_SLOPE + 1e-9, 0.01), key=lambda s: float((histogram(s)[0] ** 2).sum()))
    h, lo = histogram(slope)
    c = y - slope * (x - xc)
    rows = []
    peaks = [m for m in range(len(h)) if h[m] >= MIN_STRIPS and h[m] >= h[max(0, m - 1)] and h[m] >= h[min(len(h) - 1, m + 1)]]
    for m in sorted(peaks, key=lambda m: -h[m]):
        if any(abs(m + lo - r[0]) < 0.5 * spacing for r in rows):
            continue
        near = np.abs(c - (m + lo)) <= 0.35 * spacing
        if len(set(x[near].tolist())) < MIN_STRIPS:
            continue
        # each line its own slope (the first rectification may leave them fanned out): the line through two
        # of the peaks that passes closest to a peak in the most columns
        line = row_line(x[near] - xc, y[near], max(2.0, 0.08 * spacing))
        if line and len(line[2]) >= MIN_STRIPS and all(abs(line[0] - r[0]) >= 0.5 * spacing for r in rows):
            rows.append((line[0], line[1], sorted(xc + v for v in line[2])))
    return xc, sorted(rows)


def median_line(t: list, v: list) -> tuple[float, float]:
    """
        Theil-Sen line v = a t + b (median of the slopes between pairs of
        points), robust to a third of outliers; (0, -1e9) for fewer than 2
        points.
    """
    slopes = [(v[j] - v[i]) / (t[j] - t[i]) for i in range(len(t)) for j in range(i + 1, len(t)) if t[j] != t[i]]
    if not slopes:
        return 0.0, -1e9
    a = float(np.median(slopes))
    return a, float(np.median([y - a * x for x, y in zip(t, v)]))


def row_line(x: np.ndarray, y: np.ndarray, tolerance: float) -> tuple[float, float, list] | None:
    """
        (y at x = 0, slope, columns) of the line through two of the points
        (x, y) crossing the most distinct columns x within `tolerance` of a
        point (then the smallest sum of distances), refined by least squares
        on them.
    """
    dx = x[None, :] - x[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(dx != 0, (y[None, :] - y[:, None]) / dx, np.inf)
    i, j = np.nonzero(np.triu(np.abs(slope) <= MAX_ROW_SLOPE + 0.1, 1))
    if len(i) == 0:
        return None
    s = slope[i, j]
    distance = np.abs(y[None, :] - (y[i][:, None] + s[:, None] * (x[None, :] - x[i][:, None])))
    on = distance <= tolerance
    columns = np.unique(x, return_inverse=True)[1]
    crossed = (on.astype(np.int32) @ np.eye(columns.max() + 1, dtype=np.int32)[columns] > 0).sum(1)
    score = crossed * 1e6 - np.where(on, distance, 0).sum(1)
    on = on[int(np.argmax(score))]
    slope, intercept = np.polyfit(x[on], y[on], 1)
    return float(intercept), float(slope), sorted(set(x[on].tolist()))


def on_frame(quad: list, size: tuple[int, int]) -> bool:
    """
        A corner of `quad` is on the frame of the photo of `size`, or beyond.
    """
    return any(min(x, y, size[0] - x, size[1] - y) < 0.01 * max(size) for x, y in quad)


def right_column_card24(quad: list) -> list:
    """
        Quadrilateral of the 24-word card whose right column is the 12-word
        card `quad`.
    """
    (x0, _, _), (x1, _, _) = CARD_LINES["card12"][0], CARD_LINES["card24"][12]
    scale = (CARD_LINES["card12"][0][1] - x0) / (CARD_LINES["card24"][12][1] - x1)
    to_photo = homography(quad, CARD_CORNERS["card12"])
    return apply_homography(to_photo, [(x0 + (x - x1) * scale, y) for x, y in CARD_CORNERS["card24"]])


def left_column_lines(photo: Image.Image, quad: list, threshold: float = LINE_DARKNESS) -> int:
    """
        Lines of the left column of the 24-word card `quad` found where
        expected, with light paper between them and the numbers of the right
        column (not the border of a neighboring card).
    """
    f = LINES_CHECK_SCALE
    w, h = CARD_SIZE
    try:
        coefficients = homography(quad, [(x * f, y * f) for x, y in CARD_CORNERS["card24"]])
    except np.linalg.LinAlgError:
        return 0
    a = np.asarray(photo.transform((int(w * f), int(h * f)), Image.PERSPECTIVE, coefficients, Image.BILINEAR),
                   np.float32)
    darkness = local_mean(a, max(4, int(round(15 * f)))) - a
    spacing = LINE_SPACING * f
    paper = float(np.percentile(a, 90))
    found = 0
    for x0, x1, y in CARD_LINES["card24"][:12]:
        ya, yb = max(0, int((y - 0.45 * LINE_SPACING) * f)), int((y + 0.45 * LINE_SPACING) * f)
        line = row_percentile_max(darkness[ya:yb, int((x0 + 0.5 * (x1 - x0)) * f):int((x1 - 0.05 * (x1 - x0)) * f)])
        gap = a[ya:yb, int((x1 + 8) * f):int((x1 + 56) * f)]
        found += line > threshold and gap.size > 0 and float(np.median(gap)) > 0.8 * paper
    return found


def register_card(photo: Image.Image, quad: list, kind: str, threshold: float = LINE_DARKNESS,
                  turn: bool = True, again: bool = True) -> CardRegistration | None:
    """
        Corrects `quad` from the printed lines of the card, so that the
        registration does not depend on the card edges (an edge out of the
        photo, a card among others): the lines are found around their expected
        place, numbered by their regular spacing, and the first one is told by
        the lines missing where the photo shows the card and by the ends of the
        lines (lines 10 to 12 start further right); if no edge of `quad` is the
        frame of the photo, also by their place on `quad`. A card upside down
        is turned. None if fewer than MIN_ROWS lines are found, if they are not
        spaced and ended as those of a card (ruled sheet, text), or if the
        first one is not told apart. If lines are missing, the search is done
        again once on the corrected rectification (`again`); `turn`: False
        once the card is turned.
    """
    f = LINES_CHECK_SCALE
    w, h = CARD_SIZE
    mx, my = REGISTRATION_MARGINS[0] * w, REGISTRATION_MARGINS[1] * LINE_SPACING
    size = (int((w + 2 * mx) * f), int((h + 2 * my) * f))
    try:
        to_photo = homography(quad, [((x + mx) * f, (y + my) * f) for x, y in CARD_CORNERS[kind]])
    except np.linalg.LinAlgError:
        return None
    fill = int(np.median(np.asarray(photo.reduce(8))))
    a = np.asarray(photo.transform(size, Image.PERSPECTIVE, to_photo, Image.BILINEAR, fillcolor=fill), np.float32)
    darkness = local_mean(a, max(4, int(round(15 * f)))) - a
    spacing = LINE_SPACING * f
    # darkness below the paper (local maximum), which handwriting next to a line does not lower
    smooth = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(1)), np.float32)
    below_paper = local_max(smooth, int(0.2 * spacing)) - smooth
    lines = CARD_LINES[kind]
    count = 12  # rows of the card; the 24-word card has two lines per row
    columns = [lines[c * count:(c + 1) * count] for c in range(len(lines) // count)]
    # right part of the lines, mostly free of handwriting
    spans = [((col[0][0] + 0.45 * (col[0][1] - col[0][0]) + mx) * f, (col[0][1] - 0.03 * (col[0][1] - col[0][0]) + mx) * f)
             for col in columns]

    # upside down: ink where the numbers are printed and none after the ends of the lines, on the card turned
    # rather than as it is
    def number_contrast(d):
        def mean(x0, x1, y0, y1):
            zone = d[int((y0 + my) * f):int((y1 + my) * f), int((x0 + mx) * f):int((x1 + mx) * f)]
            return float(zone.mean()) if zone.size else 0.0
        return sum(mean(n - 60, n, y - 0.6 * LINE_SPACING, y) - mean(x1 + 15, x1 + 60, y - 0.6 * LINE_SPACING, y)
                   for (_, x1, y), n in zip(lines, NUMBER_ENDS[kind]))

    ink = np.clip(darkness, 0, None)
    if turn and number_contrast(ink[::-1, ::-1]) > max(0.0, 1.5 * number_contrast(ink)):
        return register_card(photo, quad[2:] + quad[:2], kind, threshold, False, again)
    xc, rows = card_rows(darkness, spans, spacing, threshold)
    if len(rows) < 3:
        return None

    def inside(points):
        return all(-2 <= x <= photo.width + 2 and -2 <= y <= photo.height + 2
                   for x, y in apply_homography(to_photo, points))

    in_photo = np.asarray(Image.new("L", photo.size, 255).transform(size, Image.PERSPECTIVE, to_photo,
                                                                     Image.NEAREST)) > 0

    def end(c, slope, x, direction, limit):
        """ last column along the row (c, slope) from x darker than the paper (gaps of 2 columns at most);
            minus the column where the row leaves the photo, if first; None if it is still dark after
            `limit` columns or leaves the rectification """
        xx = np.round(x + direction * np.arange(1, int(limit))).astype(int)
        yy = np.round(c + slope * (xx - xc)).astype(int)
        valid = (xx >= 0) & (xx < size[0]) & (yy >= 1) & (yy < size[1] - 1)
        n = int(np.argmin(valid)) if not valid.all() else len(xx)
        xx, yy = xx[:n], yy[:n]
        dark = np.maximum(np.maximum(below_paper[yy - 1, xx], below_paper[yy, xx]), below_paper[yy + 1, xx]) > threshold
        light = ~dark
        # first run of 3 light columns
        runs = light[:-2] & light[1:-1] & light[2:] if len(light) > 2 else np.zeros(0, bool)
        stop = int(np.argmax(runs)) if runs.any() else len(xx)
        out = ~in_photo[yy, xx]
        if out[:stop].any():
            return -int(xx[int(np.argmax(out))])
        if stop == len(xx):
            return None
        before = np.nonzero(dark[:stop])[0]
        return int(xx[before[-1]]) if len(before) else int(round(x))

    def printed_line(c, slope, xa, xb):
        """ the row (c, slope) is a thin line from xa to xb: dark along MIN_LINE_SHARE of it, and lighter a
            fifth of a spacing above and below (not text, not the edge of the card) """
        xx = np.arange(int(xa), int(xb))
        yy = np.clip(np.round(c + slope * (xx - xc)).astype(int), 1, size[1] - 2)
        line = np.maximum(np.maximum(darkness[yy - 1, xx], darkness[yy, xx]), darkness[yy + 1, xx])
        d = int(round(0.2 * spacing))
        thin = [float(np.median(below_paper[np.clip(yy + o, 0, size[1] - 1), xx])) for o in (-1, 0, 1)]
        around = [float(np.median(below_paper[np.clip(yy + o, 0, size[1] - 1), xx])) for o in (-d, d)]
        return float(np.mean(line > threshold)) >= MIN_LINE_SHARE and max(around) < 0.5 * max(thin)

    def clear_after(j, x):
        """ paper after the end x of row j, unlike text going on after a space """
        c, slope, _ = rows[j]
        xx = np.arange(int(x) + 3, min(size[0], int(x + 0.04 * w * f)))  # before the numbers of a second column
        if len(xx) == 0:
            return False
        yy = np.clip(np.round(c + slope * (xx - xc)).astype(int), 1, size[1] - 2)
        dark = np.maximum(np.maximum(below_paper[yy - 1, xx], below_paper[yy, xx]), below_paper[yy + 1, xx])
        return float(np.mean(dark > threshold)) < 0.2

    # ends of the printed lines per row and column (canvas abscissas; None: not found; negative: the row
    # leaves the photo at minus this abscissa)
    measured = []
    for c, slope, strips in rows:
        row_ends = {}
        for k, (xa, xb) in enumerate(spans):
            crossed = [s for s in strips if xa - 1 <= s <= xb + 1]
            if len(crossed) >= 2 and printed_line(c, slope, xa, xb):
                length = (columns[k][0][1] - columns[k][0][0]) * f
                xl = end(c, slope, crossed[0], -1, 0.6 * length)
                xr = end(c, slope, crossed[-1], 1, 0.3 * length)
                # a line goes on a little after the last strip, to its end
                row_ends[k] = [xl if xl is not None and crossed[0] - xl > 0.15 * length else None,
                               None if xr is not None and 0 < xr - crossed[-1] < 1 else xr]
        measured.append(row_ends)

    # text is darker than the paper along a smaller share of the row than the printed lines are
    def low_darkness(c, slope, xa, xb):
        xx = np.arange(int(xa), int(xb))
        yy = np.clip(np.round(c + slope * (xx - xc)).astype(int), 1, size[1] - 2)
        return float(np.percentile(np.maximum(np.maximum(below_paper[yy - 1, xx], below_paper[yy, xx]),
                                              below_paper[yy + 1, xx]), 10))

    level = [max((low_darkness(c, slope, *spans[k]) for k in e), default=0.0) for (c, slope, _), e in zip(rows, measured)]
    reference = float(np.median([v for v, e in zip(level, measured) if e] or [0.0]))
    for v, e in zip(level, measured):
        if v < TEXT_DARKNESS * reference:
            e.clear()
    endless = sum(1 for e in measured if e and all(v[1] is None for v in e.values()))
    # a row going on beyond the end of the line in one of its columns is text
    textual = {j for j, e in enumerate(measured) if any(v[1] is None for v in e.values())}
    # the lines of a column end on a straight line: rows ending elsewhere are text or a card edge; a row
    # leaving the photo before that line is a line cut by the frame
    for k, (xa, xb) in enumerate(spans):
        js = [j for j, e in enumerate(measured) if k in e and e[k][1] is not None and e[k][1] > 0]
        slope_r, xr0 = median_line([rows[j][0] for j in js], [measured[j][k][1] for j in js])
        tolerance = 0.03 * (columns[k][0][1] - columns[k][0][0]) * f
        for j, e in enumerate(measured):
            xr = e[k][1] if k in e else 0
            expected = slope_r * rows[j][0] + xr0
            if k in e and (xr is None or (xr > 0 and (abs(xr - expected) > tolerance or not clear_after(j, xr)))
                           or (xr < 0 and -xr > expected + tolerance)):
                del e[k]
    keep = [j for j, e in enumerate(measured) if e and j not in textual]
    rows = [rows[j] for j in keep]
    ends = [(i, k, xl if xl is not None and xl > 0 else None, xr if xr > 0 else None)
            for i, j in enumerate(keep) for k, (xl, xr) in measured[j].items()]
    if len(rows) < 3 or endless > len(rows):
        return None  # lines going on beyond their end: a ruled sheet

    # rows numbered by their spacing, measured between neighbors (it varies with the perspective)
    c = np.array([r[0] for r in rows])
    d = np.diff(c)
    single = d < 1.5 * d.min()
    centers = (c[:-1] + c[1:]) / 2
    rel = [0]
    for i in range(len(d)):
        near = np.argsort(np.abs(centers[single] - centers[i]))[:2]
        rel.append(rel[-1] + max(1, int(round(d[i] / float(np.mean(d[single][near]))))))
    rel = np.array(rel)
    # too many rows for a card: a first or last row with no end of line found is the frame of the photo
    while rel[-1] >= count and len(rows) > 3:
        ended = [any(xr is not None for i, _, _, xr in ends if i == j) for j in (0, len(rows) - 1)]
        if ended == [True, True]:
            break
        j = 0 if not ended[0] else len(rows) - 1
        rows, c, rel = rows[:j] + rows[j + 1:], np.delete(c, j), np.delete(rel, j)
        rel = rel - rel[0]
        ends = [(i - (i > j), k, xl, xr) for i, k, xl, xr in ends if i != j]
        d = np.diff(c)
        single = d < 1.5 * d.min()
        centers = (c[:-1] + c[1:]) / 2
    if any(abs(d[i] / float(np.mean(d[single][np.argsort(np.abs(centers[single] - centers[i]))[:2]]))
               - (rel[i + 1] - rel[i])) > 0.35 for i in range(len(d))):
        return None  # rows off the grid
    pitch = float(np.median(d[single]))
    if len(rows) < MIN_ROWS or not 0.65 < pitch / spacing < 1.5:
        return None  # too few rows, or not spaced as the lines of the card

    def fit(points):
        """ least squares v = p . (X, Y, 1), outliers beyond 0.02 card width dropped; (p, inliers) """
        m = np.array([(X, Y, 1.0) for X, Y, _ in points])
        v = np.array([v for _, _, v in points])
        kept = np.ones(len(v), bool)
        if len(set(m[:, 0])) < 2:  # the same template abscissa everywhere: the scale is kept
            m, v0 = m[:, 1:], v - f * m[:, 0]
        else:
            v0 = v
        for _ in range(3):
            p = np.linalg.lstsq(m[kept], v0[kept], rcond=None)[0]
            residuals = np.abs(m @ p - v0)
            kept = residuals < 0.02 * w * f
            if kept.sum() < 2:
                return None, 0
        if len(p) == 2:
            p = np.array([f, p[0], p[1]])
        return p, int(kept.sum())

    # first line: the one leaving the fewest lines missing where the photo shows the card, then the one whose
    # lines are closest to their place on `quad` if none of its edges is the frame of the photo, then the one
    # agreeing best with the left ends of the lines
    ys = [y for _, _, y in lines[:count]]
    clipped = on_frame(quad, photo.size)
    position = c / f - my  # rows on the template, as placed by `quad`

    def offset(first):
        """ distance of the rows numbered from `first` to their place on `quad`; a row off the card counts for
            two spacings """
        return sum(abs(p - ys[r + first]) if 0 <= r + first < count else 2 * LINE_SPACING for p, r in zip(position, rel))

    if rel[-1] >= count:
        # more rows than lines: text at one end, told by `quad`
        if clipped:
            return None
        first = min(range(2 - int(rel[-1]), count - 2), key=offset)
        on_card = [j for j in range(len(rows)) if 0 <= rel[j] + first < count]
        if len(on_card) < 3:
            return None
        renumber = {j: i for i, j in enumerate(on_card)}
        ends = [(renumber[j], k, xl, xr) for j, k, xl, xr in ends if j in renumber]
        rows, c, position = [rows[j] for j in on_card], c[on_card], position[on_card]
        rel = rel[on_card] - rel[on_card[0]]
    firsts = range(0, count - rel[-1])
    results = []
    for first in firsts:
        index = rel + first
        # x: ends of the lines; y: rows (polynomial of Y, slope linear in Y)
        points = [(lines[index[j] + 12 * k][0], ys[index[j]], xl) for j, k, xl, _ in ends if xl is not None] + \
                 [(lines[index[j] + 12 * k][1], ys[index[j]], xr) for j, k, _, xr in ends if xr is not None]
        px, consistent = fit(points)
        if px is None:
            continue
        yy = np.array([ys[i] for i in index], float)
        py = np.polyfit(yy, c, 2 if len(rows) >= 5 else 1)
        ps = np.polyfit(yy, [r[1] for r in rows], 1)

        def to_canvas(X, Y, px=px, py=py, ps=ps):
            x = px[0] * X + px[1] * Y + px[2]
            return x, float(np.polyval(py, Y) + np.polyval(ps, Y) * (x - xc))

        missing = 0
        for r in range(count):
            if r in index:
                continue
            row = [to_canvas(x, ys[r]) for x in np.linspace(lines[r][0], lines[r][1], 4)]
            missing += inside(row)
        results.append((missing, -consistent, first, to_canvas))
    if not results:
        return None
    results.sort(key=lambda t: (t[0], offset(t[2]) if not clipped else 0, t[1]))
    best = results[0]
    # as many lines missing and left ends agreeing: looked for again on the next rectification
    tie = clipped and len(results) > 1 and results[1][:2] == best[:2]
    if tie and not again:
        return None
    to_canvas = best[3]
    grid = [(x, y) for y in ys for x in np.linspace(min(x0 for x0, _, _ in lines), max(x1 for _, x1, _ in lines), 4)]
    to_card = homography(apply_homography(to_photo, [to_canvas(x, y) for x, y in grid]), grid)
    cut = any(not all(-2 <= x <= photo.width + 2 and -2 <= y <= photo.height + 2 for x, y in
                      apply_homography(to_card, [(x0, y - 0.7 * LINE_SPACING), (x1, y - 0.7 * LINE_SPACING),
                                                 (x1, y + 0.1 * LINE_SPACING), (x0, y + 0.1 * LINE_SPACING)]))
              for x0, x1, y in lines)
    registration = CardRegistration(apply_homography(to_card, CARD_CORNERS[kind]), cut, len(rows))
    if again and (tie or len(rows) < count):
        # lines missed on the first rectification are looked for again on the corrected one
        second = register_card(photo, registration.quad, kind, threshold, False, False)
        return second if second or tie else registration
    return registration


def adjust_line(darkness: np.ndarray, x0: int, x1: int, y: int, threshold: float = LINE_DARKNESS) -> int:
    """
        Actual row of a printed line, within 0.45 line spacing of its template
        position: the row dark along 70 % of the right half of the line.
    """
    ya, yb = int(y - 0.45 * LINE_SPACING), int(y + 0.45 * LINE_SPACING)
    zone = darkness[max(0, ya):yb, int(x0 + 0.5 * (x1 - x0)):int(x1 - 0.05 * (x1 - x0))]
    if zone.size == 0:
        return y
    v = np.percentile(zone, 30, axis=1)
    k = int(np.argmax(v))
    return max(0, ya) + k if v[k] > threshold else y


def erase_line(a: np.ndarray, darkness: np.ndarray, background: np.ndarray, xa: int, xb: int, y: int, r: int = 7,
               threshold: float = LINE_DARKNESS):
    """
        Replaces a printed line by the local background, column by column: the
        darkest row within r pixels of y and its neighbors.
    """
    ya, yb = max(0, y - r), min(a.shape[0], y + r + 1)
    xa, xb = max(0, xa), min(a.shape[1], xb)
    zone = darkness[ya:yb, xa:xb]
    if zone.size == 0:
        return
    dark = zone.max(0) > threshold
    xs = np.arange(xa, xb)[dark]
    ys = ya + np.argmax(zone, 0)[dark]
    for d in range(-2, 3):
        yy = np.clip(ys + d, 0, a.shape[0] - 1)
        a[yy, xs] = background[yy, xs]


def card_layout(photo: Image.Image, quad: list, kind: str, step=no_step,
                threshold: float = LINE_DARKNESS) -> WordLayout:
    """
        Card rectified onto its template (on the first crop), printed lines
        (darker than `threshold`, see find_card) erased; crops along the
        printed lines.
    """
    lines, number_ends = CARD_LINES[kind], NUMBER_ENDS[kind]
    coefficients = homography(quad, CARD_CORNERS[kind])
    page = []

    def rectified():
        if not page:
            card = photo.transform(CARD_SIZE, Image.PERSPECTIVE, coefficients, Image.BILINEAR)
            step(RECTIFY_COSTS[0])
            a = np.asarray(card, np.float32)
            background = local_mean(a, 15)
            darkness = background - a
            a = a.copy()
            ys = []
            for x0, x1, y in lines:
                ys.append(adjust_line(darkness, x0, x1, y, threshold))
                erase_line(a, darkness, background, x0 - 4, x1 + 6, ys[-1], threshold=threshold)
                step(RECTIFY_COSTS[1] / len(lines))
            page.extend([Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), ys])
        return page

    def box(k, variant):
        x0, x1, _ = lines[k]
        left = x0 - 4
        top, bottom = CARD_BAND if variant == 0 else CARD_TALL_BAND
        if variant == 2:
            left = number_ends[k] + NUMBER_MARGIN
            if left >= x0 - 4:
                return None
        y = rectified()[1][k]
        return left, int(y - top * LINE_SPACING), x1 + 6, int(y + bottom * LINE_SPACING)

    def crop(k, variant):
        b = box(k, variant)
        return rectified()[0].crop(b) if b else None

    def outline(k):
        x0, y0, x1, y1 = box(k, 0)
        return apply_homography(coefficients, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))

    return WordLayout(kind, len(lines), [list(range(len(lines)))], 3, crop, outline)



# ---------- free lists ----------

def ink_mask(photo: Image.Image) -> np.ndarray | None:
    """
        Ink of the sheet on the photo reduced by LIST_REDUCTION, or None if no
        sheet is found. The edges of the sheet (sharp drops of the background,
        whatever the lighting) and long horizontal strokes (ruled lines, table
        edge) are removed.
    """
    a = np.asarray(photo.reduce(LIST_REDUCTION), np.float32)
    background = local_mean(a, 12)
    t = otsu(background.ravel())
    low, high = background[background <= t], background[background > t]
    if len(low) and len(high) and low.mean() > 0.5 * high.mean():
        paper = np.ones_like(background, bool)  # no dark surroundings: the sheet fills the frame
    else:
        paper = background > t
    b = FRAME_BORDER
    paper[:b], paper[-b:], paper[:, :b], paper[:, -b:] = False, False, False, False
    nearby = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8)).reduce(4).filter(ImageFilter.MinFilter(7))
    paper &= np.asarray(nearby.resize(background.shape[::-1], Image.NEAREST), np.float32) > SHEET_EDGE_DROP * background
    # erosion, as a box blur and a threshold
    paper = local_mean(paper.astype(np.float32) * 255, PAPER_EROSION) > 254
    if not paper.any():
        return None
    # ink contrast, measured on the light paper only
    light = paper & (background > 0.7 * np.median(background[paper]))
    inside = a[light] if light.any() else a[paper]
    contrast = float(np.percentile(inside, 1) / max(np.median(inside), 1))
    ink = (a < ink_threshold(contrast) * background) & paper
    paper_width = max(int(paper.any(0).sum()), 1)
    return remove_long_strokes(ink, 0.3 * paper_width / ink.shape[1])


def ink_at_edge(ink: np.ndarray) -> bool:
    """
        Ink within EDGE_BAND pixels of the frame side of the area where ink_mask
        looks for it: the text probably goes on beyond the photo.
    """
    m, n = FRAME_BORDER + PAPER_EROSION, EDGE_BAND
    sides = (ink[m:m + n], ink[-m - n:-m], ink[:, m:m + n], ink[:, -m - n:-m])
    return any(side.sum() >= 10 for side in sides)


def ink_threshold(contrast: float) -> float:
    """
        Ink is darker than this share of the local background: adapted to the
        ink / paper contrast, from faint pencil to dark marker.
    """
    return min(max(1 - 0.4 * (1 - contrast), 0.72), 0.9)


def page_angle(ink: np.ndarray) -> float:
    """
        Tilt of the lines of text (degrees): the angle whose vertical shear
        makes the horizontal projection of the ink the most contrasted (sum of
        squares); 1.5 degree steps, then refined to 0.5 and 0.1 degree.
    """
    ys, xs = np.nonzero(ink)
    if len(xs) < 200:
        return 0.0
    xs = xs - xs.mean()
    height = ink.shape[0]

    def score(angle):
        y = np.round(ys - xs * np.tan(np.radians(angle))).astype(int)
        y = y[(y >= 0) & (y < height)]
        p = np.bincount(y, minlength=height).astype(np.float64)
        return float((p * p).sum())

    best = max(np.arange(-MAX_PAGE_ANGLE, MAX_PAGE_ANGLE + 0.01, 1.5), key=score)
    best = max(np.arange(best - 1.5, best + 1.51, 0.5), key=score)
    return float(max(np.arange(best - 0.5, best + 0.51, 0.1), key=score))


def rotation_matrix(angle: float, size: tuple[int, int]) -> tuple:
    """
        Affine coefficients of Image.rotate(angle) without expansion: pixel
        (x, y) of the rotated image comes from (a x + b y + c, d x + e y + f).
    """
    w, h = size
    t = -math.radians(angle % 360.0)
    a, b, d, e = round(math.cos(t), 15), round(math.sin(t), 15), round(-math.sin(t), 15), round(math.cos(t), 15)
    cx, cy = w / 2.0, h / 2.0
    return a, b, a * -cx + b * -cy + cx, d, e, d * -cx + e * -cy + cy


def line_peaks(ink: np.ndarray, min_peak: float) -> list[tuple[int, int, int, int]]:
    """
        Lines of text from the peaks of the horizontal projection: line pitch
        from the autocorrelation, peaks at least 0.55 pitch apart and at least
        `min_peak` times the median of the strongest ones; each line spans
        between the surrounding valleys. Returns [(y0, y1, x0, x1)].
    """
    profile = ink.sum(1).astype(np.float64)
    if profile.sum() == 0:
        return []
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    smooth = np.convolve(profile, kernel / kernel.sum(), mode="same")
    centered = smooth - smooth.mean()
    autocorrelation = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    lo, hi = 6, min(len(autocorrelation) - 1, 200)
    if hi <= lo:
        return []
    ac = autocorrelation
    maxima = [t for t in range(lo + 1, hi) if ac[t] >= ac[t - 1] and ac[t] >= ac[t + 1] and ac[t] > 0.1 * ac[0]]
    if not maxima:
        return []
    pitch = maxima[0]
    sigma = max(1.0, pitch / 6)
    kernel = np.exp(-0.5 * (np.arange(-int(3 * sigma), int(3 * sigma) + 1) / sigma) ** 2)
    q = np.convolve(profile, kernel / kernel.sum(), mode="same")
    peaks = sorted((y for y in range(1, len(q) - 1) if q[y] >= q[y - 1] and q[y] > q[y + 1]), key=lambda y: -q[y])
    kept = []
    for y in peaks:
        if all(abs(y - k) >= 0.55 * pitch for k in kept):
            kept.append(y)
    if not kept:
        return []
    reference = np.median(sorted((q[y] for y in kept), reverse=True)[:max(3, len(kept) // 2)])
    kept = sorted(y for y in kept if q[y] >= min_peak * reference)
    out = []
    for i, yc in enumerate(kept):
        y0 = int(np.argmin(q[kept[i - 1]:yc]) + kept[i - 1]) if i else max(0, int(yc - 0.6 * pitch))
        y1 = int(np.argmin(q[yc:kept[i + 1]]) + yc) if i + 1 < len(kept) else min(len(q), int(yc + 0.6 * pitch))
        y0, y1 = max(y0, int(yc - 0.75 * pitch)), min(y1, int(yc + 0.75 * pitch))
        band = ink[y0:y1]
        x0, x1 = main_group(band.sum(0), 1.5 * max(1, y1 - y0))
        if x0 is None:
            continue
        # tightened vertically on the ink of the line
        rows = np.nonzero(band[:, x0:x1].sum(1) > 0)[0]
        if len(rows):
            y0, y1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
        out.append((y0, y1, x0, x1))
    return out


def main_group(column_ink: np.ndarray, max_gap: float) -> tuple[int | None, int | None]:
    """
        (start, end) of the heaviest group of inked columns, groups being
        separated by gaps of at least `max_gap` columns.
    """
    cols = np.nonzero(column_ink > 0)[0]
    if len(cols) == 0:
        return None, None
    groups = [[cols[0], cols[0] + 1]]
    for x in cols[1:]:
        if x - groups[-1][1] < max_gap:
            groups[-1][1] = x + 1
        else:
            groups.append([x, x + 1])
    x0, x1 = max(groups, key=lambda g: column_ink[g[0]:g[1]].sum())
    return int(x0), int(x1)


def line_blocks(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
        Lines of text from a threshold on the horizontal projection: blocks of
        inked rows, close blocks merged, too tall blocks split at their
        lightest row. Returns [(y0, y1, x0, x1)].
    """
    profile = ink.sum(1)
    inked = profile[profile > 0]
    reference = np.percentile(inked, 90) if len(inked) else 0
    active = profile > max(2, 0.15 * reference)
    blocks, start = [], None
    for y, on in enumerate(np.append(active, False)):
        if on and start is None:
            start = y
        elif not on and start is not None:
            blocks.append([start, y])
            start = None
    if not blocks:
        return []
    median_height = np.median([b1 - b0 for b0, b1 in blocks])
    merged = []
    for b in blocks:
        if merged and b[0] - merged[-1][1] <= 0.25 * median_height:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    median_height = np.median([b1 - b0 for b0, b1 in merged])
    split, pending = [], list(merged)
    while pending:
        y0, y1 = pending.pop(0)
        if y1 - y0 > 1.6 * median_height:
            m = int(0.3 * median_height)
            yc = y0 + m + int(np.argmin(profile[y0 + m:y1 - m]))
            pending[:0] = [[y0, yc], [yc + 1, y1]]
        else:
            split.append([y0, y1])
    median_height = np.median([b1 - b0 for b0, b1 in split])
    out = []
    for y0, y1 in split:
        if y1 - y0 < 0.4 * median_height or ink[y0:y1].sum() <= 0.5 * median_height * median_height:
            continue
        x0, x1 = main_group(ink[y0:y1].sum(0), y1 - y0)
        out.append((y0, y1, x0, x1))
    return out


def text_columns(ink: np.ndarray, corridor: float) -> list[tuple[int, int]]:
    """
        Columns of text [(xa, xb)]: an empty vertical corridor at least
        `corridor` line heights wide, with at least 25 % of the ink on each side,
        separates two columns. A column whose ink reaches the frame side of the
        mask (list cut by the photo) is dropped if another one remains.
    """
    # line_blocks finds nothing when the lines touch (tall loops of thin cursive)
    lines = line_blocks(ink) or line_peaks(ink, 0.25)
    if not lines:
        return []
    line_height = float(np.median([y1 - y0 for y0, y1, _, _ in lines]))
    column_ink = ink.sum(0).astype(float)
    xs = np.nonzero(column_ink > 0)[0]
    columns = [(int(xs[0]), int(xs[-1]) + 1)]
    total = column_ink.sum()
    empty = column_ink <= 0.002 * column_ink.max()
    best = None
    x = int(xs[0])
    while x < xs[-1]:
        if empty[x]:
            start = x
            while x < xs[-1] and empty[x]:
                x += 1
            left, right = column_ink[:start].sum(), column_ink[x:].sum()
            if (x - start >= corridor * line_height and min(left, right) >= 0.25 * total
                    and (best is None or x - start > best[1] - best[0])):
                best = (start, x)
        x += 1
    if best:
        columns = [(int(xs[0]), best[0]), (best[1], int(xs[-1]) + 1)]
    width = ink.shape[1]
    m = FRAME_BORDER + PAPER_EROSION + 2
    inside = [c for c in columns if c[0] > m and c[1] < width - m]
    return inside or columns


def drop_stray_lines(lines: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """
        Drops a first or last line (y0, y1, x0, x1) that is both isolated (more
        than 1.6 line pitches from its neighbor) and of an unusual width (less
        than half or more than twice the median): a mark, an object or the edge
        of the sheet rather than a word.
    """
    while len(lines) >= 3:
        centers = [(y0 + y1) / 2 for y0, y1, _, _ in lines]
        pitch = float(np.median(np.diff(centers)))
        width = float(np.median([x1 - x0 for _, _, x0, x1 in lines]))
        odd = [not 0.5 * width <= x1 - x0 <= 2 * width for _, _, x0, x1 in lines]
        if centers[1] - centers[0] > 1.6 * pitch and odd[0]:
            lines = lines[1:]
        elif centers[-1] - centers[-2] > 1.6 * pitch and odd[-1]:
            lines = lines[:-1]
        else:
            break
    return lines


def line_boxes(photo: Image.Image, ink: np.ndarray, method, corridor: float) -> list[list[tuple[int, int, int, int]]]:
    """
        Boxes of the lines found by `method` in each text column (see
        text_columns for `corridor`), with margins, in `photo` coordinates.
    """
    mh, mv = LIST_MARGINS
    r = LIST_REDUCTION
    out = []
    for xa, xb in text_columns(ink, corridor):
        column = np.zeros_like(ink)
        column[:, xa:xb] = ink[:, xa:xb]
        boxes = [(max(0, int((x0 - mh * (y1 - y0)) * r)), max(0, int((y0 - mv * (y1 - y0)) * r)),
                  min(photo.width, int((x1 + mh * (y1 - y0)) * r)), min(photo.height, int((y1 + mv * (y1 - y0)) * r)))
                 for y0, y1, x0, x1 in drop_stray_lines(method(column))]
        if boxes:
            out.append(boxes)
    return out


def find_lines(photo: Image.Image, ink: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    """
        Line boxes per column, columns left to right, lines top to bottom.
        Several hypotheses are tried in order, then again with closer columns;
        the first one giving a BIP39 number of lines is kept, else the first one.
    """
    first = None
    for corridor in COLUMN_CORRIDORS:
        for method in (lambda m: line_peaks(m, 0.25), line_blocks,
                       lambda m: line_peaks(m, 0.15), lambda m: line_peaks(m, 0.4)):
            columns = line_boxes(photo, ink, method, corridor)
            if first is None:
                first = columns
            if sum(len(c) for c in columns) in SEED_LENGTHS:
                return columns
    return first


def binarize(a: np.ndarray) -> np.ndarray:
    """
        Ink mask of a line crop: darker than an adaptive share of the local
        mean; isolated one-pixel strokes removed.
    """
    v = a[::2, ::2].ravel()
    k2, km = int(0.02 * (len(v) - 1)), (len(v) - 1) // 2
    q = np.partition(v, (k2, km))
    m = a < ink_threshold(float(q[k2] / max(float(q[km]), 1.0))) * local_mean(a, 15)
    vertical, horizontal = np.zeros_like(m), np.zeros_like(m)
    vertical[1:-1] = m[1:-1] & (m[:-2] | m[2:])
    horizontal[:, 1:-1] = m[:, 1:-1] & (m[:, :-2] | m[:, 2:])
    return vertical & horizontal


def erase_ruled_lines(a: np.ndarray) -> np.ndarray:
    """
        Rows dark along at least 85 % of the width (ruled or printed lines)
        replaced by the local background.
    """
    background = local_mean(a, 15)
    ruled = np.percentile(background - a, 15, axis=1) > 6
    ruled = ruled | np.roll(ruled, 1) | np.roll(ruled, -1)
    if ruled.any():
        a = a.copy()
        a[ruled] = np.maximum(a[ruled], background[ruled] * 1.02)
    return a


def clean_lines(m: np.ndarray) -> np.ndarray:
    """
        Removes long strokes, then nearly full rows and columns.
    """
    m = remove_long_strokes(m)
    m[m.mean(1) > 0.5] = False
    m[:, m.mean(0) > 0.6] = False
    return m


def segments(m: np.ndarray) -> tuple[list[list[int]], tuple[int, int]]:
    """
        Ink segments [x0, x1] of the main line of text of mask m and its rows
        (y0, y1): fragments less than 2 pixels apart merged, specks dropped,
        segments much wider than a letter split at their lightest columns.
    """
    rows = text_rows(m)
    if rows is None:
        return [], (0, 0)
    y0, y1 = rows
    h = y1 - y0
    column_ink = m[y0:y1].sum(0)
    inked = column_ink > 0
    runs, x = [], 0
    while x < len(inked):
        if inked[x]:
            start = x
            while x < len(inked) and inked[x]:
                x += 1
            runs.append([start, x])
        x += 1
    merged = []
    for s in runs:
        if merged and s[0] - merged[-1][1] < 2:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    merged = [s for s in merged if m[y0:y1, s[0]:s[1]].sum() > 0.01 * h * h]
    letter = max(np.median([s[1] - s[0] for s in merged]) if merged else h, 0.55 * h)
    out = []
    for start, end in merged:
        n = int(round((end - start) / letter))
        if end - start > 1.55 * letter and n >= 2:
            bounds = [start]
            for k in range(1, n):
                c = start + (end - start) * k // n
                window = slice(max(start + 2, c - int(0.25 * letter)), min(end - 2, c + int(0.25 * letter)))
                cs = column_ink[window]
                bounds.append(window.start + int(cs.argmin()) if len(cs) else c)
            bounds.append(end)
            out += [[bounds[k], bounds[k + 1]] for k in range(n)]
        else:
            out.append([start, end])
    return out, (y0, y1)


def line_mask(photo: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    """
        Ink mask of the crop `box` of a line, ruled lines and long strokes removed.
    """
    return clean_lines(binarize(erase_ruled_lines(np.asarray(photo.crop(box), np.float32))))


def widen_line(photo: Image.Image, box: tuple[int, int, int, int]) -> tuple:
    """
        (box, mask): the line box is widened by steps of WIDEN_STEP text heights,
        up to MAX_WIDENING heights, while its line of text has ink within one
        step beyond its left or right side (a word cut short by a faint ink
        mask, a letter set apart from the rest of the word, or a line found on
        part of its word); `mask` is the ink mask of the returned box.
    """
    widened = 0.0
    while True:
        d = max(2, int(WIDEN_STEP * (box[3] - box[1]) / (1 + 2 * LIST_MARGINS[1])))
        probe = (max(0, box[0] - d), box[1], min(photo.width, box[2] + d), box[3])
        outer = line_mask(photo, probe)
        a, b = box[0] - probe[0], box[2] - probe[0]
        segs, _ = segments(outer)
        left = box[0] > 0 and any(x0 < a + 2 for x0, _ in segs)
        right = box[2] < photo.width and any(x1 > b - 2 for _, x1 in segs)
        if not (left or right) or widened >= MAX_WIDENING:
            return box, outer[:, a:b]
        box = (probe[0] if left else box[0], box[1], probe[2] if right else box[2], box[3])
        widened += WIDEN_STEP


def unread_text(ink: np.ndarray, columns: list[list[tuple[int, int, int, int]]]) -> bool:
    """
        True if a line of text is left out of the line boxes (`ink` coordinates
        times LIST_REDUCTION): ink lines at least UNREAD_LINE[0] line heights
        wide holding at least UNREAD_LINE[1] of the ink of a median line, as
        when a word is missed or two columns are read as one.
    """
    r = LIST_REDUCTION
    rest = ink.copy()
    heights, inks = [], []
    for x0, y0, x1, y1 in (b for column in columns for b in column):
        x0, y0, x1, y1 = x0 // r, y0 // r, -(-x1 // r), -(-y1 // r)
        heights.append((y1 - y0) / (1 + 2 * LIST_MARGINS[1]))
        inks.append(int(ink[y0:y1, x0:x1].sum()))
        rest[y0:y1, x0:x1] = False
    if not heights or not rest.any():
        return False
    width, share = UNREAD_LINE
    h, line_ink = float(np.median(heights)), float(np.median(inks))
    return any(x1 - x0 >= width * h and rest[y0:y1, x0:x1].sum() >= share * line_ink
               for y0, y1, x0, x1 in line_blocks(rest) if x0 is not None)


def word_starts(boxes: list, masks: list[np.ndarray]) -> list[int | None]:
    """
        Numbered list: in each line, the widest gap among the first four
        (>= 0.25 line height, >= 1.5 times the next three, within 2.2 heights
        of the start of the ink) separates the number from the word. The list
        is numbered if at least 2/3 of the lines have one and their word starts
        are aligned (within 0.6 height of the median). Returns, per line, the
        abscissa where to cut, or None: a line whose cut is not aligned is left
        whole, since the decoder accepts a number before the word.
    """
    found = []
    for box, m in zip(boxes, masks):
        segs, (y0, y1) = segments(m)
        h = max(y1 - y0, 1)
        gaps = [segs[j + 1][0] - segs[j][1] for j in range(len(segs) - 1)]
        best = None
        for j in range(min(4, len(gaps))):
            if segs[j][1] - segs[0][0] > 2.2 * h:
                break
            following = gaps[j + 1:j + 4] or [0]
            if gaps[j] >= 0.25 * h and gaps[j] >= 1.5 * max(following) and (best is None or gaps[j] > best[1]):
                best = (box[0] + segs[j + 1][0], gaps[j], h)
        found.append(best)
    numbered = [c for c in found if c]
    if len(numbered) < 2 * len(boxes) / 3:
        return [None] * len(boxes)
    xm = float(np.median([c[0] for c in numbered]))
    hm = float(np.median([c[2] for c in numbered]))
    if len([c for c in numbered if abs(c[0] - xm) < 0.6 * hm]) < 2 * len(boxes) / 3:
        return [None] * len(boxes)
    return [int(c[0] - 0.15 * hm) if c and abs(c[0] - xm) < 0.6 * hm else None for c in found]


def reading_orders(columns: int, count: int) -> list[list[int]]:
    """
        Possible reading orders of the words, indexed column by column: column
        by column, and for two columns of equal length also row by row.
    """
    orders = [list(range(count))]
    if columns == 2 and count % 2 == 0:
        rows = count // 2
        orders.append([i for r in range(rows) for i in (r, rows + r)])
    return orders


def list_layout(photo: Image.Image, half: Image.Image, step=no_step) -> WordLayout:
    """
        Free list: page straightened, lines and numbers found on `half`; crops
        cut from `photo` with the same rotation.
    """
    ink = ink_mask(half)
    step(INK_MASK_COST)
    at_edge = ink is not None and ink_at_edge(ink)
    angle = 0.0
    page = half
    if ink is not None:
        angle = page_angle(ink)
        step(PAGE_ANGLE_COST)
        if abs(angle) >= MIN_PAGE_ANGLE:
            page = half.rotate(angle, resample=Image.BILINEAR, fillcolor=int(np.median(np.asarray(half.reduce(8)))))
            ink = np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).rotate(angle, resample=Image.NEAREST,
                                                                               fillcolor=0)) > 127
        else:
            angle = 0.0
        step(ROTATION_COST)
    columns = find_lines(page, ink) if ink is not None else []
    step(LINES_COST)
    count = sum(len(c) for c in columns)
    if count not in SEED_LENGTHS:
        raise WordsNotFound(f"{count} lines found", at_edge=at_edge, line_count=count)
    boxes, widened = [], []
    for column in columns:  # numbers removed column by column (each column has its own alignment)
        lines = []
        for b in column:
            lines.append(widen_line(page, b))
            step(NUMBER_COST)
        widened.append([b for b, _ in lines])
        boxes += [(x, b[1], b[2], b[3]) if x is not None and b[0] < x < b[2] else b
                  for (b, _), x in zip(lines, word_starts([b for b, _ in lines], [m for _, m in lines]))]
    complete = not unread_text(ink, widened)

    f = int(round(photo.width / half.width))
    matrix = rotation_matrix(angle, half.size) if angle else None
    fill = int(np.median(np.asarray(photo.reduce(16)))) if angle else 255

    def crop_coefficients(k):
        """ (x0, y0, x1, y1) of crop k on the straightened photo, affine coefficients to the photo """
        x0, y0, x1, y1 = (int(round(v * f)) for v in boxes[k])
        a, b, c, d, e, g = matrix or (1, 0, 0, 0, 1, 0)
        return (x0, y0, x1, y1), (a, b, a * x0 + b * y0 + f * c, d, e, d * x0 + e * y0 + f * g)

    def crop(k, variant):
        if variant:
            return None
        (x0, y0, x1, y1), coefficients = crop_coefficients(k)
        if matrix is None:
            return photo.crop((x0, y0, x1, y1))
        return photo.transform((x1 - x0, y1 - y0), Image.AFFINE, coefficients, Image.BILINEAR, fillcolor=fill)

    def outline(k):
        (x0, y0, x1, y1), (a, b, c, d, e, g) = crop_coefficients(k)
        return [(a * u + b * v + c, d * u + e * v + g) for u, v in ((0, 0), (x1 - x0, 0), (x1 - x0, y1 - y0), (0, y1 - y0))]

    return WordLayout("list", count, reading_orders(len(columns), count), 1, crop, outline, complete=complete)
