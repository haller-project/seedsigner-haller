"""
    Word detector of Haller-1: a small fully convolutional network (C library,
    native/handwriting.c) that finds the handwritten words, the line numbers
    and the seed card on the photo reduced to 288 x 384 pixels. It is used when
    the search by rules (page.py) finds no valid seed: badly framed cards
    (cut, turned, far, in perspective) and lists whose lines the rules miss.

    Output maps, at half the input resolution (144 x 192):
      0 word kernel: central band of each word box
      1 number kernel: the same for line numbers
      2 card: light inner area of an official seed card
      3-6 card corners: top left, top right, bottom right, bottom left, in the
          reading orientation of the card
      7 height of the word or number box (output pixels / HEIGHT_SCALE)
    Channels 0-6 are logits, channel 7 a value.

    The card is cut along its printed lines by page.card_layout, from the
    detector corners (or those of page.find_card when both designate the same
    card); a list is cut around the detected word boxes.
"""
import ctypes
import itertools
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from . import native
from . import page
from .page import WordLayout, WordsNotFound

DETECTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "handwriting",
                             "haller-1-detector.npz")

WORD, NUMBER, CARD, CORNERS, HEIGHT = 0, 1, 2, (3, 4, 5, 6), 7
HEIGHT_SCALE = 8.0
MIN_KERNEL_PIXELS = 2         # smaller word or number kernels are noise
CORNER_THRESHOLD = 0.2        # probability of a card corner peak
MAX_CORNER_PEAKS = 4          # peaks kept per corner channel
MIN_CARD_SHARE = 0.03         # share of the output in the card area above which a card is looked for
MIN_CARD_SCORE = 0.6          # see card_score
FRAME_EDGE = 2                # output pixels along the frame where corner peaks are ignored (the frame is no card)
REFINE_TOLERANCE = 1.5        # line spacings between the detector corners and those of page.find_card

# Lists, in word box heights
LIST_CROP_MARGINS = (0.6, 0.55, 0.45)  # crop around a word box: horizontal, top, bottom
SAME_LINE = 0.55              # two boxes closer than this across the text direction are on the same line
COLUMN_GAP = 1.5              # empty width between two columns
EDGE_LINES = 0.6              # first or last line closer to the frame than this many line spacings: the list may go
                              # on beyond the photo, so that no word is certain
MIN_COLUMN_LINES = 3          # a column of fewer lines touching the frame is text of another sheet

# A reading that leaves more than this many detected words out of its crops
# (another card of a sheet, the cut column of a 24-word card, a word written
# above its line) is incomplete: no word is certain
MAX_STRAY_WORDS = 1

# Expected duration on a Pi Zero, in seconds (see page.SEARCH_COST)
DETECTOR_COST = 2.2

_detector = []



def load_detector():
    """
        (library, network structure, work buffer), loaded once.
    """
    if not _detector:
        lib = native.load()
        with np.load(DETECTOR_PATH) as data:
            net = native.detector({k: np.ascontiguousarray(data[k]) for k in data.files})
        work = np.zeros(-(-lib.hw_detector_work_size(ctypes.byref(net)) // 4), np.int32)
        _detector.append((lib, net, work))
    return _detector[0]



def network_maps(image: np.ndarray) -> np.ndarray:
    """
        Output maps (8, 192, 144) float32 of a grayscale image (384, 288) uint8.
    """
    lib, net, work = load_detector()
    width, height = native.DETECTOR_SIZE
    image = np.ascontiguousarray(image, np.uint8)
    assert image.shape == (height, width)
    maps = np.empty((native.DETECTOR_OUTPUTS, height // 2, width // 2), np.float32)
    lib.hw_detector_maps(ctypes.byref(net), image.ctypes.data, work.ctypes.data, maps.ctypes.data)
    return maps



def reduce_photo(photo: Image.Image) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
        Photo fitted into the input of the network, aspect kept, padded with its
        median gray: (image, scale from the photo to the input, offset of the
        photo in the input).
    """
    width, height = native.DETECTOR_SIZE
    s = min(width / photo.width, height / photo.height)
    w, h = max(1, round(photo.width * s)), max(1, round(photo.height * s))
    small = photo.resize((w, h), Image.BOX)
    canvas = Image.new("L", (width, height), int(np.median(np.asarray(small))))
    offset = ((width - w) // 2, (height - h) // 2)
    canvas.paste(small, offset)
    return np.asarray(canvas), s, offset



# ---------- decoding of the maps ----------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))



def components(mask: np.ndarray) -> list[np.ndarray]:
    """
        (y, x) pixels of the 8-connected components of a boolean mask, in
        raster order of their first pixel.
    """
    m = np.ascontiguousarray(mask, np.uint8)
    labels = np.empty(m.shape, np.int32)
    work = np.empty(m.size, np.int32)
    n = native.load().hw_components(m.ctypes.data, m.shape[0], m.shape[1], labels.ctypes.data, work.ctypes.data)
    order = np.argsort(labels.ravel(), kind="stable")
    ends = np.cumsum(np.bincount(labels.ravel(), minlength=n + 1))
    ys, xs = np.unravel_index(order, mask.shape)
    return [np.stack([ys[ends[k - 1]:ends[k]], xs[ends[k - 1]:ends[k]]], 1) for k in range(1, n + 1)]



def principal_angle(pixels: np.ndarray) -> tuple[float, float]:
    """
        Angle of the main axis of a set of pixels, and the ratio of its two
        variances (elongation).
    """
    y, x = pixels[:, 0].astype(float), pixels[:, 1].astype(float)
    cxx, cyy, cxy = x.var(), y.var(), ((x - x.mean()) * (y - y.mean())).mean()
    angle = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    trace, det = cxx + cyy, cxx * cyy - cxy * cxy
    d = math.sqrt(max(0.0, trace * trace / 4 - det))
    return angle, (trace / 2 + d) / max(trace / 2 - d, 0.05)



def text_angle(kernels: list[np.ndarray]) -> float:
    """
        Direction of the text: mean axis of the elongated word kernels,
        weighted by their size.
    """
    angles, weights = [], []
    for pixels in kernels:
        if len(pixels) >= 6:
            a, elongation = principal_angle(pixels)
            if elongation > 6:
                angles.append(a)
                weights.append(len(pixels))
    if not angles:
        return 0.0
    c = np.average(np.cos(2 * np.array(angles)), weights=weights)
    s = np.average(np.sin(2 * np.array(angles)), weights=weights)
    return 0.5 * math.atan2(s, c)



class Box:
    """
        Word or number box in output pixels, in the text direction (ux, uy):
        extent [a0, a1] along it, center c across it, height h.
    """
    def __init__(self, pixels: np.ndarray, probability: np.ndarray, height: np.ndarray, angle: float):
        y, x = pixels[:, 0].astype(float) + 0.5, pixels[:, 1].astype(float) + 0.5
        self.ux, self.uy = math.cos(angle), math.sin(angle)
        along, across = x * self.ux + y * self.uy, -x * self.uy + y * self.ux
        self.h = max(1.0, float(np.median(height[pixels[:, 0], pixels[:, 1]])) * HEIGHT_SCALE)
        self.a0, self.a1 = along.min() - 0.5 - 0.25 * self.h, along.max() + 0.5 + 0.25 * self.h
        self.c = float(across.mean())
        self.score = float(probability[pixels[:, 0], pixels[:, 1]].mean())
        self.pixels = len(pixels)


    @property
    def middle(self) -> float:
        return (self.a0 + self.a1) / 2


    def point(self, a: float, b: float) -> tuple[float, float]:
        """ Output pixel at a along the text and b across it. """
        return a * self.ux - b * self.uy, a * self.uy + b * self.ux


    def quad(self, margins=(0.0, 0.5, 0.5)) -> list[tuple[float, float]]:
        """
            Corners, clockwise from the top left, with margins in box heights
            (horizontal, top, bottom).
        """
        mx, mt, mb = (m * self.h for m in margins)
        a0, a1, b0, b1 = self.a0 - mx, self.a1 + mx, self.c - mt, self.c + mb
        return [self.point(a, b) for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]


    def merge(self, other):
        self.a0, self.a1 = min(self.a0, other.a0), max(self.a1, other.a1)
        n = self.pixels + other.pixels
        self.c = (self.c * self.pixels + other.c * other.pixels) / n
        self.h = max(self.h, other.h)
        self.pixels = n



def corner_peaks(maps: np.ndarray) -> list[list[tuple[float, float, float]]]:
    """
        Up to MAX_CORNER_PEAKS local maxima of each corner channel above
        CORNER_THRESHOLD, strongest first: (x, y, probability) in output pixels,
        the position refined by a weighted mean over 5 x 5 pixels.
    """
    peaks = []
    for channel in CORNERS:
        p = sigmoid(maps[channel])
        h, w = p.shape
        padded = np.pad(p, 1)
        is_max = np.ones_like(p, bool)
        for dy in range(3):
            for dx in range(3):
                if dy != 1 or dx != 1:
                    is_max &= p >= padded[dy:dy + h, dx:dx + w]
        is_max[:FRAME_EDGE] = is_max[-FRAME_EDGE:] = False
        is_max[:, :FRAME_EDGE] = is_max[:, -FRAME_EDGE:] = False
        ys, xs = np.nonzero(is_max & (p >= CORNER_THRESHOLD))
        found = []
        for k in np.argsort(-p[ys, xs], kind="stable")[:MAX_CORNER_PEAKS]:
            y, x = ys[k], xs[k]
            y0, x0 = max(0, y - 2), max(0, x - 2)
            q = p[y0:y + 3, x0:x + 3]
            yy, xx = np.mgrid[y0:y0 + q.shape[0], x0:x0 + q.shape[1]]
            found.append((float((xx * q).sum() / q.sum()) + 0.5, float((yy * q).sum() / q.sum()) + 0.5, float(p[y, x])))
        peaks.append(found)
    return peaks



class Detection:
    """
        Decoded output maps, in output pixels: word and number boxes, card
        corner peaks, share of the output on the card area. `to_photo(points)`
        maps output pixels to photo pixels.
    """
    def __init__(self, maps: np.ndarray, to_photo):
        self.maps = maps
        self.to_photo = to_photo
        words, numbers = sigmoid(maps[WORD]), sigmoid(maps[NUMBER])
        word_kernels = [p for p in components(maps[WORD] > 0) if len(p) >= MIN_KERNEL_PIXELS]
        number_kernels = [p for p in components(maps[NUMBER] > 0) if len(p) >= MIN_KERNEL_PIXELS]
        angle = text_angle(word_kernels)
        self.words = [Box(p, words, maps[HEIGHT], angle) for p in word_kernels]
        self.numbers = [Box(p, numbers, maps[HEIGHT], angle) for p in number_kernels]
        self.peaks = corner_peaks(maps)
        self.card_share = float((maps[CARD] > 0).mean())



def detect(photo: Image.Image) -> Detection:
    """
        Detection on an upright grayscale photo.
    """
    image, s, (ox, oy) = reduce_photo(photo)
    return Detection(network_maps(image), lambda points: [((2 * u - ox) / s, (2 * v - oy) / s) for u, v in points])



# ---------- card ----------

def complete_quad(corners: list, kind: str) -> list | None:
    """
        Four corners from the visible ones (at least two, None for the others):
        a parallelogram from three, the card geometry from two.
    """
    known = [k for k in range(4) if corners[k] is not None]
    if len(known) == 4:
        return [tuple(c) for c in corners]
    if len(known) < 2:
        return None
    if len(known) == 3:
        k = [i for i in range(4) if corners[i] is None][0]
        a, b, c = corners[(k + 1) % 4], corners[(k + 2) % 4], corners[(k + 3) % 4]
        quad = list(corners)
        quad[k] = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
        return quad
    # similarity mapping two corners of the template onto the found ones, in complex numbers
    template = page.CARD_CORNERS[kind]
    i, j = known[:2]
    ti, tj = complex(*template[i]), complex(*template[j])
    pi, pj = complex(*corners[i]), complex(*corners[j])
    s = (pi - pj) / (ti - tj)
    d = pi - s * ti
    return [((s * complex(*t) + d).real, (s * complex(*t) + d).imag) for t in template]



def inside(quad: list, point: tuple[float, float]) -> bool:
    """
        True if the point is inside the convex quad.
    """
    x, y = point
    sides = [(x1 - x0) * (y - y0) - (y1 - y0) * (x - x0) for (x0, y0), (x1, y1) in zip(quad, quad[1:] + quad[:1])]
    return all(c > 0 for c in sides) or all(c < 0 for c in sides)



def word_starts_in_template(det: Detection, quad: list, kind: str) -> list[tuple[float, float]]:
    """
        Start of each detected word (middle of the left side of its box) in
        the coordinates of the card template of `kind` placed on `quad`.
    """
    coefficients = page.homography(quad, page.CARD_CORNERS[kind])  # template -> output pixels
    from_output = np.linalg.inv(np.append(coefficients, 1).reshape(3, 3))
    starts = []
    for b in det.words:
        q = b.quad()
        p = from_output @ np.array([(q[0][0] + q[3][0]) / 2, (q[0][1] + q[3][1]) / 2, 1.0])
        starts.append((p[0] / p[2], p[1] / p[2]))
    return starts



def card_kind(det: Detection, quad: list) -> str:
    """
        card24 if at least three words start in the right column of the
        24-word card placed on `quad`.
    """
    right = sum(560 < x < 900 and 300 < y < 1300 for x, y in word_starts_in_template(det, quad, "card24"))
    return "card24" if right >= 3 else "card12"



def card_score(det: Detection, quad: list, card_area: np.ndarray) -> float:
    """
        Agreement of a candidate quad with the card area and the words: share
        of the quad on the card area, plus 0.05 per word (up to 24) whose
        center is inside; -1 if its shape is not that of a card.
    """
    q = np.array(quad)
    side = lambda i, j: float(np.hypot(*(q[i] - q[j])))  # noqa: E731
    width, height = (side(0, 1) + side(3, 2)) / 2, (side(0, 3) + side(1, 2)) / 2
    if height < 12 or width < 8 or not 0.5 < width / height < 1.1:
        return -1.0
    mask = Image.new("1", card_area.shape[::-1], 0)
    ImageDraw.Draw(mask).polygon([tuple(p) for p in quad], fill=1)
    mask = np.asarray(mask)
    area = mask.sum()
    if area < 60:
        return -1.0
    words_in = sum(inside(quad, b.point(b.middle, b.c)) for b in det.words)
    return (mask & card_area).sum() / area + 0.05 * min(words_in, 24)



def find_card(det: Detection) -> tuple[list, str] | None:
    """
        (corners in output pixels, kind) of the card found by the detector, or
        None. Candidates: one peak or none per corner channel, at least two
        peaks, completed by the card geometry; the best by card_score plus half
        the probability of its peaks.
    """
    if det.card_share < MIN_CARD_SHARE or sum(bool(p) for p in det.peaks) < 2:
        return None
    card_area = det.maps[CARD] > 0
    best, best_score = None, MIN_CARD_SCORE
    for combination in itertools.product(*([None] + p for p in det.peaks)):
        present = [c for c in combination if c is not None]
        if len(present) < 2:
            continue
        corners = [c[:2] if c else None for c in combination]
        quad = complete_quad(corners, "card12")
        score = card_score(det, quad, card_area)
        if score < 0:
            continue
        score += 0.5 * sum(c[2] for c in present)
        if score > best_score:
            best, best_score = corners, score
    if best is None:
        return None
    kind = card_kind(det, complete_quad(best, "card12"))
    return complete_quad(best, kind), kind



def touches_frame(quad: list, size: tuple[int, int], margin: float = 2) -> bool:
    q = np.array(quad)
    return bool(q[:, 0].min() < margin or q[:, 1].min() < margin or q[:, 0].max() > size[0] - margin
                or q[:, 1].max() > size[1] - margin)



def card_words(photo: Image.Image, det: Detection, card: tuple[list, str], step) -> WordLayout:
    """
        Card layout from the detector corners, or from those of page.find_card
        when both designate the same card (same kind, every corner within
        REFINE_TOLERANCE line spacings), these being more precise. No word is
        certain if a word crop goes out of the photo.
    """
    quad, kind = det.to_photo(card[0]), card[1]
    threshold = page.LINE_DARKNESS
    half = photo.reduce(2)
    q, found, k, t = page.find_card(half, step)
    if q is not None and found > page.MIN_CARD_LINES and k == kind:
        q = [(x * photo.width / half.width, y * photo.height / half.height) for x, y in q]
        a, b = np.array(quad), np.array(q)
        spacing = np.linalg.norm(a[3] - a[0]) * page.LINE_SPACING \
            / (page.CARD_CORNERS["card12"][3][1] - page.CARD_CORNERS["card12"][0][1])
        if np.max(np.linalg.norm(a - b, axis=1)) <= REFINE_TOLERANCE * spacing:
            quad, threshold = q, t
    layout = page.card_layout(photo, quad, kind, step, threshold)
    layout.complete = bool(page.quad_area(quad) >= page.MIN_SURE_CARD_AREA * photo.width * photo.height) \
        and all(not touches_frame(layout.outline(k), photo.size, 0) for k in range(layout.count))
    return layout



# ---------- list ----------

def stacked(a: Box, b: Box) -> bool:
    """
        Two boxes overlapping along the text by more than a third of the
        shorter one: words of two lines, not two pieces of a word.
    """
    return min(a.a1, b.a1) - max(a.a0, b.a0) > 0.33 * min(a.a1 - a.a0, b.a1 - b.a0)



def group_lines(boxes: list[Box]) -> list[list[Box]]:
    """
        Columns of lines, lines top to bottom: boxes merged when on the same
        line and not stacked; columns separated by an empty corridor.
    """
    if not boxes:
        return []
    h = float(np.median([b.h for b in boxes]))
    spans = sorted((b.a0, b.a1) for b in boxes)
    columns, (s0, s1) = [], spans[0]
    for a0, a1 in spans[1:]:
        if a0 - s1 > COLUMN_GAP * h:
            columns.append((s0, s1))
            s0, s1 = a0, a1
        else:
            s1 = max(s1, a1)
    columns.append((s0, s1))
    out = []
    for c0, c1 in columns:
        lines = []
        for b in sorted((b for b in boxes if c0 <= b.middle <= c1), key=lambda b: b.c):
            if lines and abs(b.c - lines[-1].c) < SAME_LINE * max(h, b.h) and not stacked(lines[-1], b):
                lines[-1].merge(b)
            else:
                lines.append(b)
        out.append(lines)
    return out



def upside_down(lines: list[Box], numbers: list[Box]) -> bool:
    """
        True if most numbers next to the lines lie after their words along the
        text: the page is turned over.
    """
    before = after = 0
    for n in numbers:
        x = n.middle
        near = [(max(b.a0 - x, x - b.a1), x < b.a0 + 0.5 * b.h) for b in lines
                if abs(b.c - n.c) < SAME_LINE * max(b.h, n.h)]
        if not near:
            continue
        gap, is_before = min(near)
        if gap <= 3 * n.h:
            before += is_before
            after += not is_before
    return after > before and after >= max(3, len(lines) // 3)



def column_at_edge(lines: list[Box], size: tuple[int, int]) -> bool:
    """
        The first or last line of a column is closer than EDGE_LINES line
        spacings to the frame (output pixels of `size`), across the text.
    """
    if len(lines) < 2:
        return False
    spacing = float(np.median(np.diff([b.c for b in lines])))
    w, h = size
    for b, sign in ((lines[0], -1), (lines[-1], 1)):
        x, y = b.point(b.middle, b.c + sign * EDGE_LINES * spacing)
        if not (0 <= x <= w and 0 <= y <= h):
            return True
    return False



def rectified_crop(photo: Image.Image, quad: list, fill: int) -> Image.Image:
    """
        Quad (clockwise from the top left) cut out and straightened.
    """
    q = np.array(quad)
    w = max(4, int(round((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2)))
    h = max(4, int(round((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)))
    coefficients = page.homography([tuple(p) for p in q], [(0, 0), (w, 0), (w, h), (0, h)])
    return photo.transform((w, h), Image.PERSPECTIVE, coefficients, Image.BILINEAR, fillcolor=fill)



def list_words(photo: Image.Image, det: Detection) -> WordLayout:
    """
        List layout from the detected word boxes. Columns of fewer than
        MIN_COLUMN_LINES lines mostly on the frame are dropped (text of another
        sheet) when the rest has a valid number of words. No word is certain if
        a line touches the frame or the list may go on beyond it.
    """
    columns = group_lines(det.words)
    count = sum(len(c) for c in columns)
    if count not in page.SEED_LENGTHS and len(columns) > 1:
        kept = [c for c in columns if len(c) >= MIN_COLUMN_LINES
                and 2 * sum(touches_frame(det.to_photo(b.quad()), photo.size, 8) for b in c) < len(c)]
        if sum(len(c) for c in kept) in page.SEED_LENGTHS:
            columns = kept
            count = sum(len(c) for c in kept)
    output_size = det.maps.shape[2], det.maps.shape[1]
    at_edge = any(column_at_edge(c, output_size) for c in columns) \
        or any(touches_frame(det.to_photo(b.quad()), photo.size) for c in columns for b in c)
    if count not in page.SEED_LENGTHS:
        raise WordsNotFound(f"{count} lines found", at_edge=at_edge, line_count=count)
    lines = [b for c in columns for b in c]
    quads = [det.to_photo(b.quad(LIST_CROP_MARGINS)) for b in lines]
    if upside_down(lines, det.numbers):
        quads = [q[2:] + q[:2] for q in reversed(quads)]
        columns.reverse()
    fill = int(np.median(np.asarray(photo.reduce(16))))

    def crop(k, variant):
        return None if variant else rectified_crop(photo, quads[k], fill)

    return WordLayout("list", count, page.reading_orders(len(columns), count), 1, crop, lambda k: quads[k],
                      complete=not at_edge)



def stray_words(det: Detection, layout: WordLayout) -> int:
    """
        Number of detected words whose center lies outside every crop of
        `layout` (photo pixels).
    """
    outlines = [layout.outline(k) for k in range(layout.count)]
    centers = det.to_photo([b.point(b.middle, b.c) for b in det.words])
    return sum(not any(inside(q, c) for q in outlines) for c in centers)



def find_words(image: Image.Image, step=page.no_step) -> WordLayout:
    """
        Words of the card or list on `image` (upright grayscale photo, full
        resolution) found by the detector; `layout.stray`: detected words left
        out of its crops (see MAX_STRAY_WORDS). Raises WordsNotFound if no card
        is found and no list with a valid number of lines either.
    """
    det = detect(image)
    step(DETECTOR_COST)
    card = find_card(det)
    layout = card_words(image, det, card, step) if card else list_words(image, det)
    layout.stray = stray_words(det, layout)
    if layout.stray > MAX_STRAY_WORDS:
        layout.complete = False
    return layout
