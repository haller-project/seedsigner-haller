"""
    Word detector of Haller-1: a small fully convolutional network (C library,
    native/handwriting.c) that finds the handwritten words, the line numbers
    and the cards on the photo reduced to 288 x 384 pixels. It is the first
    search of the words, for every kind of backup: the official SeedSigner
    card, other printed cards and free lists on any paper.

    Output maps, at half the input resolution (144 x 192):
      0 word kernel: central band of each word box
      1 number kernel: the same for line numbers, printed or written
      2 card: inner area of a printed card
      3-6 card corners (not used by the reading)
      7 height of the word or number box (output pixels / HEIGHT_SCALE)
      8 card kernel: card area shrunk to KERNEL_SHRINK around its center, which
        separates cards lying side by side
      9 link: band from each number to the start of its word
      10 cut or hidden word (not used by the reading)
    Channels 0-6 and 8-10 are logits, channel 7 a value.

    Reading (find_words):
      1. documents: each card kernel is a card; a word belongs to the card
         holding its center, other words are free (lists); the document read is
         the one with the most words, and another one of MIN_OTHER_WORDS words
         or more makes every word uncertain;
      2. lines and columns (group_lines), or a tight grid of rows
         (grid_columns); on a document with MIN_ANCHOR_NUMBERS numbers or more,
         one slot per number (anchored_slots);
      3. orientation: numbers after their words, or the words reading better
         turned over, mean the page is upside down;
      4. order: column by column or row by row; when both are possible, the
         numbers are read until one order leads by ORDER_MARGIN, otherwise both
         are given to the analysis, which makes every word uncertain if both
         give a plausible seed;
      5. completeness: no word is certain if a word touches the frame, a
         column may go on beyond it, or a card is cut after its text (or cut
         before it without the first number read being 1).
"""
import ctypes
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from . import native
from . import page
from .page import WordLayout, WordsNotFound
from .preprocess import normalize_word

DETECTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "handwriting",
                             "haller-1-detector.npz")

WORD, NUMBER, CARD, HEIGHT, CARD_KERNEL, LINK = 0, 1, 2, 7, 8, 9
HEIGHT_SCALE = 8.0
MIN_KERNEL_PIXELS = 2         # smaller word or number kernels are noise

# Documents
KERNEL_SHRINK = 0.88          # card kernel = card area shrunk to this share around its center
MIN_CARD_KERNEL = 150         # output pixels of a card kernel component (smaller ones: stray card
                              # kernels on lists; chosen on simulated pages kept apart from the tests)
MIN_GROUP_WORDS = 3           # a document with fewer words is a blank card or noise
MIN_OTHER_WORDS = 6           # words of another document making every word uncertain

# Lines and columns, in word box heights
SAME_LINE = 0.55              # two boxes closer than this across the text direction are on the same line
COLUMN_GAP = 1.5              # empty width between two columns
GRID_WORD_GAP = 0.9           # gap along a row above which two pieces are two words of a grid
MIN_GRID_ROWS_SHARE = 0.75    # share of the rows of a grid holding its full number of words
PIECE_GAP = 0.0               # gap between boxes (with their margins) above which two pieces may be two words
                              # of aligned columns: a blank of half a height
ALIGNMENT = 1.0               # largest step between the sorted starts of the words of a column (a slanted page
                              # makes them drift)
MIN_WORD_LENGTH = 2.0         # median length of the words of a column, numbers written against them included
EDGE_LINES = 0.6              # first or last line closer to the frame than this many line spacings: the list may go
                              # on beyond the photo, so that no word is certain
MIN_COLUMN_LINES = 3          # a column of fewer lines mostly on the frame is text of another sheet

# Numbers
MIN_ANCHOR_NUMBERS = 6        # numbers of a document above which its slots are anchored on them
WRITTEN = 0.5                 # word probability in an empty slot above which it holds a missed word
SLOT_LENGTH = 8.0             # length of a slot with no word found, in number heights
NUMBER_WORD_GAP = 0.15        # start of a slot word after the end of its number, in number heights
LINK_THRESHOLD = 0.3          # mean link probability along a number-word segment
NUMBER_GAP = 3.0              # word heights between a number and its word, without link
NUMBER_CROP_MARGINS = (0.4, 0.9, 0.9)  # crop of a number around the center line of its box, in box heights
MAX_NUMBER = 33
NUMBER_TRUST = 0.7            # a number reading is right with this probability, otherwise any value
MIN_NUMBERS_READ = 3          # numbers read before an order may be decided by them
ORDER_MARGIN = math.log(1e4)  # log-likelihood lead of the best order over the others

# Orientation: the tallest ORIENTATION_WORDS words are read in both directions; the page is upside down if the
# turned readings lead by ORIENTATION_MARGIN (summed log-likelihoods per unit of width)
ORIENTATION_WORDS = 3
ORIENTATION_MARGIN = 1.0

# Check of the order of a valid seed (see number_contradicts)
NUMBER_CHECK = 0.9
NUMBER_CHECK_OTHER = 0.05

# Word crops around the box, in box heights (horizontal, top, bottom): first reading, taller band, wider band
CROP_VARIANTS = ((0.6, 0.85, 0.75), (0.6, 1.1, 1.0), (1.2, 0.8, 0.7))

# Expected duration on a Pi Zero, in seconds (see page.SEARCH_COST): network and decoding, reading of a number
DETECTOR_COST = 1.8
NUMBER_COST = 0.15

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


    @property
    def start(self) -> tuple[float, float]:
        """ Output pixel at the start of the box, on its center line. """
        return self.point(self.a0, self.c)


    @property
    def center(self) -> tuple[float, float]:
        """ Output pixel at the center of the box. """
        return self.point(self.middle, self.c)


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



class Detection:
    """
        Decoded output maps, in output pixels: word and number boxes, cards
        (pixels of each card kernel component), card area. `to_photo(points)`
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
        self.cards = [p for p in components(maps[CARD_KERNEL] > 0) if len(p) >= MIN_CARD_KERNEL]
        self.labels = np.zeros(maps.shape[1:], np.int32)
        for i, pixels in enumerate(self.cards, 1):
            self.labels[pixels[:, 0], pixels[:, 1]] = i
        self.card_area = maps[CARD] > 0


    def card_of(self, point: tuple[float, float]) -> int:
        """
            Index of the card holding the point (output pixels): the card whose
            kernel holds it once brought KERNEL_SHRINK closer to the card center,
            or the nearest card if it lies on the card area; -1 if none.
        """
        x, y = point
        h, w = self.labels.shape
        for i, pixels in enumerate(self.cards):
            cx, cy = pixels[:, 1].mean() + 0.5, pixels[:, 0].mean() + 0.5
            u, v = cx + KERNEL_SHRINK * (x - cx), cy + KERNEL_SHRINK * (y - cy)
            if 0 <= int(v) < h and 0 <= int(u) < w and self.labels[int(v), int(u)] == i + 1:
                return i
        if self.cards and 0 <= int(y) < h and 0 <= int(x) < w and self.card_area[int(y), int(x)]:
            return int(np.argmin([math.hypot(x - p[:, 1].mean() - 0.5, y - p[:, 0].mean() - 0.5)
                                  for p in self.cards]))
        return -1


    def cut_sides(self, card: int) -> list[tuple[int, int]]:
        """
            Outward normals (x, y) of the sides of the frame touched by the
            kernel of `card`.
        """
        p = self.cards[card]
        h, w = self.labels.shape
        sides = (((0, -1), p[:, 0].min() <= 0), ((0, 1), p[:, 0].max() >= h - 1),
                 ((-1, 0), p[:, 1].min() <= 0), ((1, 0), p[:, 1].max() >= w - 1))
        return [n for n, hit in sides if hit]


    def link_score(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """
            Mean link probability along the segment from a to b (output pixels).
        """
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])))
        xs = np.clip(np.linspace(a[0], b[0], n).astype(int), 0, self.maps.shape[2] - 1)
        ys = np.clip(np.linspace(a[1], b[1], n).astype(int), 0, self.maps.shape[1] - 1)
        return float(sigmoid(self.maps[LINK][ys, xs]).mean())



def detect(photo: Image.Image) -> Detection:
    """
        Detection on an upright grayscale photo.
    """
    image, s, (ox, oy) = reduce_photo(photo)
    return Detection(network_maps(image), lambda points: [((2 * u - ox) / s, (2 * v - oy) / s) for u, v in points])



# ---------- lines and columns ----------

def touches_frame(quad: list, size: tuple[int, int], margin: float = 2) -> bool:
    q = np.array(quad)
    return bool(q[:, 0].min() < margin or q[:, 1].min() < margin or q[:, 0].max() > size[0] - margin
                or q[:, 1].max() > size[1] - margin)



def copy_box(b: Box) -> Box:
    """
        Copy of a box (merges change boxes in place).
    """
    c = Box.__new__(Box)
    c.__dict__.update(b.__dict__)
    return c



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



def grid_columns(boxes: list[Box]) -> list[list[Box]] | None:
    """
        Columns of a grid of words written in rows, however close the columns:
        rows of boxes on the same line, split into words where the gap exceeds
        GRID_WORD_GAP heights. Every row holds the same number m >= 2 of words
        except a shorter last row, at least MIN_GRID_ROWS_SHARE of them in
        full, and the columns do not overlap along the rows. None otherwise.
    """
    if len(boxes) < 4:
        return None
    boxes = [copy_box(b) for b in boxes]
    h = float(np.median([b.h for b in boxes]))
    rows = []
    for b in sorted(boxes, key=lambda b: b.c):
        if rows and abs(b.c - np.mean([x.c for x in rows[-1]])) < SAME_LINE * max(h, b.h):
            rows[-1].append(b)
        else:
            rows.append([b])
    words = []
    for row in rows:
        row.sort(key=lambda b: b.a0)
        merged = [row[0]]
        for b in row[1:]:
            if b.a0 - merged[-1].a1 > GRID_WORD_GAP * h or stacked(merged[-1], b):
                merged.append(b)
            else:
                merged[-1].merge(b)
        words.append(merged)
    m = max(len(r) for r in words)
    if m < 2 or len(words) < 2:
        return None
    if any(len(r) != m for r in words[:-1]) or sum(len(r) == m for r in words) < MIN_GRID_ROWS_SHARE * len(words):
        return None
    starts = [np.median([r[j].a0 for r in words if j < len(r)]) for j in range(m)]
    ends = [np.median([r[j].a1 for r in words if j < len(r)]) for j in range(m)]
    if any(starts[j + 1] <= ends[j] - 0.5 * h for j in range(m - 1)):
        return None
    return [[r[j] for r in words if j < len(r)] for j in range(m)]



def aligned_columns(boxes: list[Box]) -> list[list[Box]] | None:
    """
        Columns of words written in rows with almost no space between the
        columns (a number written against the end of the word before it):
        the pieces of each row, split where the gap exceeds PIECE_GAP heights,
        start at the same places along the rows. A column starts where pieces
        of at least MIN_GRID_ROWS_SHARE of the rows start, sorted starts less
        than ALIGNMENT heights apart forming one place; every row fills the same number m >= 2 of columns except a
        shorter last row, and the words of each column are at least
        MIN_WORD_LENGTH heights long (a column of numbers apart from their
        words is not a column of words). None otherwise.
    """
    if len(boxes) < 4:
        return None
    boxes = [copy_box(b) for b in boxes]
    h = float(np.median([b.h for b in boxes]))
    rows = []
    for b in sorted(boxes, key=lambda b: b.c):
        if rows and abs(b.c - np.mean([x.c for x in rows[-1]])) < SAME_LINE * max(h, b.h):
            rows[-1].append(b)
        else:
            rows.append([b])
    if len(rows) < 2:
        return None
    pieces = []
    for row in rows:
        row.sort(key=lambda b: b.a0)
        merged = [row[0]]
        for b in row[1:]:
            if b.a0 - merged[-1].a1 > PIECE_GAP * h or stacked(merged[-1], b):
                merged.append(b)
            else:
                merged[-1].merge(b)
        pieces.append(merged)
    starts = sorted((p.a0, i) for i, row in enumerate(pieces) for p in row)
    clusters, current = [], [starts[0]]
    for a, i in starts[1:]:
        if a - current[-1][0] <= ALIGNMENT * h:
            current.append((a, i))
        else:
            clusters.append(current)
            current = [(a, i)]
    clusters.append(current)
    columns = [min(a for a, _ in c) for c in clusters if len({i for _, i in c}) >= MIN_GRID_ROWS_SHARE * len(rows)]
    if len(columns) < 2:
        return None
    grid = []
    for row in pieces:
        cells = {}
        for p in row:
            j = max([j for j, a in enumerate(columns) if a <= p.a0 + ALIGNMENT * h], default=0)
            if j in cells:
                cells[j].merge(p)
            else:
                cells[j] = p
        grid.append(cells)
    m = len(columns)
    full = [sorted(cells) == list(range(m)) for cells in grid]
    last = grid[-1]
    if not all(full[:-1]) or not (full[-1] or sorted(last) == list(range(len(last)))):
        return None
    out = [[cells[j] for cells in grid if j in cells] for j in range(m)]
    if any(np.median([b.a1 - b.a0 for b in column]) < MIN_WORD_LENGTH * h for column in out):
        return None
    return out



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



def band_max(logits: np.ndarray, box: Box) -> float:
    """
        Highest probability of a logit map on the central band of a box
        (30 % of its height on either side of its center line).
    """
    h, w = logits.shape
    im = Image.new("1", (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in box.quad((0.0, 0.3, 0.3))], fill=1)
    mask = np.asarray(im)
    return float(sigmoid(logits[mask]).max()) if mask.any() else 0.0



# ---------- numbers ----------

def numbers_before(lines: list[Box], numbers: list[Box]) -> bool:
    """
        True if at least three numbers lie just before a word along the text
        (upright page).
    """
    count = 0
    for n in numbers:
        for b in lines:
            if abs(b.c - n.c) < SAME_LINE * max(b.h, n.h) and n.middle < b.a0 + 0.5 * b.h \
                    and b.a0 - n.middle < 3 * max(b.h, n.h):
                count += 1
                break
    return count >= 3



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



def link_numbers(det: Detection, lines: list[Box], numbers: list[Box]) -> dict[int, Box]:
    """
        {line index: number}: each number linked to the word line that follows
        it on its line (or precedes it, page upside down), by the link map, or
        else to the nearest line within NUMBER_GAP heights.
    """
    pairs = []
    for j, n in enumerate(numbers):
        for i, b in enumerate(lines):
            if abs(b.c - n.c) > SAME_LINE * max(b.h, n.h) * 1.6:
                continue
            gap = max(b.a0 - n.middle, n.middle - b.a1)
            if gap < -0.3 * b.h:
                continue  # inside the word
            s = det.link_score(n.center, b.start if n.middle < b.a0 else b.center)
            near = gap <= NUMBER_GAP * max(b.h, n.h)
            if s >= LINK_THRESHOLD or near:
                pairs.append((-(s + (0.5 if near else 0)), gap, i, j))
    linked, used = {}, set()
    for _, _, i, j in sorted(pairs):
        if i not in linked and j not in used:
            linked[i] = numbers[j]
            used.add(j)
    return linked



def split_at_numbers(words: list[Box], numbers: list[Box]) -> list[Box]:
    """
        Word boxes cut where a number lies inside them on their line (tight
        columns: the number of the next column written against the word of the
        previous one); pieces shorter than half a height are dropped.
    """
    out = []
    for b in words:
        cuts = sorted((n for n in numbers if abs(n.c - b.c) < SAME_LINE * max(b.h, n.h)
                       and b.a0 + 0.5 * n.h < n.middle < b.a1 - 0.5 * n.h), key=lambda n: n.a0)
        if not cuts:
            out.append(b)
            continue
        bounds = [b.a0] + [x for n in cuts for x in (n.a0, n.a1)] + [b.a1]
        for a0, a1 in zip(bounds[::2], bounds[1::2]):
            if a1 - a0 > 0.5 * b.h:
                piece = copy_box(b)
                piece.a0, piece.a1 = a0, a1
                piece.pixels = max(1, int(round(b.pixels * (a1 - a0) / (b.a1 - b.a0))))
                out.append(piece)
    return out



def anchored_slots(det: Detection, words: list[Box], numbers: list[Box]) -> dict | None:
    """
        Slots of a numbered document: the numbers in columns (group_lines), each
        one followed by the words found on its line up to the next column
        (merged), or, if none, by a slot of SLOT_LENGTH number heights taken as
        written if the word map shows ink in it. The filled slots must come
        first in a reading order of the grid (column by column or row by row).
        {"columns", "count", "orders", "numbers", "synthetic"} over the filled
        slots, or None.
    """
    words = split_at_numbers([copy_box(b) for b in words], numbers)
    if not numbers_before(words, numbers):
        return None
    number_columns = [c for c in group_lines([copy_box(n) for n in numbers]) if len(c) >= 2]
    if not number_columns:
        return None
    number_columns.sort(key=lambda c: min(n.a0 for n in c))
    starts = [min(n.a0 for n in c) for c in number_columns]
    grid = []
    for ci, column in enumerate(number_columns):
        following = starts[ci + 1] if ci + 1 < len(number_columns) else math.inf
        slots = []
        for n in column:
            found = [b for b in words if abs(b.c - n.c) < SAME_LINE * max(b.h, n.h) and b.a1 > n.a1
                     and b.middle < following - 0.3 * n.h]
            if found:
                box = copy_box(found[0])
                for b in found[1:]:
                    box.merge(copy_box(b))
                box.ink = box.a1 - box.a0  # length of the writing found, before the slot is widened
                box.a0 = min(box.a0, n.a1 + NUMBER_WORD_GAP * n.h)  # a missed first piece is taken in
                filled = True
            else:
                box = copy_box(n)
                box.a0 = n.a1 + 0.3 * n.h
                box.a1 = min(following - 0.5 * n.h, box.a0 + SLOT_LENGTH * n.h)
                box.h = 1.2 * n.h
                filled = band_max(det.maps[WORD], box) >= WRITTEN
            slots.append((box, filled, n, not found))
        grid.append(slots)
    flat = [s for column in grid for s in column]
    count = sum(s[1] for s in flat)
    if count == 0:
        return None
    starts = np.cumsum([0] + [len(c) for c in grid])
    rows = max(len(c) for c in grid)
    grid_orders = [list(range(len(flat)))]
    if len(grid) > 1:
        grid_orders.append([int(starts[k] + r) for r in range(rows) for k in range(len(grid)) if r < len(grid[k])])
    index = {i: j for j, i in enumerate(i for i, s in enumerate(flat) if s[1])}
    orders = []
    for order in grid_orders:
        if all(flat[i][1] for i in order[:count]):
            mapped = [index[i] for i in order[:count]]
            if mapped not in orders:
                orders.append(mapped)
    if not orders:
        return None
    columns = [c for c in ([s[0] for s in column if s[1]] for column in grid) if c]
    return dict(columns=columns, count=count, orders=orders, numbers={index[i]: flat[i][2] for i in index},
                synthetic={index[i] for i in index if flat[i][3]})



def ctc_log_probs(log_probs: np.ndarray, labels: list[list[int]]) -> np.ndarray:
    """
        log P(label | frames) of labels of one or two classes, by the CTC
        forward algorithm run on all labels at once; log_probs (frames,
        classes), blank = class 0.
    """
    n = len(labels)
    ext = np.zeros((n, 5), int)                         # blank, c1, blank, c2, blank
    ext[:, 1] = [label[0] for label in labels]
    ext[:, 3] = [label[-1] for label in labels]
    two = np.array([len(label) == 2 for label in labels])
    length = np.where(two, 5, 3)
    skip = np.zeros((n, 5), bool)                       # transition s - 2 -> s allowed
    skip[:, 3] = two & (ext[:, 3] != ext[:, 1])
    valid = np.arange(5)[None, :] < length[:, None]
    a = np.full((n, 5), -np.inf)
    a[:, 0], a[:, 1] = log_probs[0, 0], log_probs[0, ext[:, 1]]
    for t in range(1, log_probs.shape[0]):
        previous = a
        a = previous.copy()
        a[:, 1:] = np.logaddexp(a[:, 1:], previous[:, :-1])
        a[:, 2:] = np.where(skip[:, 2:], np.logaddexp(a[:, 2:], previous[:, :-2]), a[:, 2:])
        a = np.where(valid, a + log_probs[t][ext], -np.inf)
    last = np.take_along_axis(a, (length - 1)[:, None], 1)[:, 0]
    before = np.take_along_axis(a, (length - 2)[:, None], 1)[:, 0]
    return np.logaddexp(last, before)



_number_labels = []



def number_probabilities(photo: Image.Image, quad: list, fill: int) -> np.ndarray | None:
    """
        Probabilities of the values 1 to MAX_NUMBER of the number in `quad`
        (photo pixels), read by the first word network: CTC probability of its
        digit strings ("7" and "07"), normalized over the values. None if no ink
        is found.
    """
    from . import load_engine
    ink = normalize_word(rectified_crop(photo, quad, fill))
    if ink is None:
        return None
    if not _number_labels:
        labels, values = [], []
        for v in range(1, MAX_NUMBER + 1):
            for s in sorted({str(v), f"{v:02d}"}):
                labels.append([native.FIRST_DIGIT_CLASS + int(ch) for ch in s])
                values.append(v - 1)
        _number_labels.extend((labels, values))
    labels, values = _number_labels
    scores = np.full(MAX_NUMBER, -np.inf)
    np.logaddexp.at(scores, values, ctc_log_probs(load_engine()[0].log_probs(ink).T.astype(np.float64), labels))
    return np.exp(scores - np.logaddexp.reduce(scores))



def candidate_orders(columns: list[list[Box]]) -> list[list[int]]:
    """
        Column by column, then row by row when there are several columns (word
        indices numbered column by column).
    """
    count = sum(len(c) for c in columns)
    orders = [list(range(count))]
    if len(columns) > 1:
        starts = np.cumsum([0] + [len(c) for c in columns])
        rows = max(len(c) for c in columns)
        by_rows = [int(starts[k] + r) for r in range(rows) for k in range(len(columns)) if r < len(columns[k])]
        if by_rows != orders[0]:
            orders.append(by_rows)
    return orders



def order_by_numbers(orders: list[list[int]], probabilities: dict) -> list[list[int]] | None:
    """
        The order supported by the numbers read (at least MIN_NUMBERS_READ) by
        ORDER_MARGIN over every other one, alone; None if there is none.
    """
    if len(probabilities) < MIN_NUMBERS_READ:
        return None
    scores = []
    for order in orders:
        s = 0.0
        for position, k in enumerate(order):
            p = probabilities.get(k)
            if p is not None:
                q = p[position] if position < len(p) else 0.0
                s += math.log(NUMBER_TRUST * q + (1 - NUMBER_TRUST) / MAX_NUMBER)
        scores.append(s)
    best = int(np.argmax(scores))
    if all(scores[best] - s >= ORDER_MARGIN for i, s in enumerate(scores) if i != best):
        return [orders[best]]
    return None



# ---------- orientation ----------

def turned_by_reading(photo: Image.Image, quads: list, fill: int) -> bool:
    """
        True if the tallest words read better turned by 180 degrees: sum over
        them of the score of the best dictionary word, per unit of width.
    """
    from . import load_engine
    first, decoder, _ = load_engine()
    tallest = sorted(range(len(quads)), key=lambda k: -np.linalg.norm(np.subtract(quads[k][3], quads[k][0])))
    lead = 0.0
    for k in tallest[:ORIENTATION_WORDS]:
        crop = rectified_crop(photo, quads[k], fill)
        best = []
        for im in (crop, crop.rotate(180)):
            ink = normalize_word(im)
            if ink is None:
                best.append(-20.0)
            else:
                best.append(float(decoder.scores(first.log_probs(ink)).astype(np.float64).max()) / max(1, ink.shape[1] / 32))
        lead += best[1] - best[0]
    return lead > ORIENTATION_MARGIN



# ---------- checks of a valid seed ----------

def number_contradicts(probabilities: np.ndarray | None, position: int) -> bool:
    """
        True if a number read with a probability of at least NUMBER_CHECK is
        not `position` + 1, which it is given a probability of less than
        NUMBER_CHECK_OTHER.
    """
    if probabilities is None or position >= len(probabilities):
        return False
    best = int(np.argmax(probabilities))
    return best != position and probabilities[best] >= NUMBER_CHECK and probabilities[position] < NUMBER_CHECK_OTHER


# ---------- reading ----------

def find_words(image: Image.Image, step=page.no_step, fingerprint=None) -> WordLayout:
    """
        Words of the card or list on `image` (upright grayscale photo, full
        resolution) found by the detector (see the steps above).
        `fingerprint(crop)`: see page.find_words; a list with one line more
        than a valid number of words leaves out the one line it accepts
        (`layout.fingerprint`, see page.spare_lines).
        `layout.doubtful`: words never certain (slots where the word map shows
        ink but no word box was found). Raises WordsNotFound if no document
        with a valid number of words is found.
    """
    det = detect(image)
    step(DETECTOR_COST)
    size = det.maps.shape[2], det.maps.shape[1]
    groups, number_groups = {}, {}
    for b in det.words:
        if b.score > 0:
            groups.setdefault(det.card_of(b.center), []).append(b)
    for n in det.numbers:
        number_groups.setdefault(det.card_of(n.center), []).append(n)
    candidates = []
    for g, boxes in groups.items():
        if len(boxes) >= MIN_GROUP_WORDS:
            columns = group_lines([copy_box(b) for b in boxes])
            n = sum(len(c) for c in columns)
            candidates.append((n in page.SEED_LENGTHS, n, g, columns))
    if not candidates:
        raise WordsNotFound("no words found")
    candidates.sort(key=lambda m: (m[0], m[1]), reverse=True)
    _, count, g, columns = candidates[0]
    ambiguous = any(m[1] >= MIN_OTHER_WORDS for m in candidates[1:])
    anchored = None
    if len(number_groups.get(g, [])) >= MIN_ANCHOR_NUMBERS:
        anchored = anchored_slots(det, groups[g], number_groups[g])
        if anchored and anchored["count"] in page.SEED_LENGTHS \
                and (count not in page.SEED_LENGTHS or anchored["count"] == count):
            columns, count = anchored["columns"], anchored["count"]
        else:
            anchored = None
    if anchored is None:
        grid = grid_columns(groups[g])
        if grid is not None and sum(len(c) for c in grid) in page.SEED_LENGTHS \
                and (count not in page.SEED_LENGTHS or len(grid) > len(columns)):
            columns, count = grid, sum(len(c) for c in grid)
    if anchored is None and count not in page.SEED_LENGTHS:
        aligned = aligned_columns(groups[g])
        if aligned is not None and sum(len(c) for c in aligned) in page.SEED_LENGTHS:
            columns, count = aligned, sum(len(c) for c in aligned)
    if g < 0 and count not in page.SEED_LENGTHS and len(columns) > 1:
        # short columns mostly on the frame are text of another sheet
        kept = [c for c in columns if len(c) >= MIN_COLUMN_LINES
                and 2 * sum(touches_frame(det.to_photo(b.quad()), image.size, 8) for b in c) < len(c)]
        if kept and sum(len(c) for c in kept) in page.SEED_LENGTHS:
            columns, count = kept, sum(len(c) for c in kept)
    fill = int(np.median(np.asarray(image.reduce(16))))
    spare = None
    if g < 0 and anchored is None and count not in page.SEED_LENGTHS and count - 1 in page.SEED_LENGTHS \
            and fingerprint is not None:
        accepted = []
        for c, i in page.spare_lines(columns):
            crop = rectified_crop(image, det.to_photo(columns[c][i].quad(CROP_VARIANTS[0])), fill)
            if fingerprint(crop):
                accepted.append((c, i, crop))
        if len(accepted) == 1:
            c, i, spare = accepted[0]
            columns = [[b for j, b in enumerate(column) if (k, j) != (c, i)] for k, column in enumerate(columns)]
            columns = [column for column in columns if column]
            count -= 1
    at_edge = any(column_at_edge(c, size) for c in columns) \
        or any(touches_frame(det.to_photo(b.quad()), image.size) for c in columns for b in c)
    if count not in page.SEED_LENGTHS:
        raise WordsNotFound(f"{count} lines found", at_edge=at_edge, line_count=count, card=g >= 0 and at_edge)
    lines = [b for c in columns for b in c]
    numbers = number_groups.get(g, [])
    linked = anchored["numbers"] if anchored else link_numbers(det, lines, numbers)
    quads = [[det.to_photo(b.quad(m)) for b in lines] for m in CROP_VARIANTS]
    turned = upside_down(lines, list(linked.values()) or numbers)
    if g >= 0 or (not turned and not numbers_before(lines, list(linked.values()) or numbers)):
        # no number on either side of the words: reading a few words both ways tells
        turned = turned_by_reading(image, quads[0], fill)

    first_number_needed = False
    if g >= 0 and det.cut_sides(g):
        # card cut after its text or below it: words may be missing; cut before or above it: the first number
        # must be 1
        ux, uy = (-lines[0].ux, -lines[0].uy) if turned else (lines[0].ux, lines[0].uy)
        for nx, ny in det.cut_sides(g):
            if nx * ux + ny * uy > 0.5 or -nx * uy + ny * ux > 0.5:
                at_edge = True
            else:
                first_number_needed = True
    orders = anchored["orders"] if anchored else candidate_orders(columns)
    if turned:
        quads = [[q[2:] + q[:2] for q in variant] for variant in quads]
        orders = [list(reversed(o)) for o in orders]

    read_numbers = {}

    def number(k):
        if k not in read_numbers:
            q = det.to_photo(linked[k].quad(NUMBER_CROP_MARGINS))
            step(NUMBER_COST)
            read_numbers[k] = number_probabilities(image, q[2:] + q[:2] if turned else q, fill)
        return read_numbers[k]

    if first_number_needed and not at_edge:
        p = number(orders[0][0]) if orders[0][0] in linked else None
        at_edge = p is None or p[0] < 0.5
    if len(orders) > 1 and linked:
        # numbers read first where the orders disagree, until one of them leads by ORDER_MARGIN
        positions = [{k: i for i, k in enumerate(o)} for o in orders]
        probabilities = {}
        for k in sorted(linked, key=lambda k: (len({p[k] for p in positions}) == 1, k)):
            p = number(k)
            if p is not None:
                probabilities[k] = p
            decided = order_by_numbers(orders, probabilities)
            if decided:
                orders = decided
                break

    def crop(k, variant):
        return rectified_crop(image, quads[variant][k], fill)

    kind = "list" if g < 0 else ("card24" if count > 12 else "card12")
    layout = WordLayout(kind, count, orders, len(CROP_VARIANTS), crop, lambda k: quads[0][k],
                        complete=not (at_edge or ambiguous))
    layout.doubtful = anchored["synthetic"] if anchored else set()
    layout.fingerprint = spare
    layout.lengths = [getattr(b, "ink", b.a1 - b.a0) / b.h for b in lines]

    def check(order: list[int]) -> bool:
        """
            True if the number of the first or last word of a seed read in
            `order` contradicts its position (page upside down or order wrong).
        """
        return any(number_contradicts(number(order[i]), i) for i in {0, len(order) - 1} if order[i] in linked)
    layout.check = check
    return layout
