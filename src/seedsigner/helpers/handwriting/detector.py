"""
    Word detector of Haller-1: a small fully convolutional network (C library,
    native/handwriting.c) that finds the handwritten words, the line numbers
    and the cards on the photo reduced to 288 x 384 pixels. It is the only
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
      0. a document covering a small part of the photo is detected again on
         its region (document_region), at a higher resolution;
      1. documents: each card kernel is a card; a word belongs to the card
         holding its center, other words are free (lists); the document read is
         the one with the most words, and another one of MIN_OTHER_WORDS words
         or more makes every word uncertain;
      2. lines and columns (group_lines), or a tight grid of rows
         (grid_columns); on a document with MIN_ANCHOR_NUMBERS numbers or more,
         one slot per number (anchored_slots);
      3. orientation: numbers after their words mean the page is upside down,
         numbers before them that it is upright; without numbers, the words
         reading better turned over;
      4. order: column by column or row by row; when both are possible, the
         numbers are read until one order leads by ORDER_MARGIN, otherwise both
         are given to the analysis, which makes every word uncertain if both
         give a plausible seed;
      5. completeness: no word is certain if a word touches the frame, a
         column may go on beyond it, or a card is cut after its text (or cut
         before it without the first number read being 1).
"""
import copy
import ctypes
import itertools
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
ALL = -2                      # document made of every word found, cards and free words together
MIN_OTHER_WORDS = 6           # words of another document making every word uncertain
BEYOND_TEXT = 0.3             # writing of a card farther than this many word heights beyond its text: card cut there
ZOOM_SHARE = 0.5              # documents covering less of the photo are detected again on their region (0.4 before
                              # the detector of 21/11/2026)
ZOOM_MARGIN = 0.15            # margin of that region, share of its size on each side

# Lines and columns, in word box heights
SAME_LINE = 0.55              # two boxes closer than this across the text direction are on the same line
SAME_WORD = 0.3               # two boxes closer than this across the text are on the same line even if they overlap
MIN_HEIGHT_SHARE = 0.6        # boxes lower than this share of the median height are dropped when the count is wrong
COLUMN_GAP = 1.5              # empty width between two columns
GRID_WORD_GAP = 0.9           # gap along a row above which two pieces are two words of a grid
MIN_GRID_ROWS_SHARE = 0.75    # share of the rows of a grid holding its full number of words
PIECE_GAP = 0.0               # gap between boxes (with their margins) above which two pieces may be two words
                              # of aligned columns: a blank of half a height
MIN_CUT_LENGTH = 1.2          # length of each part of a piece cut at the start of the next column (aligned_columns)
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
MIN_DRIFT_WORDS = 6           # words on the lines of their numbers needed to fit that drift
MAX_DRIFT = 0.2               # largest slope of the lines from the text direction
SLOT_SLACK = 1.8              # a word off every number line goes to a free slot this many times SAME_LINE away
NUMBER_WORD_GAP = 0.15        # start of a slot word after the end of its number, in number heights
LINK_THRESHOLD = 0.3          # mean link probability along a number-word segment
NUMBER_GAP = 3.0              # word heights between a number and its word, without link
NUMBER_CROP_MARGINS = (0.4, 0.9, 0.9)  # crop of a number around the center line of its box, in box heights
MAX_NUMBER = 33
NUMBER_TRUST = 0.7            # a number reading is right with this probability, otherwise any value
MIN_NUMBERS_READ = 3          # numbers read before an order may be decided by them
ORDER_MARGIN = math.log(1e4)  # log-likelihood lead of the best order over the others
MIN_ROW_WORDS = 3             # words of each row but the last of words written in rows (row_words)
MIN_ROW_WORD_LENGTH = 1.2     # length of every word written in rows (three letters at least), in heights
BAND_GAP = 1.2                # largest gap between two lines of every column, at the same row, above this many
                              # median line spacings: the page may be two bands read one after the other

# Orientation: the tallest ORIENTATION_WORDS words are read in both directions; the page is upside down if the
# turned readings lead by ORIENTATION_MARGIN (summed log-likelihoods per unit of width)
ORIENTATION_WORDS = 3
ORIENTATION_MARGIN = 1.0

# Word crops around the box, in box heights (horizontal, top, bottom): first reading, taller band, wider band
CROP_VARIANTS = ((0.6, 0.85, 0.75), (0.6, 1.1, 1.0), (1.2, 0.8, 0.7))
# Crops of a line given to the fingerprint judge, first one accepted: first reading, lower band
FINGERPRINT_CROPS = (CROP_VARIANTS[0], (0.6, 0.6, 0.5))
# Lines outside the words tried as the fingerprint: at least this long (text heights; 8 hexadecimal digits are
# longer), the longest ones first, at most this many
FINGERPRINT_MIN_LENGTH = 2.0
FINGERPRINT_MAX_LINES = 3
# A word box whose background (OFF_SHEET_PERCENTILE of the gray levels of its crop) is darker than OFF_SHEET times
# the median background of the words of its document is off the sheet, left out when the count is wrong (real photos
# of the Pi, 21/11/2026: phantom words on the dark table below the sheet at 0.02 to 0.11, words on paper at 0.79 or more)
# A word box at least MAX_HEIGHT_SHARE times taller than the median is an object around the sheet, left out when the
# count is wrong (series of the Pi of 21/11/2026: an object at the top of the frame read as a word 2.5 times taller)
MAX_HEIGHT_SHARE = 2.0
OFF_SHEET = 0.4
OFF_SHEET_PERCENTILE = 90
# Gray levels under the boxes are measured on the photo reduced this many times, on the box widened by STATS_MARGIN
# heights above and below
STATS_REDUCE = 4
STATS_MARGIN = 0.35
# A word box with a share of dark pixels at least CROSSED_OUT times the median share of the words of the photo may be
# a word scribbled over (real photo of the Pi, 21/11/2026: 0.66 for the line crossed out, 0.24 to 0.40 for the 15
# words; but up to 2.7 times the median for short or bold words of other photos): it is left out, with the number
# written against it, only by a second search made when the first reading is neither verified nor valid (see
# read_seed in __init__.py)
CROSSED_OUT = 1.8
# A word kernel of fewer than SPECK times the median pixels of the kernels of its document is a speck or text seen
# through the sheet, left out when the count is wrong (same photo: 4 and 7 pixels against a median of 90)
SPECK = 0.15
# a number box ending at most this many heights before the start of a line tried as the fingerprint, or starting at
# most this many heights after its end, is taken in with it (its first or last digits read as a number), the line
# alone tried next
FINGERPRINT_NUMBER_GAP = 0.6

# No word box found but at least SMALL_PIXELS output pixels of word probability SMALL_WORD_PROBABILITY or more, of a
# median height under SMALL_HEIGHT output pixels: words too small to be found (real photo of the Pi, 21/11/2026:
# sheet far from the camera, 8 pixels at 0.2-0.5, height 3.7; words of a readable photo 5.5 or more)
SMALL_WORD_PROBABILITY = 0.2
SMALL_PIXELS = 4
SMALL_HEIGHT = 4.8

# A first or last row of a list holding at most this many pieces is tried as the fingerprint when the other words
# make a valid count (extra_rows: « FP: 1a2b 3c4d »)
MAX_EXTRA_ROW_PIECES = 3

# Expected duration on a Pi Zero, in seconds (see Progress in __init__.py): network and decoding, reading of a number
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



def reduce_photo(photo: Image.Image) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
    """
        Photo fitted into the input of the network, aspect kept, padded with its
        median gray: (image, scale from the photo to the input, offset of the
        photo in the input, its size there).
    """
    width, height = native.DETECTOR_SIZE
    s = min(width / photo.width, height / photo.height)
    w, h = max(1, round(photo.width * s)), max(1, round(photo.height * s))
    small = photo.resize((w, h), Image.BOX)
    canvas = Image.new("L", (width, height), int(np.median(np.asarray(small))))
    offset = ((width - w) // 2, (height - h) // 2)
    canvas.paste(small, offset)
    return np.asarray(canvas), s, offset, (w, h)



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
    def __init__(self, maps: np.ndarray, to_photo, frame: dict):
        self.maps = maps
        self.to_photo = to_photo
        self.frame = frame  # outward normal of a side of the photo: its position across, in output pixels
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
            Outward normals (x, y) of the sides of the photo (`frame`) touched
            by `card`: by its kernel, or by the card area holding it and no
            other card kernel.
        """
        p = self.cards[card]
        y, x = p[:, 0], p[:, 1]
        for area in components(self.card_area):
            held = set(np.unique(self.labels[area[:, 0], area[:, 1]]).tolist()) - {0}
            if held == {card + 1}:
                y, x = np.concatenate([y, area[:, 0]]), np.concatenate([x, area[:, 1]])
                break
        extent = {(0, -1): -y.min(), (0, 1): y.max() + 1, (-1, 0): -x.min(), (1, 0): x.max() + 1}
        return [n for n, at in self.frame.items() if extent[n] >= at - 1]


    def link_score(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """
            Mean link probability along the segment from a to b (output pixels).
        """
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])))
        xs = np.clip(np.linspace(a[0], b[0], n).astype(int), 0, self.maps.shape[2] - 1)
        ys = np.clip(np.linspace(a[1], b[1], n).astype(int), 0, self.maps.shape[1] - 1)
        return float(sigmoid(self.maps[LINK][ys, xs]).mean())



def detect(photo: Image.Image, region: tuple[int, int, int, int] | None = None) -> Detection:
    """
        Detection on an upright grayscale photo, or on the region (left, top,
        right, bottom, photo pixels) of it.
    """
    x0, y0, x1, y1 = region or (0, 0, photo.width, photo.height)
    image, s, (ox, oy), (w, h) = reduce_photo(photo.crop(region) if region else photo)
    # sides of the region that are sides of the photo, where the photo ends in the output maps (padding left out)
    sides = {(0, -1): (y0 <= 0, -oy / 2), (0, 1): (y1 >= photo.height, (oy + h) / 2),
             (-1, 0): (x0 <= 0, -ox / 2), (1, 0): (x1 >= photo.width, (ox + w) / 2)}
    frame = {n: at for n, (on, at) in sides.items() if on}
    return Detection(network_maps(image),
                     lambda points: [((2 * u - ox) / s + x0, (2 * v - oy) / s + y0) for u, v in points], frame)



def on_card(det: Detection, point: tuple[float, float], card: int) -> bool:
    """
        True if the point (output pixels) lies on `card`: on its kernel or on
        the card area around it.
    """
    if det.card_of(point) == card:
        return True
    x, y = int(point[0]), int(point[1])
    h, w = det.card_area.shape
    return 0 <= y < h and 0 <= x < w and bool(det.card_area[y, x]) and det.card_of(point) in (card, -1)



def document_region(det: Detection, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """
        Region of the photo (photo pixels, `size`) holding the words and cards
        found, widened by ZOOM_MARGIN of its size on each side, if it covers less
        than ZOOM_SHARE of the area seen by the network: the words are then
        found again on it, at a higher resolution. None otherwise.
    """
    points = [p for b in det.words if b.score > 0 for p in b.quad()]
    points += [p for c in det.cards for p in ((c[:, 1].min(), c[:, 0].min()), (c[:, 1].max() + 1, c[:, 0].max() + 1))]
    if len(points) < 4 * MIN_GROUP_WORDS:
        return None
    xs, ys = zip(*det.to_photo(points))
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = ZOOM_MARGIN * (x1 - x0), ZOOM_MARGIN * (y1 - y0)
    x0, x1 = max(0, int(x0 - mx)), min(size[0], int(x1 + mx) + 1)
    y0, y1 = max(0, int(y0 - my)), min(size[1], int(y1 + my) + 1)
    if (x1 - x0) * (y1 - y0) >= ZOOM_SHARE * size[0] * size[1] or x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1



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



def covers(line: Box, point: tuple[float, float], margin: float = 0.2, across: float = 0.6) -> bool:
    """
        True if the output pixel `point` lies on the box `line`, widened by
        `margin` heights along the text, within `across` heights of its center
        line.
    """
    x, y = point
    a, b = x * line.ux + y * line.uy, -x * line.uy + y * line.ux
    return line.a0 - margin * line.h <= a <= line.a1 + margin * line.h and abs(b - line.c) <= across * line.h



class InkStats:
    """
        Gray levels under the word boxes, on the photo reduced STATS_REDUCE
        times (cheap on a Pi Zero), computed once per box: background
        (OFF_SHEET_PERCENTILE of its crop), ink (5th percentile) and crop
        pixels (box with a margin of STATS_MARGIN heights above and below).
    """
    def __init__(self, image: Image.Image, det: Detection):
        self.small = image.reduce(STATS_REDUCE)
        self.det = det
        self.cache = {}


    def crop(self, box: Box) -> np.ndarray:
        if id(box) not in self.cache:
            quad = [(x / STATS_REDUCE, y / STATS_REDUCE)
                    for x, y in self.det.to_photo(box.quad((0.0, STATS_MARGIN, STATS_MARGIN)))]
            self.cache[id(box)] = np.asarray(rectified_crop(self.small, quad, 0), dtype=float)
        return self.cache[id(box)]


    def background(self, box: Box) -> float:
        c = self.crop(box)
        return float(np.percentile(c, OFF_SHEET_PERCENTILE)) if c.size else 0.0


    def ink(self, box: Box) -> float:
        c = self.crop(box)
        return float(np.percentile(c, 5)) if c.size else 0.0


    def off_sheet(self, boxes: list[Box]) -> set[int]:
        """
            Indices of the boxes whose background is darker than OFF_SHEET
            times the median background of `boxes`: off the sheet.
        """
        levels = [self.background(b) for b in boxes]
        median = float(np.median(levels))
        return {i for i, v in enumerate(levels) if v < OFF_SHEET * median}


    def crossed_out(self, boxes: list[Box]) -> set[int]:
        """
            Indices of the boxes whose share of dark pixels (darker than
            halfway between their background and the median ink of `boxes`)
            is at least CROSSED_OUT times the median share: a word scribbled
            over.
        """
        if len(boxes) < MIN_GROUP_WORDS:
            return set()
        ink = float(np.median([self.ink(b) for b in boxes]))
        shares = []
        for b in boxes:
            c, bg = self.crop(b), self.background(b)
            shares.append(float((c < bg - 0.5 * (bg - ink)).mean()) if c.size else 0.0)
        median = float(np.median(shares))
        return {i for i, v in enumerate(shares) if median > 0 and v >= CROSSED_OUT * median}



def with_number(line: Box, numbers: list[Box], after: bool = False) -> Box | None:
    """
        The line widened to the number boxes centered before its start on its
        line and ending at most FINGERPRINT_NUMBER_GAP heights before it, or
        overlapping it (a first digit of a fingerprint taken for a line number), or None
        if there is none. With `after`, also widened to the number boxes
        running past its end and starting at most FINGERPRINT_NUMBER_GAP
        heights after it (last digits of a fingerprint taken for a number).
    """
    on_line = [n for n in numbers if abs(n.c - line.c) < SAME_LINE * max(line.h, n.h)]
    before = [n for n in on_line if n.middle < line.a0 and n.a1 >= line.a0 - FINGERPRINT_NUMBER_GAP * line.h]
    beyond = [n for n in on_line if after and n.middle > line.middle and n.a1 > line.a1
              and n.a0 <= line.a1 + FINGERPRINT_NUMBER_GAP * line.h]
    if not before and not beyond:
        return None
    wide = copy_box(line)
    wide.a0 = min([n.a0 for n in before] + [line.a0])
    wide.a1 = max([n.a1 for n in beyond] + [line.a1])
    return wide



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
            gap = abs(b.c - lines[-1].c) if lines else math.inf
            if gap < SAME_WORD * max(h, b.h) or (gap < SAME_LINE * max(h, b.h) and not stacked(lines[-1], b)):
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
    # a piece running over the start of the next column (a word of the left column merged with the number and the
    # word written against it) is cut there: the left part ends halfway between the usual end of the words of its
    # column and the start of the next one, the right part starts with the words of the next column
    for j in range(1, len(columns)):
        ends = [p.a1 for row in pieces for p in row
                if columns[j - 1] - ALIGNMENT * h <= p.a0 <= columns[j - 1] + ALIGNMENT * h and p.a1 < columns[j]]
        if not ends:
            continue
        cut = (float(np.median(ends)) + columns[j]) / 2
        for r, row in enumerate(pieces):
            out = []
            for p in row:
                if p.a0 < cut - MIN_CUT_LENGTH * h and p.a1 > columns[j] + MIN_CUT_LENGTH * h:
                    for a0, a1 in ((p.a0, cut), (columns[j], p.a1)):
                        piece = copy_box(p)
                        piece.a0, piece.a1 = a0, a1
                        piece.pixels = max(1, int(round(p.pixels * (a1 - a0) / (p.a1 - p.a0))))
                        out.append(piece)
                else:
                    out.append(p)
            pieces[r] = out
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



def chained_rows(boxes: list[Box], h: float) -> list[list[Box]]:
    """
        Rows of boxes followed from one box to the next along the text: each
        box, from the start of the rows, joins the row whose last box is
        nearest across the text, within SAME_LINE heights (a slanted row
        drifts across the text more than SAME_LINE heights from start to end,
        never between two neighbouring words). Rows top to bottom.
    """
    rows = []
    for b in sorted(boxes, key=lambda b: b.a0):
        near = [(abs(b.c - row[-1].c), k) for k, row in enumerate(rows)
                if abs(b.c - row[-1].c) < SAME_LINE * max(h, b.h) and not stacked(row[-1], b)]
        if near:
            rows[min(near)[1]].append(b)
        else:
            rows.append([b])
    return sorted(rows, key=lambda row: float(np.mean([b.c for b in row])))



def row_words(boxes: list[Box], chained: bool = False) -> list[list[Box]] | None:
    """
        Words written one after the other along a few rows, without columns
        (« wide toss supply version motor / pupil toy … »): rows of boxes on
        the same line, split into words where the gap exceeds GRID_WORD_GAP
        heights, read left to right, row after row. Every row but the last
        holds at least MIN_ROW_WORDS words and every word is at least
        MIN_ROW_WORD_LENGTH heights long. The rows, as lists of words, or None.
        With `chained`, the rows are followed from box to box (chained_rows,
        slanted rows close to each other) and every box is a word.
    """
    if len(boxes) < MIN_ROW_WORDS:
        return None
    boxes = [copy_box(b) for b in boxes]
    h = float(np.median([b.h for b in boxes]))
    if chained:
        words = [sorted(row, key=lambda b: b.a0) for row in chained_rows(boxes, h)]
    else:
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
    if any(len(r) < MIN_ROW_WORDS for r in words[:-1]):
        return None
    if any(b.a1 - b.a0 < MIN_ROW_WORD_LENGTH * h for r in words for b in r):
        return None
    return words



def word_columns(boxes: list[Box]) -> list[list[Box]] | None:
    """
        Columns of the words of `boxes` as find_words makes them without
        numbers: lines in columns (group_lines), or else a grid of words
        written in rows (grid_columns, aligned_columns), if one of them has a
        valid number of words; None otherwise.
    """
    columns = group_lines([copy_box(b) for b in boxes])
    if sum(len(c) for c in columns) in page.SEED_LENGTHS:
        return columns
    for other in (grid_columns(boxes), aligned_columns(boxes)):
        if other is not None and sum(len(c) for c in other) in page.SEED_LENGTHS:
            return other
    return None



def extra_rows(boxes: list[Box]) -> list[tuple[Box, list[Box], list[list[Box]]]]:
    """
        The first and the last row of `boxes` (boxes on the same line across
        the whole document) holding at most MAX_EXTRA_ROW_PIECES pieces split
        where the gap exceeds GRID_WORD_GAP heights, when the other boxes make
        columns of a valid number of words (word_columns): (the row as one
        line, its boxes along the text, those columns) for each.
    """
    if len(boxes) < MIN_GROUP_WORDS + 1:
        return []
    h = float(np.median([b.h for b in boxes]))
    rows = []
    for b in sorted(boxes, key=lambda b: b.c):
        if rows and abs(b.c - np.mean([x.c for x in rows[-1]])) < SAME_LINE * max(h, b.h):
            rows[-1].append(b)
        else:
            rows.append([b])
    found = []
    for row in ([rows[0], rows[-1]] if len(rows) > 1 else []):
        row = sorted(row, key=lambda b: b.a0)
        pieces = 1 + sum(b.a0 - a.a1 > GRID_WORD_GAP * h for a, b in zip(row, row[1:]))
        if pieces > MAX_EXTRA_ROW_PIECES:
            continue
        columns = word_columns([b for b in boxes if not any(b is x for x in row)])
        if columns is None:
            continue
        line = copy_box(row[0])
        for b in row[1:]:
            line.merge(copy_box(b))
        found.append((line, row, columns))
    return found



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
    drift = [0.0, 0.0]

    def gap(b, n):
        # distance across the text from a word to the line of a number, once the drift is taken out
        return abs(b.c - n.c - drift[0] * (b.middle - n.middle) - drift[1])

    def on_line(b, n, following, tolerance):
        return gap(b, n) < tolerance * max(b.h, n.h) and b.a1 > n.a1 and b.middle < following - 0.3 * n.h

    followings = [starts[ci + 1] if ci + 1 < len(number_columns) else math.inf for ci in range(len(number_columns))]
    # drift of the words from the lines of their numbers: slope of the lines on the photo (perspective) and
    # height of the writing above them, fitted on the words nearest to a number on its line
    nearest = {}
    for ci, column in enumerate(number_columns):
        for n in column:
            for k, b in enumerate(words):
                if on_line(b, n, followings[ci], SAME_LINE) and gap(b, n) < nearest.get(k, (math.inf,))[0]:
                    nearest[k] = (gap(b, n), b.middle - n.middle, b.c - n.c, max(b.h, n.h))
    if len(nearest) >= MIN_DRIFT_WORDS:
        x, y, h = (np.array(v) for v in zip(*[(a, c, hh) for _, a, c, hh in nearest.values()]))
        keep = np.ones(len(x), bool)
        for _ in range(2):
            slope, offset = np.polyfit(x[keep], y[keep], 1) if np.ptp(x[keep]) > 0 else (0.0, float(np.median(y)))
            keep = np.abs(y - slope * x - offset) < 0.5 * SAME_LINE * h
            if keep.sum() < MIN_DRIFT_WORDS:
                break
        limit = SAME_LINE * float(np.median(h))
        drift[:] = float(np.clip(slope, -MAX_DRIFT, MAX_DRIFT)), float(np.clip(offset, -limit, limit))
    # words on the line of their number; a word left over goes to the nearest number with no word on its line
    # within SLOT_SLACK times that distance (words written above or below the printed line)
    taken = {id(b) for ci, column in enumerate(number_columns) for n in column for b in words
             if on_line(b, n, followings[ci], SAME_LINE)}
    # words left over are placed nearest first; a number takes several of them only if they are on one line
    pairs = sorted((gap(b, n), k, id(n), b) for k, b in enumerate(words) if id(b) not in taken
                   for ci, column in enumerate(number_columns) for n in column
                   if on_line(b, n, followings[ci], SLOT_SLACK * SAME_LINE)
                   and not any(on_line(w, n, followings[ci], SAME_LINE) for w in words))
    extra, placed = {}, set()
    for _, k, n_id, b in pairs:
        if k not in placed and all(abs(w.c - b.c) < SAME_LINE * max(w.h, b.h) for w in extra.get(n_id, [])):
            extra.setdefault(n_id, []).append(b)
            placed.add(k)
    # a word box on the lines of several numbers holds the words of all of them (words touching across lines): each
    # number takes the part of it on its own line, of the height of the words on one line
    found_of = {id(n): [b for b in words if on_line(b, n, followings[ci], SAME_LINE)] or extra.get(id(n), [])
                for ci, column in enumerate(number_columns) for n in column}
    users = {}
    for n_id, found in found_of.items():
        for b in found:
            users.setdefault(id(b), []).append(n_id)
    shared = {i for i, u in users.items() if len(u) > 1}
    if shared:
        single = [b.h for b in words if id(b) in users and id(b) not in shared]
        by_id = {id(n): n for column in number_columns for n in column}
        for n_id, found in found_of.items():
            n = by_id[n_id]
            parts = []
            for b in found:
                if id(b) in shared:
                    b = copy_box(b)
                    b.h = float(np.median(single)) if single else 1.5 * n.h
                    b.c = n.c + drift[0] * (b.middle - n.middle) + drift[1]
                parts.append(b)
            found_of[n_id] = parts
    grid = []
    for ci, column in enumerate(number_columns):
        following = followings[ci]
        slots = []
        for n in column:
            found = found_of[id(n)]
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
        banded = band_order([[s[2] for s in column] for column in grid])
        if banded is not None and banded not in grid_orders:
            grid_orders.append(banded)
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



def band_order(columns: list[list[Box]]) -> list[int] | None:
    """
        Column by column within two bands of lines read one after the other
        (two columns of 3 words above two columns of 3 words: 1-3 | 4-6, then
        7-9 | 10-12), word indices numbered column by column. The columns have
        the same number of lines, and each one has its largest gap between
        two lines at the same row, more than BAND_GAP times its median line
        spacing: the bands are split there. None otherwise.
    """
    if len(columns) < 2 or len({len(c) for c in columns}) > 1 or len(columns[0]) < 4:
        return None
    cut = None
    for c in columns:
        gaps = [b.c - a.c for a, b in zip(c, c[1:])]
        k = int(np.argmax(gaps))
        if gaps[k] <= BAND_GAP * float(np.median(gaps)) or cut not in (None, k):
            return None
        cut = k
    rows = len(columns[0])
    return [k * rows + r for band in (range(cut + 1), range(cut + 1, rows)) for k in range(len(columns)) for r in band]



def candidate_orders(columns: list[list[Box]], bands: bool = False) -> list[list[int]]:
    """
        Column by column, then row by row when there are several columns (word
        indices numbered column by column), then, with `bands` (numbered
        lines, which tell the order), column by column within bands
        (band_order) when the page is split into bands.
    """
    count = sum(len(c) for c in columns)
    orders = [list(range(count))]
    if len(columns) > 1:
        starts = np.cumsum([0] + [len(c) for c in columns])
        rows = max(len(c) for c in columns)
        by_rows = [int(starts[k] + r) for r in range(rows) for k in range(len(columns)) if r < len(columns[k])]
        if by_rows != orders[0]:
            orders.append(by_rows)
        banded = band_order(columns) if bands else None
        if banded is not None and banded not in orders:
            orders.append(banded)
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



# ---------- reading ----------

_detections = {}


def detections(image: Image.Image, step) -> Detection:
    """
        Detection of the document on `image` (see find_words), kept for a
        second search on the same photo (the last photo only).
    """
    key = (id(image), image.size)
    if key not in _detections:
        _detections.clear()
        det = detect(image)
        step(DETECTOR_COST)
        region = document_region(det, image.size)
        if region:
            det = detect(image, region)
            step(DETECTOR_COST)
        _detections[key] = det
    return _detections[key]



def find_words(image: Image.Image, step=page.no_step, fingerprint=None, crossed: bool = False) -> WordLayout:
    """
        Words of the card or list on `image` (upright grayscale photo, full
        resolution) found by the detector (see the steps above).
        `fingerprint(crop)`: True if the line on `crop` reads as a master key
        fingerprint; a list with one line more than a valid number of words
        leaves out the one line it accepts (`layout.fingerprint`, see
        page.spare_lines).
        `layout.doubtful`: words never certain (slots where the word map shows
        ink but no word box was found). Raises WordsNotFound if no document
        with a valid number of words is found.
    """
    det = detections(image, step)
    size = det.maps.shape[2], det.maps.shape[1]
    stats = InkStats(image, det)
    if crossed:
        # second search: the words scribbled over are left out, with the number written against them (a line crossed
        # out and written again); none: nothing new to read
        found = [b for b in det.words if b.score > 0]
        out = stats.crossed_out(found)
        scribbled = [b for i, b in enumerate(found) if i in out]
        if not scribbled:
            raise WordsNotFound("no word crossed out")
        det = copy.copy(det)
        det.words = [b for b in det.words if not any(b is c for c in scribbled)]
        det.numbers = [n for n in det.numbers if not any(with_number(c, [n]) for c in scribbled)]
    groups, number_groups = {}, {}
    for b in det.words:
        if b.score > 0:
            groups.setdefault(det.card_of(b.center), []).append(b)
    stray = [g for g, boxes in groups.items() if g >= 0 and len(boxes) < MIN_GROUP_WORDS]
    for g in stray:
        # a card kernel holding fewer words is no document: its words and numbers are free
        groups.setdefault(-1, []).extend(groups.pop(g))
    for n in det.numbers:
        g = det.card_of(n.center)
        number_groups.setdefault(-1 if g in stray else g, []).append(n)
    candidates = []

    def add_candidate(g, boxes):
        columns = group_lines([copy_box(b) for b in boxes])
        n = sum(len(c) for c in columns)
        if n not in page.SEED_LENGTHS:
            # boxes on a background much darker than that of the others are off the sheet (the table or the shadow
            # around it), boxes much taller than the others are objects around the sheet: never words
            h = float(np.median([b.h for b in boxes]))
            dark = stats.off_sheet(boxes) | {i for i, b in enumerate(boxes) if b.h >= MAX_HEIGHT_SHARE * h}
            if dark and len(boxes) - len(dark) >= MIN_GROUP_WORDS:
                boxes = [b for i, b in enumerate(boxes) if i not in dark]
                groups[g] = boxes
                columns = group_lines([copy_box(b) for b in boxes])
                n = sum(len(c) for c in columns)
        if n not in page.SEED_LENGTHS:
            # word kernels much smaller than the others are specks or text seen through the sheet: never words
            median = float(np.median([b.pixels for b in boxes]))
            specks = [b for b in boxes if b.pixels < SPECK * median]
            if specks and len(boxes) - len(specks) >= MIN_GROUP_WORDS:
                boxes = [b for b in boxes if b.pixels >= SPECK * median]
                groups[g] = boxes
                columns = group_lines([copy_box(b) for b in boxes])
                n = sum(len(c) for c in columns)
        if n not in page.SEED_LENGTHS:
            # boxes much lower than the others are specks or text seen through the sheet
            h = float(np.median([b.h for b in boxes]))
            kept = [b for b in boxes if b.h >= MIN_HEIGHT_SHARE * h]
            if MIN_GROUP_WORDS <= len(kept) < len(boxes):
                other = group_lines([copy_box(b) for b in kept])
                if sum(len(c) for c in other) in page.SEED_LENGTHS:
                    groups[g], columns, n = kept, other, sum(len(c) for c in other)
        if n not in page.SEED_LENGTHS and len(number_groups.get(g, [])) >= MIN_ANCHOR_NUMBERS:
            # the numbers give the count of a document whose words are split or merged
            anchored = anchored_slots(det, boxes, number_groups[g])
            if anchored and anchored["count"] in page.SEED_LENGTHS:
                n = anchored["count"]
        cut = g >= 0 and bool(det.cut_sides(g))
        candidates.append((n in page.SEED_LENGTHS, not cut, n, g, columns))

    for g, boxes in list(groups.items()):
        if len(boxes) >= MIN_GROUP_WORDS:
            add_candidate(g, boxes)
    if len(groups) > 1 and not any(m[0] for m in candidates):
        # no document with a valid number of words: the card areas may be parts of one sheet
        groups[ALL] = [b for boxes in groups.values() for b in boxes]
        number_groups[ALL] = list(det.numbers)
        add_candidate(ALL, groups[ALL])
        if candidates[-1][0]:
            candidates = [candidates[-1]]
    if not candidates:
        # faint writing lower than any word box: words too small in the photo (sheet far from the camera)
        faint = sigmoid(det.maps[WORD]) >= SMALL_WORD_PROBABILITY
        small = int(faint.sum()) >= SMALL_PIXELS \
            and float(np.median(det.maps[HEIGHT][faint])) * HEIGHT_SCALE < SMALL_HEIGHT
        raise WordsNotFound("words too small" if small else "no words found", small=small)
    # a card cut by the frame is read only if no whole document has a valid number of words, and does not make
    # the reading of another one uncertain
    candidates.sort(key=lambda m: m[:3], reverse=True)
    _, _, count, g, columns = candidates[0]
    count = sum(len(c) for c in columns)
    ambiguous = any(m[2] >= MIN_OTHER_WORDS and m[1] for m in candidates[1:])
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
    in_rows = False
    if g < 0 and anchored is None and count not in page.SEED_LENGTHS:
        for chained in (False, True):
            rows = row_words(groups[g], chained)
            if rows is not None and sum(len(r) for r in rows) in page.SEED_LENGTHS:
                # words written in rows: each row is taken as a column, read in that order only; rows followed from
                # box to box (slanted rows close to each other) only if the rows as lines do not give a valid count
                columns, count, in_rows = rows, sum(len(r) for r in rows), True
                break
    if g < 0 and count not in page.SEED_LENGTHS and len(columns) > 1:
        # short columns mostly on the frame are text of another sheet
        kept = [c for c in columns if len(c) >= MIN_COLUMN_LINES
                and 2 * sum(touches_frame(det.to_photo(b.quad()), image.size, 8) for b in c) < len(c)]
        if kept and sum(len(c) for c in kept) in page.SEED_LENGTHS:
            columns, count = kept, sum(len(c) for c in kept)
    fill = int(np.median(np.asarray(image.reduce(16))))
    spare = spare_first = None
    hypothesis = False
    if g < 0 and anchored is None and count not in page.SEED_LENGTHS and fingerprint is not None:
        # lines that may be a fingerprint written with the words: (line, its pieces, columns of the words without it)
        tried = []
        if count - 1 in page.SEED_LENGTHS:
            for c, i in page.spare_lines(columns):
                line = columns[c][i]
                rest = [[b for j, b in enumerate(column) if (k, j) != (c, i)] for k, column in enumerate(columns)]
                tried.append((line, sorted((b for b in groups[g] if covers(line, b.center)), key=lambda b: b.a0),
                              [column for column in rest if column]))
        for line, pieces, rest in extra_rows(groups[g]):
            # first or last row of the document (a grid whose columns are too close to be told apart, a fingerprint
            # in a cell of the grid): tried if the words without it make a valid count
            if not any(covers(t[0], line.center) for t in tried):
                tried.append((line, pieces, rest))
        accepted, crops = [], {}
        for t, (line, pieces, rest) in enumerate(tried):
            # the whole line, then its last word alone (a fingerprint written after a label: "fp: 1a2b3c4d")
            wide = with_number(line, number_groups.get(g, []), after=True)
            for box, margins in itertools.product([wide] * (wide is not None) + [line] + pieces[-1:] * (len(pieces) > 1),
                                                  FINGERPRINT_CROPS):
                crop = rectified_crop(image, det.to_photo(box.quad(margins)), fill)
                crops.setdefault(t, []).append(crop)
                if fingerprint(crop):
                    accepted.append((t, crop))
                    break
        margin_of = getattr(fingerprint, "margin", None)
        if not accepted and margin_of is not None:
            # no line passes the thresholds: the line read best as 8 hexadecimal digits by the fingerprint network is
            # left out as a hypothesis (layout.fingerprint_hypothesis), kept by read_seed only if its fingerprint
            # matches a seed read (handwriting.LOOSE_FINGERPRINT_MARGIN)
            # margin_of(crop): margin of the line read as a fingerprint, or None below the threshold; exactly one
            # line must pass it, its first crop passing it taken (the widest first, as above)
            passing = []
            for t, found in crops.items():
                crop = next((crop for crop in found if margin_of(crop) is not None), None)
                if crop is not None:
                    passing.append((t, crop))
            if len(passing) == 1:
                accepted, hypothesis = passing, True
        if len(accepted) == 1:
            t, spare = accepted[0]
            spare_first, _, columns = tried[t]
            count = sum(len(c) for c in columns)
    at_edge = any(column_at_edge(c, size) for c in columns) \
        or any(touches_frame(det.to_photo(b.quad()), image.size) for c in columns for b in c)
    if count not in page.SEED_LENGTHS:
        raise WordsNotFound(f"{count} lines found", at_edge=at_edge, line_count=count, card=g >= 0 and at_edge)
    lines = [b for c in columns for b in c]
    spare_line = spare_first
    if spare is None and fingerprint is not None:
        # a line of the document outside the words (above, below or beside them, numbered or not, with a list
        # counted by its numbers too) is tried as the fingerprint before it is taken for a word left out
        left = [copy_box(b) for b in det.words if b.score > 0 and det.card_of(b.center) in (g, -1)
                and not any(covers(line, b.center) for line in lines)]
        h = float(np.median([b.h for b in lines]))
        tried = sorted((line for column in group_lines(left) for line in column
                        if (line.a1 - line.a0) >= FINGERPRINT_MIN_LENGTH * h),
                       key=lambda line: -(line.a1 - line.a0))[:FINGERPRINT_MAX_LINES]
        accepted = []
        for line in tried:
            pieces = sorted((b for b in left if covers(line, b.center)), key=lambda b: b.a0)
            wide = with_number(line, det.numbers, after=True)
            for box, margins in itertools.product([wide] * (wide is not None) + [line] + pieces[-1:] * (len(pieces) > 1),
                                                  FINGERPRINT_CROPS):
                crop = rectified_crop(image, det.to_photo(box.quad(margins)), fill)
                if fingerprint(crop):
                    accepted.append((line, crop))
                    break
        if len(accepted) == 1:
            spare_line, spare = accepted[0]
    extra = [(b.score, (b.a1 - b.a0) / b.h, det.card_of(b.center) != g) for b in det.words
             if not any(covers(line, b.center) for line in lines + ([spare_line] if spare_line else []))]
    numbers = number_groups.get(g, [])
    linked = anchored["numbers"] if anchored else link_numbers(det, lines, numbers)
    quads = [[det.to_photo(b.quad(m)) for b in lines] for m in CROP_VARIANTS]
    turned = upside_down(lines, list(linked.values()) or numbers)
    if not turned and not numbers_before(lines, list(linked.values()) or numbers):
        # no number on either side of the words: reading a few words both ways tells
        turned = turned_by_reading(image, quads[0], fill)

    first_number_needed = last_number_needed = False
    if g >= 0 and det.cut_sides(g):
        # card reaching the frame after its text or below it: cut if writing of the card is seen beyond its text on
        # that side, otherwise whole only if the number of its last word is its count; before or above it: the
        # first number must be 1
        ux, uy = (-lines[0].ux, -lines[0].uy) if turned else (lines[0].ux, lines[0].uy)
        h = float(np.median([b.h for b in lines]))
        for nx, ny in det.cut_sides(g):
            if nx * ux + ny * uy > 0.5 or -nx * uy + ny * ux > 0.5:
                end = max(x * nx + y * ny for b in lines for x, y in b.quad())
                beyond = [b for b in det.words + det.numbers if (b in det.numbers or b.score > 0)
                          and b.center[0] * nx + b.center[1] * ny > end + BEYOND_TEXT * h
                          and on_card(det, b.center, g)]
                if beyond:
                    at_edge = True
                else:
                    last_number_needed = True
            else:
                first_number_needed = True
    orders = anchored["orders"] if anchored else \
        [list(range(count))] if in_rows else candidate_orders(columns, bands=len(linked) >= MIN_NUMBERS_READ)
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
    orders_found = len(orders)
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

    if last_number_needed and not at_edge:
        last = orders[0][-1] if len(orders) == 1 else None
        p = number(last) if last in linked else None
        at_edge = p is None or len(p) < count or p[count - 1] < 0.5

    def crop(k, variant):
        return rectified_crop(image, quads[variant][k], fill)

    kind = "list" if g < 0 else ("card24" if count > 12 else "card12")
    layout = WordLayout(kind, count, orders, len(CROP_VARIANTS), crop, lambda k: quads[0][k],
                        complete=not (at_edge or ambiguous))
    layout.doubtful = anchored["synthetic"] if anchored else set()
    layout.fingerprint = spare
    layout.fingerprint_hypothesis = hypothesis
    layout.not_found = WordsNotFound(f"{count + 1} lines found", at_edge=at_edge, line_count=count + 1) if hypothesis else None
    layout.lengths = [getattr(b, "ink", b.a1 - b.a0) / b.h for b in lines]
    layout.extra_words = extra
    layout.orders_found = orders_found
    layout.numbers_linked = len(linked)
    layout.numbers_read = len(read_numbers)
    layout.anchored = anchored is not None
    return layout
