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
      3. orientation: the page is read upright, as given (the photo is turned
         by its EXIF only; small slants and perspective are followed by the
         boxes);
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
import io
import itertools
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from . import native
from . import page
from .page import WordLayout, WordsNotFound
from .preprocess import box_mean, normalize_word, remove_long_runs

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
CARD_SHAPE = False            # (variant, off) a card kernel holds a document only if it has the shape of a card
                              # (Detection.shaped); on the benches it frees the words of SeedSigner cards cut by the
                              # frame (task-4434: 16 seeds lost on cadrages and devr)
CARD_MIN_PIXELS = 3000        # output pixels of a card kernel with the shape of a card (cards read on the photos of the
                              # site: 4 700 to 23 000; parts of a sheet taken for a card: 150 to 2 500)
CARD_MIN_COMPACTNESS = 0.55   # share of its oriented bounding rectangle covered by such a kernel (cards: 0.61 to 0.96;
                              # parts of a sheet: 0.31 to 0.56)
NUMBER_CHECK = True           # numbers read that contradict the order of the words make the list incoherent
                              # (find_words: layout.numbers_incoherent)
NUMBER_SURE = 0.9             # probability of the value of a number read taken into account by that check
BEYOND_TEXT = 0.3             # writing of a card farther than this many word heights beyond its text: card cut there
ZOOM_SHARE = 0.5              # documents covering less of the photo are detected again on their region (0.4 before
                              # the detector of 21/11/2026)
ZOOM_MARGIN = 0.15            # margin of that region, share of its size on each side
ZOOM_STRAY_CARDS = True       # (variant, off when False) a card kernel without the shape of a card (card_shaped) holding
                              # fewer than MIN_GROUP_WORDS words widens the region of the document. False (task-4497):
                              # such kernels left out, so that a small list on a landscape photo is detected again
                              # closer; on the frozen benches 21 exact seeds gained and 23 lost, 13 verified lost: kept
                              # as it was

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

# Word crops around the box, in box heights (horizontal, top, bottom): first reading, taller band, wider band
CROP_VARIANTS = ((0.6, 0.85, 0.75), (0.6, 1.1, 1.0), (1.2, 0.8, 0.7))
# Crops of a line given to the fingerprint judge, first one accepted: first reading, lower band
FINGERPRINT_CROPS = (CROP_VARIANTS[0], (0.6, 0.6, 0.5))
# Lines outside the words tried as the fingerprint: at least this long (text heights; 8 hexadecimal digits are
# longer), the longest ones first, at most this many
FINGERPRINT_MIN_LENGTH = 2.0
FINGERPRINT_MAX_LINES = 3
FINGERPRINT_TAIL = True       # the last piece of the first or last line of a column, far from the word, is tried as the
                              # fingerprint (« 12. GUARD 2cd9b02c », find_words)
TAIL_GAP = 1.5                # heights between the word and that piece, at least
FINGERPRINT_OPTIONAL_ANY = True  # with a valid count and no line passing the fingerprint thresholds, the one line outside
                              # the words (or tail) passing the loose margin is checked, the list kept if it matches no
                              # seed (layout.fingerprint_optional; before: rows across the columns only)
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

# Extent of a line along the text by the ink of its row (InkReach): the word map may cover only part of a word (the
# first or last letters lost: real photos of the site, 22/11/2026). Each line is extended on either side while strokes
# follow one another, up to the next box of its row, the next column or INK_REACH heights
INK_REACH = 6.0               # farthest a line is extended on either side, in heights
INK_GAP = 0.8                 # largest gap between two strokes of one word, in heights (GRID_WORD_GAP: between words)
INK_DIGIT_GAP = 1.8           # the same for a line tried as the fingerprint (digits written apart: « 0c5 23866 »)
INK_BAND = 0.3                # strokes are looked for this many heights on either side of the center line
INK_STRIP = 16                # height of a line in the strip analysed, in pixels (the photo is reduced to it)
INK_LEVEL = 0.5               # ink: darker than this share of the way from the background to the darkest strokes
INK_CONTRAST = 24             # gray levels between the background and the strokes of the line below which nothing is done
INK_RULE = 2.0                # horizontal runs longer than this many heights are ruled or printed lines, not strokes
INK_DARK = 0.95               # a column of the strip (two heights high) dark on this share of its rows is the edge of
                              # the sheet or a shadow, not a stroke: a line is never extended across it
INK_THICK = 0.15              # a stroke covers at least this many heights of the central band, across the text
INK_EDGE = 0.5                # columns closer than this many heights to a dark column are no strokes
INK_SHIFT = 0.35              # strokes whose mean height differs by more than this many heights from those of the line
                              # belong to another line
INK_ALIGN = 0.5               # a line of a column of MIN_COLUMN_LINES or more is not extended to start more than this many
                              # heights before the median start of the lines of its column
INK_MIN_LEAD = 1.0            # the same before the line
INK_MIN_GROWTH = 0.5           # a side of a line is extended only if it grows by this many heights or more (a letter)
BRIDGE_ROLES = False          # (variant) a row across the columns is left out only if it also reads as no BIP39 word, a
                              # label or a fingerprint (handwriting.line_role)
INK_MIN_WORD = 1.0            # ink at least this long (heights) in an empty numbered slot may be a word (anchored_slots)

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
        with np.load(io.BytesIO(native.read_weights(DETECTOR_PATH))) as data:
            net = native.detector({k: np.ascontiguousarray(data[k]) for k in data.files})
        work = np.zeros(-(-lib.hw_detector_work_size(ctypes.byref(net)) // 4), np.int32)
        _detector.append((lib, net, work))
    return _detector[0]



def network_maps(image: np.ndarray) -> np.ndarray:
    """
        Output maps (native.DETECTOR_OUTPUTS = 11, 192, 144) float32 of a
        grayscale image (384, 288) uint8. Raises ValueError for an image of
        another shape.
    """
    lib, net, work = load_detector()
    width, height = native.DETECTOR_SIZE
    image = np.ascontiguousarray(image, np.uint8)
    if image.shape != (height, width):
        raise ValueError(f"detector input of shape {image.shape}, {(height, width)} expected")
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



def card_shaped(pixels: np.ndarray) -> bool:
    """
        True if a card kernel component has the shape of a card: at least
        CARD_MIN_PIXELS output pixels covering at least CARD_MIN_COMPACTNESS
        of its bounding rectangle along its main axis.
    """
    if len(pixels) < CARD_MIN_PIXELS:
        return False
    a, _ = principal_angle(pixels)
    y, x = pixels[:, 0].astype(float), pixels[:, 1].astype(float)
    u, v = x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)
    return len(pixels) >= CARD_MIN_COMPACTNESS * (np.ptp(u) + 1) * (np.ptp(v) + 1)



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
        self.from_photo = None  # photo pixels to output pixels (set by detect)
        self.first = None  # detection on the whole photo when this one is on the document region (detections)
        words, numbers = sigmoid(maps[WORD]), sigmoid(maps[NUMBER])
        word_kernels = [p for p in components(maps[WORD] > 0) if len(p) >= MIN_KERNEL_PIXELS]
        number_kernels = [p for p in components(maps[NUMBER] > 0) if len(p) >= MIN_KERNEL_PIXELS]
        angle = text_angle(word_kernels)
        self.words = [Box(p, words, maps[HEIGHT], angle) for p in word_kernels]
        self.numbers = [Box(p, numbers, maps[HEIGHT], angle) for p in number_kernels]
        self.cards = [p for p in components(maps[CARD_KERNEL] > 0) if len(p) >= MIN_CARD_KERNEL]
        self.shaped = [card_shaped(p) for p in self.cards]
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
    det = Detection(network_maps(image),
                    lambda points: [((2 * u - ox) / s + x0, (2 * v - oy) / s + y0) for u, v in points], frame)
    det.from_photo = lambda points: [(((x - x0) * s + ox) / 2, ((y - y0) * s + oy) / 2) for x, y in points]
    return det



def first_pass_words(det: Detection) -> list[Box]:
    """
        Word boxes of the detection on the whole photo (`det.first`) moved to
        the output pixels of `det` (detection on the document region): both
        map the photo by a scale and an offset, so a box keeps its direction
        and its extents scale by the ratio of the two. Empty without a first
        detection.
    """
    first = det.first
    if first is None or det.from_photo is None:
        return []
    (x0, y0), (x1, y1) = det.from_photo(first.to_photo([(0.0, 0.0), (1.0, 0.0)]))
    k = math.hypot(x1 - x0, y1 - y0)
    moved = []
    for b in first.words:
        if b.score <= 0:
            continue
        m = copy_box(b)
        m.a0, m.a1, m.c, m.h = k * b.a0 + x0 * b.ux + y0 * b.uy, k * b.a1 + x0 * b.ux + y0 * b.uy, \
            k * b.c - x0 * b.uy + y0 * b.ux, k * b.h
        moved.append(m)
    return moved



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
        found (with ZOOM_STRAY_CARDS False, a card kernel too small or loose for
        a card holding fewer than MIN_GROUP_WORDS words left out), widened by ZOOM_MARGIN of its size on each side, if it covers less
        than ZOOM_SHARE of the area seen by the network: the words are then
        found again on it, at a higher resolution. None otherwise.
    """
    points = [p for b in det.words if b.score > 0 for p in b.quad()]
    cards = det.cards
    if not ZOOM_STRAY_CARDS:
        held = [sum(det.card_of(b.center) == i for b in det.words if b.score > 0) for i in range(len(cards))]
        cards = [c for i, c in enumerate(cards) if card_shaped(c) or held[i] >= MIN_GROUP_WORDS]
    points += [p for c in cards for p in ((c[:, 1].min(), c[:, 0].min()), (c[:, 1].max() + 1, c[:, 0].max() + 1))]
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



class InkReach:
    """
        Strokes along the lines of the photo: extent of a line by the ink of its
        row (reach), ink of an empty numbered slot (slot; `slots`: used by
        anchored_slots). The strip of a line
        is cut from the photo reduced so that a line is about INK_STRIP pixels
        high (cheap on a Pi Zero); ruled and printed lines are removed.
    """
    def __init__(self, image: Image.Image, det: Detection, fill: int, slots: bool = True):
        self.image, self.det, self.fill, self.slots = image, det, fill, slots
        self.reduced = {}


    def columns(self, box: Box, a0: float, a1: float, reference: tuple[float, float] | None) \
            -> tuple[np.ndarray, np.ndarray, float] | None:
        """
            (inked columns of the central band of `box` between a0 and a1 along
            the text, columns dark on most of the strip (INK_DARK), strip pixels
            per output pixel), or None if the strip has too little contrast.
            The strip is two heights high; the ink level is that of the part
            `reference` (a0, a1) of the strip, or of the whole strip.
        """
        strip = copy_box(box)
        strip.a0, strip.a1 = a0, a1
        quad = self.det.to_photo(strip.quad((0.0, 1.0, 1.0)))
        q = np.array(quad)
        height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        r = max(1, int(height / (2 * INK_STRIP)))
        if r not in self.reduced:
            self.reduced[r] = self.image.reduce(r) if r > 1 else self.image
        a = np.asarray(rectified_crop(self.reduced[r], [(x / r, y / r) for x, y in quad], self.fill), np.float32)
        rows, width = a.shape
        per = width / max(a1 - a0, 1e-6)
        top, bottom = int(rows * (0.5 - INK_BAND / 2)), int(math.ceil(rows * (0.5 + INK_BAND / 2)))
        # ink level of the strokes on the central band (a ruled or printed line below the words is darker)
        if reference is not None:
            inside = a[top:bottom, max(0, int((reference[0] - a0) * per)):max(1, int(math.ceil((reference[1] - a0) * per)))]
        else:
            inside = a[top:bottom]
        background = float(np.percentile(a, 90))
        ink = float(np.percentile(inside, 3)) if inside.size else background
        if background - ink < INK_CONTRAST:
            return None
        mask = a < background - INK_LEVEL * (background - ink)
        dark = mask.mean(0) >= INK_DARK
        mask = remove_long_runs(mask, max(3, int(INK_RULE * rows / 2)))
        # isolated pixels (grain of the paper) are no strokes
        mask &= box_mean(mask.astype(np.float32), 1) >= 2.5 / 9
        # a stroke crosses at least INK_THICK heights of the band (a ruled line slanted across the strip, cut into short
        # runs, does not); a column next to a dark one (the edge of the sheet) is no stroke either
        inked = mask[top:bottom].sum(0) >= max(2, INK_THICK * rows / 2)
        near = max(1, int(INK_EDGE * rows / 2))
        dark = np.convolve(dark, np.ones(2 * near + 1), "same") > 0
        # height of the strokes of each column (mean row of its ink, in heights from the center line)
        counts = mask.sum(0)
        middle = np.where(counts > 0, (mask * np.arange(rows)[:, None]).sum(0) / np.maximum(counts, 1), np.nan)
        self.middle = (middle - rows / 2) / (rows / 2)
        return inked & ~dark, dark, per


    def reach(self, box: Box, before: float, after: float, gap: float = INK_GAP) -> tuple[float, float]:
        """
            (a0, a1) of `box` extended while strokes follow one another (gaps of
            at most `gap` heights) on either side, not beyond `before` and
            `after` (along the text, output pixels), INK_REACH heights nor a
            dark column (INK_DARK).
        """
        h = box.h
        lo, hi = max(before, box.a0 - INK_REACH * h), min(after, box.a1 + INK_REACH * h)
        if lo >= box.a0 and hi <= box.a1:
            return box.a0, box.a1
        lo, hi = min(lo, box.a0), max(hi, box.a1)
        found = self.columns(box, lo, hi, (box.a0, box.a1))
        if found is None:
            return box.a0, box.a1
        inked, dark, per = found
        i0, i1 = int((box.a0 - lo) * per), min(len(inked) - 1, int((box.a1 - lo) * per))
        own = np.nonzero(inked[i0:i1 + 1])[0]
        if not len(own):
            return box.a0, box.a1
        # strokes of another line (slanted rows, close lines) lie higher or lower than those of the line
        level = float(np.nanmedian(self.middle[i0:i1 + 1][own]))
        inked = inked & (np.abs(np.nan_to_num(self.middle, nan=9.0) - level) <= INK_SHIFT)
        gap = gap * h * per
        first, last = i0 + own[0], i0 + own[-1]
        j = last + 1
        while j < len(inked) and j - last <= gap and not dark[j]:
            if inked[j]:
                last = j
            j += 1
        j = first - 1
        while j >= 0 and first - j <= gap and not dark[j]:
            if inked[j]:
                first = j
            j -= 1
        a0, a1 = max(lo, lo + first / per - 0.25 * h), min(hi, lo + (last + 1) / per + 0.25 * h)
        # a side grows only by a letter or more (the box already holds the margins of the word map)
        # before the line, at least INK_MIN_LEAD (a number missed by the detector is shorter than lost letters)
        return (a0 if a0 <= box.a0 - INK_MIN_LEAD * h else box.a0), (a1 if a1 >= box.a1 + INK_MIN_GROWTH * h else box.a1)


    def slot(self, box: Box) -> tuple[float, float] | None:
        """
            (a0, a1) of the longest group of strokes (gaps of at most INK_GAP
            heights) on the central band of an empty slot, if it is at least
            INK_MIN_WORD heights long, else None.
        """
        found = self.columns(box, box.a0, box.a1, None)
        if found is None:
            return None
        inked, _, per = found
        columns = np.nonzero(inked)[0]
        if not len(columns):
            return None
        groups = np.split(columns, np.nonzero(np.diff(columns) > INK_GAP * box.h * per)[0] + 1)
        best = max(groups, key=lambda g: g[-1] - g[0])
        if (best[-1] + 1 - best[0]) / per < INK_MIN_WORD * box.h:
            return None
        return box.a0 + best[0] / per - 0.25 * box.h, box.a0 + (best[-1] + 1) / per + 0.25 * box.h



def ink_extend(reach: "InkReach", columns: list[list[Box]], others: list[Box], skip: set = frozenset(),
               gap: float = INK_GAP, on_line=None):
    """
        Extends every line of `columns` (in place) by the ink of its row
        (InkReach.reach), up to the nearest box of `others` on its row (word,
        number or line) and to the start of the next column. `skip`: indices
        of lines (in reading of the columns one after the other) not extended;
        `gap`: see InkReach.reach; `on_line(done, total)` is called before each
        line (display only).
    """
    starts = [min(b.a0 for b in c) for c in columns]
    total = sum(len(c) for c in columns)
    k = 0
    for ci, column in enumerate(columns):
        following = min((s for s in starts if s > starts[ci]), default=math.inf)
        found = []
        for b in column:
            if on_line is not None:
                on_line(k + 1, total)
            found.append(ink_line(reach, b, others, following, gap) if k not in skip else (b.a0, b.a1))
            k += 1
        if len(column) >= MIN_COLUMN_LINES:
            # the words of a column start together: ink further before the start of most of them is a number missed
            # by the detector or a margin, not letters
            start = float(np.median([a0 for a0, _ in found])) - INK_ALIGN * float(np.median([b.h for b in column]))
            found = [(max(a0, min(b.a0, start)), a1) for b, (a0, a1) in zip(column, found)]
        for b, (a0, a1) in zip(column, found):
            if hasattr(b, "ink"):
                b.ink += (b.a0 - a0) + (a1 - b.a1)
            b.a0, b.a1 = a0, a1



def ink_line(reach: "InkReach", b: Box, others: list[Box], following: float = math.inf, gap: float = INK_GAP) \
        -> tuple[float, float]:
    """
        (a0, a1) of line `b` extended by the ink of its row (InkReach.reach) up
        to the nearest box of `others` on its row and to `following` (start of
        the next column, along the text).
    """
    # any box reaching the height of the line stops it (the words of a slanted line next to it)
    row = [o for o in others if o is not b and abs(o.c - b.c) < 0.5 * (o.h + b.h)
           and (o.a1 <= b.a0 + 0.5 * b.h or o.a0 >= b.a1 - 0.5 * b.h)]
    before = max([o.a1 for o in row if o.middle < b.a0] + [-math.inf])
    after = min([o.a0 for o in row if o.middle > b.a1] + [following - 0.5 * b.h])
    return reach.reach(b, before, after, gap)

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


def bridged_columns(boxes: list[Box]) -> tuple[list[list[Box]], list[list[Box]]] | None:
    """
        Columns of a list whose corridor between two columns is covered only
        by one or two rows at its start or end (a title or a fingerprint
        written across the columns: group_lines then finds a single column):
        the columns of the other boxes (group_lines, or grid_columns) if there
        are at least two of them, of a valid number of words, and every row
        left out overlaps a corridor between them, the rows left out together
        closing one of them (no gap of COLUMN_GAP heights left in it: the
        columns on either side are merged). The fewest rows left out first (the first, the
        last, then two): (those columns, the rows left out); None if no such
        set of rows is found.
    """
    if len(boxes) < MIN_GROUP_WORDS + 1:
        return None
    h = float(np.median([b.h for b in boxes]))
    rows = []
    for b in sorted(boxes, key=lambda b: b.c):
        if rows and abs(b.c - np.mean([x.c for x in rows[-1]])) < SAME_LINE * max(h, b.h):
            rows[-1].append(b)
        else:
            rows.append([b])
    n = len(rows)
    for removed in ([0], [n - 1], [0, 1], [n - 2, n - 1], [0, n - 1]):
        if n - len(set(removed)) < 2 or len(set(removed)) < len(removed):
            continue
        body = [b for i, row in enumerate(rows) if i not in removed for b in row]
        columns = group_lines([copy_box(b) for b in body])
        if len(columns) < 2 or sum(len(c) for c in columns) not in page.SEED_LENGTHS:
            # columns too close for a corridor: a grid of words written in rows
            columns = grid_columns(body)
            if columns is None or sum(len(c) for c in columns) not in page.SEED_LENGTHS:
                continue
        corridors = [(max(b.a1 for b in columns[i]), min(b.a0 for b in columns[i + 1])) for i in range(len(columns) - 1)]
        if any(c1 <= c0 for c0, c1 in corridors):
            continue

        def closes(spans, c0, c1):
            # the spans leave no gap of COLUMN_GAP heights or more inside the corridor (c0, c1): group_lines merges
            # the columns
            edge, widest = c0, 0.0
            for a0, a1 in sorted(spans):
                if a1 <= c0 or a0 >= c1:
                    continue
                widest, edge = max(widest, a0 - edge), max(edge, a1)
            return max(widest, c1 - edge) < COLUMN_GAP * h

        out = [b for i in removed for b in rows[i]]
        if all(any(b.a0 < c1 and b.a1 > c0 for b in rows[i] for c0, c1 in corridors) for i in removed) \
                and any(closes([(b.a0, b.a1) for b in out], c0, c1) for c0, c1 in corridors):
            return columns, [rows[i] for i in removed]
    return None



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



def extra_line_sets(boxes: list[Box]) -> list[tuple[list[tuple[Box, list[Box]]], list[list[Box]]]]:
    """
        Sets of one or two rows at the start or the end of `boxes` (the first,
        the last, the first two, the last two, the first and the last; rows as
        in extra_rows, of at most MAX_EXTRA_ROW_PIECES pieces) whose removal
        leaves columns of a valid number of words (word_columns): lines
        written with the words that may not be words (a fingerprint, a label
        such as "Fingerprint" or "Seed words", a second fingerprint). For
        each: ([(the row as one line, its boxes along the text)], those
        columns).
    """
    if len(boxes) < MIN_GROUP_WORDS + 2:
        return []
    h = float(np.median([b.h for b in boxes]))
    rows = []
    for b in sorted(boxes, key=lambda b: b.c):
        if rows and abs(b.c - np.mean([x.c for x in rows[-1]])) < SAME_LINE * max(h, b.h):
            rows[-1].append(b)
        else:
            rows.append([b])
    n = len(rows)
    found = []
    for removed in ([0], [n - 1], [0, 1], [n - 2, n - 1], [0, n - 1]):
        if n - len(set(removed)) < 2 or len(set(removed)) < len(removed):
            continue
        out = []
        for i in removed:
            row = sorted(rows[i], key=lambda b: b.a0)
            if 1 + sum(b.a0 - a.a1 > GRID_WORD_GAP * h for a, b in zip(row, row[1:])) > MAX_EXTRA_ROW_PIECES:
                break
            line = copy_box(row[0])
            for b in row[1:]:
                line.merge(copy_box(b))
            out.append((line, row))
        else:
            gone = [b for i in removed for b in rows[i]]
            columns = word_columns([b for b in boxes if not any(b is x for x in gone)])
            if columns is not None:
                found.append((out, columns))
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



def link_numbers(det: Detection, lines: list[Box], numbers: list[Box]) -> dict[int, Box]:
    """
        {line index: number}: each number linked to the word line that follows
        it on its line (or precedes it), by the link map, or
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
            slots.append([box, filled, n, not found])
        grid.append(slots)
    flat = [s for column in grid for s in column]
    count = sum(s[1] for s in flat)
    reach = getattr(det, "ink_reach", None)
    if count not in page.SEED_LENGTHS and reach is not None and reach.slots:
        # words too faint for the word map: an empty slot holding strokes (InkReach.slot) on the line of the words is
        # taken as written, its extent that of the strokes, if the count then becomes valid
        heights = [s[0].h for s in flat if not s[3]]
        h = float(np.median(heights)) if heights else 1.2 * float(np.median([s[2].h for s in flat]))
        inked = []
        for s in flat:
            if s[1]:
                continue
            box, n = copy_box(s[0]), s[2]
            box.h = h
            box.c = n.c + drift[0] * (box.middle - n.middle) + drift[1]
            found = reach.slot(box)
            if found is not None:
                box.a0, box.a1 = found
                inked.append((s, box))
        if inked and count + len(inked) in page.SEED_LENGTHS:
            for s, box in inked:
                s[0], s[1] = box, True
            count += len(inked)
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



def candidate_orders(columns: list[list[Box]]) -> list[list[int]]:
    """
        Every order the layout makes plausible: column by column, then row by
        row when there are several columns (word indices numbered column by
        column), then column by column within bands (band_order) when the page
        is split into bands. The line numbers, when read, choose among them;
        otherwise two orders giving a plausible seed leave the order ambiguous
        (read_layout).
    """
    count = sum(len(c) for c in columns)
    orders = [list(range(count))]
    if len(columns) > 1:
        starts = np.cumsum([0] + [len(c) for c in columns])
        rows = max(len(c) for c in columns)
        by_rows = [int(starts[k] + r) for r in range(rows) for k in range(len(columns)) if r < len(columns[k])]
        if by_rows != orders[0]:
            orders.append(by_rows)
        banded = band_order(columns)
        if banded is not None and banded not in orders:
            orders.append(banded)
    return orders



def numbers_contradict(orders: list[list[int]], values: dict[int, int]) -> bool:
    """
        True if the numbers read {line: value} contradict every order of the
        lines: in the order that fits them best, at least two of them, and at
        least half, differ from the position of their line (1 for the first).
    """
    if not values:
        return False
    wrong = min(sum(order.index(k) + 1 != v for k, v in values.items()) for order in orders)
    return wrong >= max(2, len(values) / 2)



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



# ---------- reading ----------

_detections = {}


def stage(step, *args):
    """
        Reports a stage of the reading to `step` if it takes them
        (handwriting.Progress.stage), for display only.
    """
    report = getattr(step, "stage", None)
    if report is not None:
        report(*args)


def detections(image: Image.Image, step) -> Detection:
    """
        Detection of the document on `image` (see find_words), kept for a
        second search on the same photo (the last photo only).
    """
    key = (id(image), image.size)
    if key not in _detections:
        _detections.clear()
        stage(step, "find")
        det = detect(image)
        step(DETECTOR_COST)
        region = document_region(det, image.size)
        if region:
            stage(step, "closer")
            first, det = det, detect(image, region)
            det.first = first
            step(DETECTOR_COST)
        _detections[key] = det
    return _detections[key]



def find_words(image: Image.Image, step=page.no_step, fingerprint=None, crossed: bool = False,
               extra: bool = False, merged: bool = False) -> WordLayout:
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
        With `extra`, the list is read without one or two rows at its start or
        end that read as a fingerprint, a label or no word (see
        extra_line_sets and the second part below); WordsNotFound if no such
        set of rows is found. `layout.extra_lines` (without `extra`): a list
        numbered on most of its lines, with a valid count, every word box on
        one of its lines, some of whose lines have no number while such a set
        of rows exists: worth reading with `extra` (handwriting.read_seed).
        `layout.numbers_incoherent`: numbers read and linked to their line that
        contradict every order of the words left (see NUMBER_CHECK). With
        `merged`, the card kernels are ignored: every word is on one list (a
        list whose numbers contradict its order, read again with the words of
        another document).
    """
    det = detections(image, step)
    stage(step, "lines")
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
    fill = int(np.median(np.asarray(image.reduce(16))))
    # the second search leaves out words scribbled over: their slots are never filled again by their ink
    det.ink_reach = InkReach(image, det, fill, slots=not crossed)
    groups, number_groups = {}, {}
    for b in det.words:
        if b.score > 0:
            groups.setdefault(det.card_of(b.center), []).append(b)
    stray = [g for g, boxes in groups.items() if g >= 0 and (len(boxes) < MIN_GROUP_WORDS or merged
                                                              or (CARD_SHAPE and not det.shaped[g]))]
    for g in stray:
        # a card kernel holding fewer words, or without the shape of a card (a part of a sheet), is no document: its
        # words and numbers are free
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
        # empty slots filled by their ink only when the lines alone do not make a valid count
        det.ink_reach.slots = not crossed and count not in page.SEED_LENGTHS
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
    bridged_rows = []
    if anchored is None and not crossed:
        # a title or a fingerprint written across the columns: the list read without it (its boxes are then outside
        # the words, tried as the fingerprint below), if the count is wrong or if it splits the list into more columns
        # (a count made valid by a column merged into one: the title counted, two words joined into one line); not in the
        # second search, whose words scribbled over are already left out
        bridged = bridged_columns(groups[g])
        if bridged is not None and BRIDGE_ROLES and getattr(fingerprint, "role", None) is not None:
            # lexical signal too (variant): every row left out reads as no word, a label or a fingerprint
            fill_ = int(np.median(np.asarray(image.reduce(16))))
            for row in bridged[1]:
                line = copy_box(row[0])
                for b in row[1:]:
                    line.merge(copy_box(b))
                crops = [rectified_crop(image, det.to_photo(line.quad(m)), fill_) for m in FINGERPRINT_CROPS]
                if fingerprint.role(crops)[0] is None:
                    bridged = None
                    break
        if bridged is not None and (count not in page.SEED_LENGTHS or len(bridged[0]) > len(columns)):
            columns, count = bridged[0], sum(len(c) for c in bridged[0])
            bridged_rows = bridged[1]
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
    # every line extended by the ink of its row, up to the boxes around it (words found only by the ink of an empty
    # numbered slot are left as found)
    ink_extend(det.ink_reach, columns, [b for b in det.words if b.score > 0] + list(det.numbers)
               + [b for c in columns for b in c], anchored["synthetic"] if anchored else frozenset(),
               on_line=lambda done, total: stage(step, "lines", done, total))
    spare = spare_first = None
    hypothesis = False
    others, removed_lines, strong = [], [], False
    if extra:
        # rows at the start or the end of a list that may not be words: each is a fingerprint (strict thresholds,
        # or the loose margin of the fingerprint network), a label, or (with a fingerprint only) a line read as no
        # word; exactly one set of rows must pass. A set with a loose fingerprint or a line read as no word is a
        # hypothesis, kept by read_seed only if a fingerprint matches a seed read.
        roles_of = getattr(fingerprint, "role", None)
        if g >= 0 or anchored is not None or roles_of is None:
            raise WordsNotFound("no extra lines")
        numbers_g = number_groups.get(g, [])
        cache, passed = {}, []
        for rows, rest in extra_line_sets(groups[g]):
            roles = []
            for line, pieces in rows:
                key = tuple(id(b) for b in pieces)
                if key not in cache:
                    # the row (widened to the numbers at its ends) on both fingerprint crops, then its last piece
                    # (a fingerprint after a label on the same row): three crops at most, read on a Pi Zero
                    wide = with_number(line, numbers_g, after=True) or line
                    boxes = [(wide, m) for m in FINGERPRINT_CROPS] + [(pieces[-1], FINGERPRINT_CROPS[0])] * (len(pieces) > 1)
                    crops = [rectified_crop(image, det.to_photo(box.quad(margins)), fill) for box, margins in boxes]
                    cache[key] = roles_of(crops)
                roles.append(cache[key])
            kinds = [r[0] for r in roles]
            fingerprints = [r[1] for r in roles if r[0] in ("fingerprint", "loose")]
            if None in kinds or not (fingerprints or all(k == "label" for k in kinds)):
                continue
            passed.append((sum(k == "fingerprint" for k in kinds), -sum(k == "other" for k in kinds), rows, rest,
                           fingerprints, any(k in ("loose", "other") for k in kinds),
                           all(k in ("fingerprint", "label") for k in kinds)))
        passed.sort(key=lambda p: p[:2], reverse=True)
        if not passed or (len(passed) > 1 and passed[0][:2] == passed[1][:2]):
            raise WordsNotFound("no extra lines")
        _, _, rows, columns, fingerprints, hypothesis, strong = passed[0]
        count = sum(len(c) for c in columns)
        removed_lines = [line for line, _ in rows]
        if fingerprints:
            spare, others = fingerprints[0], fingerprints[1:]
            spare_first = removed_lines[0]
    elif g < 0 and anchored is None and count not in page.SEED_LENGTHS and fingerprint is not None:
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
        around = [b for b in det.words if b.score > 0] + list(det.numbers) + [b for c in columns for b in c]
        for t, (line, pieces, rest) in enumerate(tried):
            # digits partly missed by the word map: the line extended by the ink of its row, digits written apart
            line = copy_box(line)
            line.a0, line.a1 = ink_line(det.ink_reach, line, around, gap=INK_DIGIT_GAP)
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
    if spare is None and fingerprint is not None and not extra:
        # a line of the document outside the words (above, below or beside them, numbered or not, with a list
        # counted by its numbers too) is tried as the fingerprint before it is taken for a word left out
        left = [copy_box(b) for b in det.words if b.score > 0 and det.card_of(b.center) in (g, -1)
                and not any(covers(line, b.center) for line in lines)]
        # boxes of the detection on the whole photo that the detection on the document region missed (a line of
        # digits seen only at the lower resolution), for the fingerprint only
        left += [b for b in first_pass_words(det) if det.card_of(b.center) in (g, -1)
                 and not any(covers(line, b.center) for line in lines)
                 and not any(covers(w, b.center) or covers(b, w.center) for w in det.words)]
        h = float(np.median([b.h for b in lines]))
        outside = group_lines(left)
        # a line of digits partly missed by the word map (its first or last digits): extended by the ink of its row
        ink_extend(det.ink_reach, outside, [b for b in det.words if b.score > 0] + list(det.numbers) + lines,
                   gap=INK_DIGIT_GAP, on_line=lambda done, total: stage(step, "lines", done, total))
        tried = sorted((line for column in outside for line in column
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
        # a fingerprint written on the row of the first or last word of a column, after it (« 12. GUARD 2cd9b02c »):
        # the last piece of that line, more than TAIL_GAP heights after the piece before it, is tried as the fingerprint;
        # taken, it is cut off the line (tails: (line, piece, end of the word))
        tails = []
        if FINGERPRINT_TAIL and not accepted:
            ends = {id(b): b for column in columns for b in (column[0], column[-1])}
            for line in ends.values():
                pieces = sorted((b for b in groups[g] if covers(line, b.center)), key=lambda b: b.a0)
                if len(pieces) < 2:
                    continue
                last, before = pieces[-1], pieces[-2]
                if last.a0 - before.a1 < TAIL_GAP * h or last.a1 - last.a0 < FINGERPRINT_MIN_LENGTH * h:
                    continue
                tail = copy_box(last)
                tail.a1 = max(tail.a1, line.a1)
                tails.append((line, tail, before.a1))
            for line, tail, _ in tails:
                for margins in FINGERPRINT_CROPS:
                    crop = rectified_crop(image, det.to_photo(tail.quad(margins)), fill)
                    if fingerprint(crop):
                        accepted.append((tail, crop))
                        break
        margin_of = getattr(fingerprint, "margin", None)
        optional = False
        loose_line = None
        if not accepted and margin_of is not None and (bridged_rows or FINGERPRINT_OPTIONAL_ANY):
            # a row left out across the columns (bridged_columns), or else (FINGERPRINT_OPTIONAL_ANY) any line outside
            # the words or tail, read best as 8 hexadecimal digits by the fingerprint network: checked as the
            # fingerprint, the list kept as read if it matches no seed (layout.fingerprint_optional); exactly one must
            # pass the loose margin (one check per search, as on the other paths)
            passing = []
            if bridged_rows:
                for line in tried:
                    if any(covers(line, b.center) for row in bridged_rows for b in row):
                        wide = with_number(line, det.numbers, after=True)
                        for box, margins in itertools.product([wide] * (wide is not None) + [line], FINGERPRINT_CROPS):
                            crop = rectified_crop(image, det.to_photo(box.quad(margins)), fill)
                            if margin_of(crop) is not None:
                                passing.append((line, crop))
                                break
            else:
                # one crop of one line, at the cost of one pass of the fingerprint network: the tail if there is one,
                # else the longest line outside the words
                line = tails[0][1] if tails else (tried[0] if tried else None)
                if line is not None:
                    crop = rectified_crop(image, det.to_photo(line.quad(FINGERPRINT_CROPS[0])), fill)
                    if margin_of(crop) is not None:
                        passing.append((line, crop))
                        loose_line = line
            if len(passing) == 1:
                accepted, optional = passing, True
        if len(accepted) == 1:
            spare_line, spare = accepted[0]
            for line, tail, end in tails:
                if tail is spare_line:
                    # the word alone on its line, a quarter of a height after its last piece
                    line.a1 = min(line.a1, end + 0.25 * line.h)
                    if hasattr(line, "ink"):
                        line.ink = min(line.ink, line.a1 - line.a0)
    else:
        optional = False
        loose_line = None
    # a line checked as an optional fingerprint outside the rows across the columns stays a word left out until its
    # fingerprint matches (the guard against a word left out of the list then applies)
    kept_out = [spare_line] if spare_line is not None and spare_line is not loose_line else []
    words_left = [(b.score, (b.a1 - b.a0) / b.h, det.card_of(b.center) != g) for b in det.words
             if not any(covers(line, b.center) for line in lines + kept_out + removed_lines)]
    numbers = number_groups.get(g, [])
    linked = anchored["numbers"] if anchored else link_numbers(det, lines, numbers)
    quads = [[det.to_photo(b.quad(m)) for b in lines] for m in CROP_VARIANTS]

    first_number_needed = last_number_needed = False
    if g >= 0 and det.cut_sides(g):
        # card reaching the frame after its text or below it: cut if writing of the card is seen beyond its text on
        # that side, otherwise whole only if the number of its last word is its count; before or above it: the
        # first number must be 1
        ux, uy = lines[0].ux, lines[0].uy
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
        [list(range(count))] if in_rows else candidate_orders(columns)

    read_numbers = {}

    def number(k):
        if k not in read_numbers:
            q = det.to_photo(linked[k].quad(NUMBER_CROP_MARGINS))
            step(NUMBER_COST)
            stage(step, "numbers", len(read_numbers) + 1, len(linked))
            read_numbers[k] = number_probabilities(image, q, fill)
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

    incoherent, checked = False, 0
    if NUMBER_CHECK and read_numbers:
        # numbers read, sure of their value and linked to their line by the link map, that contradict every order left
        # (a list missing its first words, a fingerprint whose digits were taken for numbers): the list is incoherent
        sure = {}
        for k, p in read_numbers.items():
            if p is None:
                continue
            v = int(np.argmax(p))
            b, n = lines[k], linked[k]
            if p[v] >= NUMBER_SURE and det.link_score(n.center, b.start if n.middle < b.a0 else b.center) >= LINK_THRESHOLD:
                sure[k] = v + 1
        checked = len(sure)
        incoherent = numbers_contradict(orders, sure)

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
    layout.fingerprint_others = others
    layout.fingerprint_hypothesis = hypothesis
    layout.fingerprint_optional = optional
    layout.fingerprint_single = optional and loose_line is not None
    # The second reading without label or fingerprint rows (read_seed, `extra`) is offered only when every word box is
    # on a line of the list: a word left out (a title, words of another document) closes it. This was the behaviour
    # since 986e493, through a variable `extra` reused for the words left out; written out here. Offering it also when
    # words are left out (task-4434, C6) gained no seed on the frozen benches and cost up to 88 s more on a Pi Zero
    # (a photo already read, read again for nothing).
    layout.extra_lines = not words_left and g < 0 and anchored is None and spare is None and fingerprint is not None \
        and len(linked) >= MIN_ANCHOR_NUMBERS and len(linked) < count and bool(extra_line_sets(groups[g]))
    layout.extra_lines_removed = len(removed_lines)
    layout.extra_lines_strong = strong
    found_lines = count + max(1, len(removed_lines))
    layout.not_found = WordsNotFound(f"{found_lines} lines found", at_edge=at_edge, line_count=found_lines) \
        if hypothesis else None
    layout.lengths = [getattr(b, "ink", b.a1 - b.a0) / b.h for b in lines]
    layout.extra_words = words_left
    layout.orders_found = orders_found
    layout.numbers_linked = len(linked)
    layout.numbers_read = len(read_numbers)
    layout.anchored = anchored is not None
    layout.numbers_incoherent = incoherent
    layout.numbers_checked = checked
    layout.merged = merged
    # words held by a card kernel (in this document or another one): reading them as one list may change something
    layout.cards_held = any(k >= 0 for k in groups)
    return layout
