"""
    Reading of a seed handwritten on paper, from a photo: the official
    SeedSigner 12- or 24-word card, another printed card, or a free list of 12
    to 24 words (any paper, one or more columns, with or without numbers).

        reading = read_seed(image, progress, on_word)
        analysis = reading.analyze(confirmed)
        trust = reading.trust(confirmed)

    Each word is found and cut out of the photo by the word detector of
    Haller-1 (detector.py), normalized (preprocess.py),
    read by a small neural network (word_model.py) and scored against the 2 048
    BIP39 words (decoder.py); the checksum then chooses the seed among the
    plausible readings and flags uncertain words (checksum.py). The network is
    Haller-1, made of two sets of weights: a seed that is not valid with the
    first one is read again by the second one, trained with more lowercase and
    cursive writing. The detector finds words, line numbers and cards with no
    rule about the kind of document; the reading order comes from the numbers
    when they tell it, otherwise from the geometry of the lines and columns.
    The result must always be reviewed by the user against the paper.

    A seed is only verified when the master key fingerprint written with the
    words is read and matches them (see check_fingerprint). Otherwise, however
    sure each word is, the reading is never called valid, sure or verified:
    its trust level (see Trust) only says how often such readings are fully
    right.
"""
import math
import threading

import numpy as np
from PIL import Image

from . import detector, fingerprint
from .checksum import Analysis, MIN_PLAUSIBLE_SEED, MIN_SEED_PROBABILITY
from .decoder import number_probabilities
from .page import WordsNotFound
from .preprocess import normalize_word, unread_ink
from .word_model import MODEL_PATH

# Readings kept per word (the following ones are negligible)
CANDIDATES = 30

# While the seed is not valid, at most MAX_REREADS words, least probable
# first, are read again on their other crop variants; both readings are then
# combined, a word missing from one of them getting FUSION_FLOOR there
MAX_REREADS = 4
FUSION_FLOOR = math.log(1e-4)

STATUS_RANK = {"invalid": 0, "uncertain": 1, "valid": 2}

# Second network, used when the first one gives no valid seed. Its seeds need
# a higher probability to be declared valid: it reads the photos the first one
# failed on, where wrong readings that pass the checksum are more frequent.
LOWERCASE_MODEL_PATH = MODEL_PATH.replace("haller-1-capitals.npz", "haller-1-lowercase.npz")

# Third network, of the same architecture, trained from the first one on
# synthetic lines of which half are master key fingerprints (8 hexadecimal
# digits, see fingerprint.py): it reads the fingerprint line that is checked
# (check_fingerprint), and is loaded the first time one is (1 to 2 seconds on
# a Pi Zero). The lines that may be a fingerprint are still told apart by the
# first two networks (FINGERPRINT_MARGIN).
FINGERPRINT_MODEL_PATH = MODEL_PATH.replace("haller-1-capitals.npz", "haller-1-fingerprint.npz")
LOWERCASE_MIN_SEED_PROBABILITY = 0.85

# The same photos are also read with a blend of both networks: weighted sum of
# their word log-probabilities, BLEND_WEIGHT for the first one. It reuses the
# scores already computed. Scores of words dropped by the decoder beam are
# raised to MIN_WORD_SCORE first, so that the weighted sum stays finite.
BLEND_WEIGHT = 0.65
BLEND_MIN_SEED_PROBABILITY = 0.5
MIN_WORD_SCORE = -1e4

FIRST, LOWERCASE, BLEND = (1.0, 0.0), (0.0, 1.0), (BLEND_WEIGHT, 1 - BLEND_WEIGHT)

# Word scores of each network, and of the blend, are divided by a temperature
# so that they can be used as calibrated log-likelihoods by checksum.Analysis.
# Chosen on 640 simulated pages kept apart from the test photos (EMNIST test
# capitals, two lowercase fonts never used elsewhere): the lowest temperature
# with no wrong seed declared valid; the probability of the first word then
# matches its accuracy within 1 % on capitals (2.6 would fit capitals best,
# 1.7 cards and cursive).
TEMPERATURES = {FIRST: 2.2, LOWERCASE: 2.2, BLEND: 2.2}

# Line numbers written before the words of a list in two columns fix the
# reading order: a number read with a probability of at least
# NUMBER_PROBABILITY is a vote for the orders in which its word has that
# position (words at the same position in every order do not vote). An order
# with at least MIN_NUMBER_VOTES votes, the others having none, is the order
# written. Read only when the order matters (more than one order gives a seed
# that passes the checksum).
NUMBER_PROBABILITY = 0.9
MIN_NUMBER_VOTES = 3

# A list with one line more than a valid number of words may carry the master
# key fingerprint written under (or above) the words: its first line, or the
# last line of a column, is left out of the words if it reads as 8
# hexadecimal digits (see fingerprint.margins), the mean over both networks
# of its log-likelihood exceeding that of its best BIP39 word by at least
# FINGERPRINT_MARGIN, and falling short of the free reading of the network
# by at most -FINGERPRINT_FIT. Chosen on simulated pages kept apart from the
# test photos (DEV2, lists with a fingerprint): 72 % of 660 fingerprints left
# out; checked through the whole reading on every bench, none of 1 208 word
# lines within 3 of both thresholds.
FINGERPRINT_MARGIN = 17.0
FINGERPRINT_FIT = -21.0
# A line whose mean margin by both networks is at least FINGERPRINT_RESCUE_MARGIN without passing both thresholds is
# judged by the fingerprint network alone, with the same thresholds (real photos of the Pi, 21/11/2026: a lowercase
# fingerprint at a margin of 19 and a fit of -22.6 by the word networks, 27.7 and +0.6 by the fingerprint network;
# the word lines of the same photos at -20 or below for all three). The fingerprint network is loaded only then.
# Lowered from 12 to 10 with the detector of 21/11/2026 (tuning benches: 4 more exact seeds, no wrong seed verified).
FINGERPRINT_RESCUE_MARGIN = 10.0
# A list with one line too many none of which passes these thresholds: the one line read by the fingerprint network
# with a margin of at least LOOSE_FINGERPRINT_MARGIN (and a fit of at least FINGERPRINT_FIT) is left out as a
# fingerprint hypothesis, kept only if the fingerprint matches one of the seeds read (verified); otherwise the list
# is not found, as without it. Real photos of the Pi, 21/11/2026 (grid of 3 columns numbered by rows, fingerprint
# above, its 9 written with a tail and read as a g by the word networks): fingerprint line at +5, word lines of the
# same photos at -26 or below.
LOOSE_FINGERPRINT_MARGIN = 0.0
# Rows at the start or the end of a list that may not be words (detector.extra_line_sets), read when the list is not
# found or is numbered on most of its lines but not all (read_seed): a row is a label if its best label
# (fingerprint.LABELS) reads better than its best BIP39 word by LABEL_MARGIN and falls short of the free reading of
# the networks by at most -LABEL_FIT (means of both networks); it is read as no word if its best BIP39 word falls
# short of the free reading by more than -NO_WORD_FIT (taken only with a fingerprint, the set of rows being then a
# hypothesis kept only if verified). Chosen on the tuning pages only (real photos DEV, simulated lists with a
# fingerprint fpdev, 200 simulated pages with a label kept apart from the tests; 21/11/2026): the right words read
# (8 428, 776 of them real) reach a label margin of -4.9 at most (-13.5 on the real photos), the rows taken for labels
# +9 or more for 95 % of them; the cursive "fingerprint" of a real photo is at +2.2.
LABEL_MARGIN = 0.0
LABEL_FIT = -15.0
NO_WORD_FIT = -25.0

# The fingerprint line is checked against the FINGERPRINT_CANDIDATES most
# probable valid seeds of the readings (every reading order, network and
# layout), in order of rank: the first one whose fingerprint matches (see
# fingerprint.MIN_MATCH_PROBABILITY) is the seed written, every word certain.
# When none matches, the seed is never called wrong (a word or a digit may be
# misread): every word is uncertain. With 3 candidates and the threshold of
# fingerprint.py, a wrong seed is verified by chance with a probability of at
# most 1e-6 per photo (the right seed was among the first 3 on all but 6 of
# 1 698 pages with a fingerprint read, 21/11/2026). FINGERPRINT_COST: computation of one fingerprint on a Pi
# Zero (PBKDF2-HMAC-SHA512, 2048 rounds, 0.062 s; master key and HASH160,
# 0.005 s; measured with embit on 20/11/2026) and its check, seconds.
FINGERPRINT_CANDIDATES = 3
FINGERPRINT_COST = 0.1
# The fingerprint line is read three times: as cut, and without FINGERPRINT_TRIM of its height at the top, then at
# the bottom; the probability of a fingerprint is the mean of the three (FingerprintCheck), which still sums to 1 over
# the strings of 8 digits, so the bound of fingerprint.MIN_MATCH_PROBABILITY is unchanged. A single cut is too
# sensitive to its edges: on a real photo of the Pi (21/11/2026, nines written with a tail), the probability of the
# right fingerprint went from 9e-6 to 3e-2 when the cut moved by 2 % of its height.
FINGERPRINT_TRIM = 0.02

# Guards of a valid seed (see guard), chosen on simulated pages kept apart
# from the tests and on the development half of the real photos:
# * its probability is at least MIN_DETECTOR_SEED_PROBABILITY;
# * no handwritten word found by the detector with a box score of at least
#   LEFT_OUT_SCORE and a length of at least LEFT_OUT_LENGTH heights was left
#   out of the reading (another document, a word crossed out and rewritten);
# * a word whose box is at least STRETCH times longer per letter of the word
#   read than the median of the page is uncertain: part of it was probably
#   not read (see stretched_words);
# * a word with faint ink over UNREAD_INK heights of text beyond the part read
#   is uncertain (faded or broken letters, see preprocess.unread_ink).
MIN_DETECTOR_SEED_PROBABILITY = 0.9
LEFT_OUT_SCORE = 0.8
LEFT_OUT_LENGTH = 1.0
STRETCH = 1.7
UNREAD_INK = 1.8

# Expected duration of the steps of a reading, in seconds on a Pi Zero (see
# Progress): loading of the models, reading of a word by the first network
# (crop, normalization, network, decoder), by the second network or the blend
# (the normalized word is reused), re-reading of a card word on its other crops.
# The second reading is counted for half its cost before it is known to be needed.
LOAD_COST = 0.8
WORD_COST = 0.2
OTHER_WORD_COST = 0.06
REREAD_COST = 0.4

# Trust levels of a reading (see Trust). TRUST_VERIFIED is reserved for a seed
# verified by its fingerprint. A fingerprint read that matches none of the
# seeds read is evidence against them: TRUST_LOW. Without one: TRUST_HIGH when the checksum
# analysis leaves no word uncertain, TRUST_MEDIUM when the seed is valid with
# a probability of at least MIN_MEDIUM_SEED_PROBABILITY, TRUST_LOW otherwise.
# TRUST_RATES: lowest share of fully right seeds each level stands for. Chosen
# on the simulated tuning pages only (DEV2, SERRE_DEV and lists with a
# fingerprint, 2 757 pages, none from the test benches): the Wilson lower
# bound (95 %) of the measured share is above the rate announced, 94.7 % for
# TRUST_HIGH (667 of 694 seeds right) and 50.7 % for TRUST_MEDIUM (159 of 283).
TRUST_LOW = 1
TRUST_MEDIUM = 2
TRUST_HIGH = 3
TRUST_VERIFIED = 4
MIN_MEDIUM_SEED_PROBABILITY = 0.5
TRUST_RATES = {TRUST_LOW: 0.0, TRUST_MEDIUM: 0.4, TRUST_HIGH: 0.9, TRUST_VERIFIED: 1.0}

_engine = []
_engine_lock = threading.Lock()
_fingerprint_network = []



def load_engine():
    """
        Loads the two word models and the dictionary decoder once (3 to 4
        seconds on a Pi Zero). Their work buffers are shared: one reading at a
        time.
    """
    with _engine_lock:
        if not _engine:
            from .decoder import Decoder
            from .word_model import WordModel
            _engine.extend((WordModel(), Decoder(), WordModel(LOWERCASE_MODEL_PATH)))
    return _engine



def fingerprint_network():
    """
        The fingerprint network (FINGERPRINT_MODEL_PATH), loaded once.
    """
    with _engine_lock:
        if not _fingerprint_network:
            from .word_model import WordModel
            _fingerprint_network.append(WordModel(FINGERPRINT_MODEL_PATH))
    return _fingerprint_network[0]



class WordReading:
    """
        image: grayscale crop of the handwritten word
        candidates: (word, log-likelihood) pairs, most likely first
    """
    def __init__(self, image: Image.Image, candidates: list[tuple[str, float]]):
        self.image = image
        self.candidates = candidates



class Trust:
    """
        Trust in a reading.

        verified: the fingerprint written with the words matches the seed
        level: TRUST_LOW, TRUST_MEDIUM, TRUST_HIGH or TRUST_VERIFIED
        rate: lowest share of fully right seeds announced for the level (see
            TRUST_RATES)
    """
    def __init__(self, level: int):
        self.level = level
        self.verified = level == TRUST_VERIFIED
        self.rate = TRUST_RATES[level]


    def __repr__(self):
        return f"Trust({self.level})"



class SeedReading:
    """
        kind: "card12", "card24" or "list"
        words: one WordReading per position
        min_seed_probability: see checksum.Analysis
        complete: False if text was left out of the crops, so that the number
            of words is uncertain: no word is then certain unless confirmed
    """
    def __init__(self, kind: str, words: list[WordReading], min_seed_probability: float = MIN_SEED_PROBABILITY,
                 complete: bool = True):
        self.kind = kind
        self.words = words
        self.min_seed_probability = min_seed_probability
        self.complete = complete
        self.doubtful = set() if complete else set(range(len(words)))
        self.ambiguous_order = False
        self.numbered = False
        self.disputed = False
        self.fingerprint_crop = None
        self.fingerprint_others = []
        self.fingerprint = None
        self.verified = False
        self.extra_lines = False
        self.numbers_incoherent = False
        self._last = None


    def mark_ambiguous_order(self):
        """
            The words may be in another order: none is certain unless confirmed.
        """
        self.ambiguous_order = True
        self.doubtful = set(range(len(self.words)))
        self._last = None


    def mark_disputed(self):
        """
            Another reading of the photo disagrees: no word is certain unless
            confirmed.
        """
        self.disputed = True
        self.doubtful = set(range(len(self.words)))
        self._last = None


    def mark_fingerprint_mismatch(self):
        """
            The fingerprint written with the words matches none of the seeds
            read: no word is certain unless confirmed.
        """
        self.doubtful = set(range(len(self.words)))
        self._last = None


    def is_plausible(self) -> bool:
        """
            True if the seed of this reading passes the checksum with a
            probability of at least MIN_PLAUSIBLE_SEED.
        """
        analysis = self.analyze()
        return analysis.status != "invalid" and analysis.confidence >= MIN_PLAUSIBLE_SEED


    def analyze(self, confirmed: dict[int, str] = None) -> Analysis:
        """
            Seed chosen from the readings, the words confirmed by the user
            ({position: word}) being fixed. The last analysis is kept: it takes
            up to 2 seconds on a Pi Zero when the seed is uncertain.
        """
        confirmed = dict(confirmed or {})
        readings = [[(confirmed[k], 0.0)] if k in confirmed else w.candidates for k, w in enumerate(self.words)]
        if self._last is None or self._last[0] != confirmed or len(self._last[1]) != len(readings) \
                or any(a is not b for a, b in zip(self._last[1], readings)):
            self._last = (confirmed, readings, Analysis(readings, self.min_seed_probability, self.doubtful))
        return self._last[2]


    def trust(self, confirmed: dict[int, str] = None) -> Trust:
        """
            Trust in the seed of analyze(confirmed): TRUST_VERIFIED only if its
            fingerprint was read and matched (words confirmed by the user that
            differ from the verified seed take that away), else the level of
            its checksum analysis (see TRUST_RATES).
        """
        analysis = self.analyze(confirmed)
        if self.verified and all(self.words[k].candidates[0][0] == w for k, w in (confirmed or {}).items()):
            return Trust(TRUST_VERIFIED)
        if self.fingerprint is not None and self.fingerprint.status == "mismatch":
            return Trust(TRUST_LOW)
        if self.extra_lines:
            # rows of the list read as a fingerprint or a label rather than as words, not verified without them
            return Trust(TRUST_LOW)
        if self.numbers_incoherent:
            # numbers read that contradict the order of the words (detector.NUMBER_CHECK)
            return Trust(TRUST_LOW)
        if analysis.status == "valid":
            return Trust(TRUST_HIGH)
        if analysis.status == "uncertain" and analysis.confidence >= MIN_MEDIUM_SEED_PROBABILITY:
            return Trust(TRUST_MEDIUM)
        return Trust(TRUST_LOW)



class Progress:
    """
        Progress of a reading, reported as a fraction to `report(fraction)`.
        Each step done moves the fraction by its share of the cost (seconds on
        a Pi Zero) expected to remain: the progress is steady when the expected
        costs are right, and never goes back nor stops long when they are not.
    """
    def __init__(self, report, remaining: float):
        self.report = report
        self.remaining = remaining
        self.fraction = 0.0
        report(0.0)


    def step(self, cost: float):
        """
            A step of expected cost `cost` is done.
        """
        share = 0.5 if self.remaining <= 2 * cost else cost / self.remaining
        self.remaining = max(self.remaining - cost, 0.0)
        self.fraction += (1 - self.fraction) * share
        self.report(self.fraction)


    def expect(self, remaining: float):
        """
            New estimate of the cost of the steps left.
        """
        self.remaining = remaining



def top_probability(candidates: list[tuple[str, float]]) -> float:
    """
        Probability of the most likely word among the candidates.
    """
    return 1 / sum(math.exp(s - candidates[0][1]) for _, s in candidates)



def log_softmax(scores: np.ndarray) -> np.ndarray:
    """
        Word log-probabilities from word scores.
    """
    scores = np.maximum(scores.astype(np.float64), MIN_WORD_SCORE)
    top = scores.max()
    return scores - (top + math.log(np.exp(scores - top).sum()))



class LayoutReader:
    """
        Reads the words of `layout` with either network or their blend; the
        scores of each crop are computed once per network.
    """
    def __init__(self, layout):
        self.layout = layout
        self.inks = {}     # (k, variant): (crop, ink map or None)
        self.scores = {}   # (k, variant, model): word scores
        self.log_probs = {}  # (k, variant, model): outputs of the network
        self.numbers = {}  # k: (number, probability) or None
        self._alternative = []


    def alternative(self):
        """
            LayoutReader of the alternative layout, or None (computed once).
        """
        if not self._alternative:
            try:
                layout = self.layout.alternative() if self.layout.alternative else None
            except WordsNotFound:
                layout = None
            self._alternative.append(LayoutReader(layout) if layout else None)
        return self._alternative[0]


    def model_scores(self, k: int, variant: int, model: int) -> np.ndarray | None:
        """
            Scores of every word for crop (k, variant) by network `model` (0:
            first, 1: lowercase), or None if no ink is found on it.
        """
        _, ink = self.inks[(k, variant)]
        if ink is None:
            return None
        if (k, variant, model) not in self.scores:
            first, decoder, lowercase = load_engine()
            network = lowercase if model else first
            self.log_probs[(k, variant, model)] = network.log_probs(ink)
            self.scores[(k, variant, model)] = decoder.scores(self.log_probs[(k, variant, model)])
        return self.scores[(k, variant, model)]


    def read(self, k: int, variant: int, weights: tuple[float, float]) -> WordReading | None:
        """
            The CANDIDATES most likely words of crop (k, variant), the scores of
            the networks being weighted by `weights`; if no ink is found,
            CANDIDATES words with equal scores. None if there is no such crop.
        """
        words = load_engine()[1].words
        if (k, variant) not in self.inks:
            crop = self.layout.crop(k, variant)
            self.inks[(k, variant)] = (crop, None if crop is None else normalize_word(crop))
        crop, ink = self.inks[(k, variant)]
        if crop is None:
            return None
        if ink is None:
            return WordReading(crop, [(w, 0.0) for w in words[:CANDIDATES]])
        if weights[1] == 0:
            scores = self.model_scores(k, variant, 0)
        elif weights[0] == 0:
            scores = self.model_scores(k, variant, 1)
        else:
            scores = weights[0] * log_softmax(self.model_scores(k, variant, 0)) \
                + weights[1] * log_softmax(self.model_scores(k, variant, 1))
        scores = scores / TEMPERATURES[weights]
        best = np.argsort(-scores, kind="stable")[:CANDIDATES]
        return WordReading(crop, [(words[i], float(scores[i])) for i in best])



    def number(self, k: int) -> tuple[int, float] | None:
        """
            Most probable number 1 to layout.count written before word k, read
            by the first network, and its probability: on the number cut off
            the line if any, else on the crop of the word (a number written
            against the word is part of it). None if no ink is found.
        """
        if k not in self.numbers:
            self.numbers[k] = None
            crop = self.layout.number_crop(k) if self.layout.number_crop else None
            if crop is not None:
                ink = normalize_word(crop)
                log_probs = None if ink is None else load_engine()[0].log_probs(ink)
            else:
                self.read(k, 0, FIRST)
                log_probs = None if self.model_scores(k, 0, 0) is None else self.log_probs[(k, 0, 0)]
            if log_probs is not None:
                probabilities = number_probabilities(log_probs, self.layout.count)
                best = int(np.argmax(probabilities))
                self.numbers[k] = (best + 1, float(probabilities[best]))
        return self.numbers[k]


    def numbered_order(self) -> int | None:
        """
            Index in layout.orders of the order fixed by the line numbers, or
            None (see NUMBER_PROBABILITY).
        """
        orders = self.layout.orders
        votes = [0] * len(orders)
        for k in range(self.layout.count):
            number = self.number(k)
            positions = [order.index(k) + 1 for order in orders]
            if number is None or number[1] < NUMBER_PROBABILITY or len(set(positions)) == 1:
                continue
            for i, position in enumerate(positions):
                votes[i] += position == number[0]
        best = max(range(len(orders)), key=lambda i: votes[i])
        if votes[best] >= MIN_NUMBER_VOTES and all(v == 0 for i, v in enumerate(votes) if i != best):
            return best
        return None



def combine(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
        Sum of the log-probabilities of two readings of the same word. Equal
        sums keep the order of `a`, then `b`, so that the result does not
        depend on the hash seed of Python.
    """
    def log_probabilities(candidates):
        top = candidates[0][1]
        total = top + math.log(sum(math.exp(s - top) for _, s in candidates))
        return {w: s - total for w, s in candidates}
    la, lb = log_probabilities(a), log_probabilities(b)
    combined = {w: la.get(w, FUSION_FLOOR) + lb.get(w, FUSION_FLOOR) for w in [*la, *(w for w in lb if w not in la)]}
    return sorted(combined.items(), key=lambda item: -item[1])[:CANDIDATES]



def read_layout(reader: LayoutReader, progress: Progress, weights: tuple[float, float], min_seed_probability: float,
                on_word=None, plausible: list = None, pool: list = None) -> SeedReading:
    """
        Reads the words of `reader.layout` with `weights`, reporting each one to
        `progress` and to `on_word` (see read_seed). The plausible seeds of the
        reading orders are appended to `plausible`; their most probable valid
        seeds to `pool`, as (rank, words, word readings).
    """
    layout = reader.layout
    words = []

    def seed_reading(order):
        reading = SeedReading(layout.kind, [words[k] for k in order], min_seed_probability, layout.complete)
        reading.fingerprint_crop = getattr(layout, "fingerprint", None)
        reading.fingerprint_others = list(getattr(layout, "fingerprint_others", None) or [])
        reading.layout, reading.order = layout, list(order)
        return reading

    for k in range(layout.count):
        words.append(reader.read(k, 0, weights))
        progress.step(WORD_COST if weights == FIRST else OTHER_WORD_COST)
        if on_word:
            on_word(layout, k, words[-1])
    reading = seed_reading(layout.orders[0])

    # Re-reads of the least probable words, on their other crops
    if layout.variants > 1:
        least_probable = sorted(range(layout.count), key=lambda k: top_probability(words[k].candidates))
        for k in least_probable[:MAX_REREADS]:
            if reading.analyze().status == "valid":
                break
            others = [r for r in (reader.read(k, v, weights) for v in range(1, layout.variants)) if r is not None]
            progress.step(REREAD_COST)
            if others:
                best = max((r.candidates for r in others), key=top_probability)
                words[k] = WordReading(words[k].image, combine(words[k].candidates, best))
                reading = seed_reading(layout.orders[0])

    # Two columns of equal length: column by column, or row by row if only
    # that order gives a valid seed. A permutation of a seed passes the
    # checksum once in 16 for 12 words: when both orders give a plausible
    # seed, the order written is unknown, unless the line numbers tell it.
    readings = [reading] + [seed_reading(order) for order in layout.orders[1:]]
    if pool is not None and getattr(layout, "fingerprint", None) is not None:
        for r in readings:
            for rank, seed in enumerate(r.analyze().valid_seeds(FINGERPRINT_CANDIDATES)):
                pool.append((rank, seed, r.words))
    if any(r.analyze().status != "invalid" for r in readings[1:]):
        order = reader.numbered_order()
        if order is not None:
            reading = readings[order]
            reading.numbered = True
            if plausible is not None and reading.is_plausible():
                plausible.append(reading.analyze().mnemonic)
            return reading
    seeds = [r.analyze().mnemonic for r in readings if r.is_plausible()]
    if plausible is not None:
        plausible.extend(seeds)
    for alternative in readings[1:]:
        if reading.analyze().status == "invalid" and alternative.analyze().status != "invalid":
            reading = alternative
    if len(seeds) > 1:
        reading.mark_ambiguous_order()
    return reading



def read_page(reader: LayoutReader, progress: Progress, weights: tuple[float, float], min_seed_probability: float,
              on_word=None, plausible: list = None, pool: list = None) -> SeedReading:
    """
        Reads `reader.layout` with `weights`, and its list reading if it is a
        card found on part of its lines that gives no valid seed (the better
        one is kept).
    """
    reading = read_layout(reader, progress, weights, min_seed_probability, on_word, plausible, pool)
    if reading.analyze().status != "valid" and reader.alternative():
        other = read_layout(reader.alternative(), progress, weights, min_seed_probability, on_word, plausible, pool)
        if STATUS_RANK[other.analyze().status] > STATUS_RANK[reading.analyze().status]:
            reading = other
    return reading



def reading_cost(layout) -> float:
    """
        Expected cost of the reading of the words of `layout` (see Progress).
    """
    return layout.count * (WORD_COST + OTHER_WORD_COST)



def read_words(layout, progress: Progress, on_word=None, plausible: list = None, pool: list = None) -> SeedReading:
    """
        Reads the words of `layout` with the first network; if the seed is not
        valid, again with the second network, then with the blend of both. A
        reading is kept if its status is better, or as good with a more
        probable seed. The plausible seeds of every reading are appended to
        `plausible`, their most probable valid seeds to `pool` (see read_layout).
    """
    reader = LayoutReader(layout)
    reading = read_page(reader, progress, FIRST, MIN_SEED_PROBABILITY, on_word, plausible, pool)
    best = reading.analyze()
    if best.status != "valid":
        progress.expect(2 * layout.count * OTHER_WORD_COST)
        second = ((LOWERCASE, LOWERCASE_MIN_SEED_PROBABILITY), (BLEND, BLEND_MIN_SEED_PROBABILITY))
        for weights, min_seed_probability in second:
            other = read_page(reader, progress, weights, min_seed_probability, plausible=plausible, pool=pool)
            analysis = other.analyze()
            if (STATUS_RANK[analysis.status], analysis.confidence) > (STATUS_RANK[best.status], best.confidence):
                reading, best = other, analysis
    reading.reader = reader
    return reading



def fingerprint_margins(crop: Image.Image) -> list[tuple[float, float]] | None:
    """
        fingerprint.margins of the line on `crop` for the first and the
        second network, or None if no ink is found.
    """
    ink = normalize_word(crop)
    if ink is None:
        return None
    first, decoder, lowercase = load_engine()
    found = []
    for network in (first, lowercase):
        log_probs = network.log_probs(ink)
        found.append(fingerprint.margins(log_probs, decoder.scores(log_probs)))
    return found



def is_fingerprint(crop: Image.Image) -> bool:
    """
        True if the line on `crop` reads as a fingerprint rather than a word
        (see FINGERPRINT_MARGIN).
    """
    found = fingerprint_margins(crop)
    if fingerprint_accepted(found):
        return True
    if found is None or sum(m for m, _ in found) / len(found) < FINGERPRINT_RESCUE_MARGIN:
        return False
    # near the threshold: the fingerprint network has the last word, with the same thresholds
    _, decoder, _ = load_engine()
    log_probs = fingerprint_network().log_probs(normalize_word(crop))
    return fingerprint_accepted([fingerprint.margins(log_probs, decoder.scores(log_probs))])



def fingerprint_hypothesis_margin(crop: Image.Image) -> float | None:
    """
        Margin of the line on `crop` read as a fingerprint by the fingerprint
        network (fingerprint.margins), if it is at least
        LOOSE_FINGERPRINT_MARGIN with a fit of at least FINGERPRINT_FIT, else
        None. Used only when no line of a list with one line too many passes
        is_fingerprint: the line is then left out as a hypothesis, kept only
        if its fingerprint matches a seed read.
    """
    ink = normalize_word(crop)
    if ink is None:
        return None
    _, decoder, _ = load_engine()
    log_probs = fingerprint_network().log_probs(ink)
    margin, fit = fingerprint.margins(log_probs, decoder.scores(log_probs))
    return margin if margin >= LOOSE_FINGERPRINT_MARGIN and fit >= FINGERPRINT_FIT else None



def line_role(crops: list[Image.Image]) -> tuple[str | None, Image.Image | None]:
    """
        What a row of a list left out of the words reads as, on its crops
        (widest first): ("fingerprint", crop) if a crop passes is_fingerprint,
        ("label", None) if one of the first two reads as a label (see
        LABEL_MARGIN), ("loose", crop) if a crop passes
        fingerprint_hypothesis_margin, ("other", None) if the first reads as
        no BIP39 word (NO_WORD_FIT), else (None, None).
    """
    for crop in crops:
        if is_fingerprint(crop):
            return "fingerprint", crop
    first, decoder, lowercase = load_engine()
    other = False
    for i, crop in enumerate(crops[:2]):
        ink = normalize_word(crop)
        if ink is None:
            continue
        found = []
        for network in (first, lowercase):
            log_probs = network.log_probs(ink)
            found.append(fingerprint.label_margins(log_probs, decoder.scores(log_probs)))
        margin, fit, word_fit = (sum(f[j] for f in found) / len(found) for j in range(3))
        if margin >= LABEL_MARGIN and fit >= LABEL_FIT:
            return "label", None
        other = other or (i == 0 and word_fit <= NO_WORD_FIT)
    for crop in crops:
        if fingerprint_hypothesis_margin(crop) is not None:
            return "loose", crop
    return ("other", None) if other else (None, None)



def fingerprint_accepted(found: list[tuple[float, float]] | None) -> bool:
    """
        True if the margins of a line by both networks (fingerprint_margins)
        make it a fingerprint (see FINGERPRINT_MARGIN).
    """
    if found is None:
        return False
    margin = sum(m for m, _ in found) / len(found)
    fit = sum(f for _, f in found) / len(found)
    return margin >= FINGERPRINT_MARGIN and fit >= FINGERPRINT_FIT



def search(find, gray: Image.Image, progress: Progress, on_word, pool: list, fingerprint) \
        -> tuple[SeedReading | None, list, Exception]:
    """
        (reading, its plausible seeds, None) of the words found by `find`
        (`fingerprint`: see detector.find_words), or (None, [], WordsNotFound) if
        it finds none. The most probable valid seeds are appended to `pool`
        (see read_layout).
    """
    plausible = []
    try:
        layout = find(gray, progress.step, fingerprint=fingerprint)
    except WordsNotFound as e:
        return None, plausible, e
    progress.expect(progress.remaining + reading_cost(layout))
    reading = read_words(layout, progress, on_word, plausible, pool)
    doubtful = getattr(layout, "doubtful", set())
    if doubtful:
        # words found only on the word map are never certain
        order = layout.orders[0]
        reading.doubtful |= set(range(layout.count)) if len(layout.orders) > 1 else {order.index(k) for k in doubtful}
        reading._last = None
    return reading, plausible, None



def stretched_words(reading: SeedReading) -> set[int]:
    """
        Positions of the words whose box is at least STRETCH times longer per
        letter of the word read than the median of the page (detector
        layouts): part of the word was probably not read (faded or broken
        letters).
    """
    lengths = getattr(reading.layout, "lengths", None)
    if lengths is None:
        return set()
    per_letter = [lengths[k] / len(w.candidates[0][0]) for k, w in zip(reading.order, reading.words)]
    median = float(np.median(per_letter))
    return {i for i, x in enumerate(per_letter) if x >= STRETCH * median}



def guard(reading: SeedReading):
    """
        Guards of a valid seed (see MIN_DETECTOR_SEED_PROBABILITY): every word
        becomes uncertain if a handwritten word was left out; stretched words
        and words with faint ink left unread become uncertain.
    """
    layout = reading.layout
    if any(score >= LEFT_OUT_SCORE and length >= LEFT_OUT_LENGTH for score, length, _ in layout.extra_words):
        reading.mark_disputed()
        return
    reading.min_seed_probability = max(reading.min_seed_probability, MIN_DETECTOR_SEED_PROBABILITY)
    reading.doubtful |= stretched_words(reading)
    for i, k in enumerate(reading.order):
        crop = reading.reader.inks[(k, 0)][0] if (k, 0) in reading.reader.inks else layout.crop(k, 0)
        if crop is not None and unread_ink(crop) >= UNREAD_INK:
            reading.doubtful.add(i)
    reading._last = None



def read_seed(image: Image.Image, progress=None, on_word=None) -> SeedReading:
    """
        Reads the seed written on `image` (upright photo, full resolution).
        `progress(fraction)` is called as the reading goes, at least every
        second or so on a Pi Zero; `on_word(layout, k, word_reading)` after the
        first reading of each word. Raises WordsNotFound if no document with a
        valid number of words is found.

        The words are found by the detector (detector.find_words) and read;
        a valid seed then goes through the guards (see guard), and a
        fingerprint line written with the words is checked
        (check_fingerprint). A reading neither verified nor valid is tried
        again without the words that may be scribbled over (see
        detector.CROSSED_OUT); the second one is kept if it is verified, or
        else has a better status, or else a more probable seed.
    """
    progress = Progress(progress or (lambda fraction: None), detector.DETECTOR_COST + 12 * (WORD_COST + OTHER_WORD_COST))
    if not _engine:
        progress.expect(progress.remaining + LOAD_COST)
        load_engine()
        progress.step(LOAD_COST)
    gray = image.convert("L")
    detector._detections.clear()

    def spare_line(crop):
        progress.step(WORD_COST + OTHER_WORD_COST)
        return is_fingerprint(crop)

    def hypothesis_margin(crop):
        progress.step(WORD_COST)
        return fingerprint_hypothesis_margin(crop)

    spare_line.margin = hypothesis_margin

    def role(crops):
        progress.step(2 * WORD_COST)
        return line_role(crops)

    spare_line.role = role

    def attempt(crossed: bool):
        pool = []

        def find(image, step, fingerprint):
            return detector.find_words(image, step, fingerprint=fingerprint, crossed=crossed)

        reading, _, not_found = search(find, gray, progress, on_word, pool, spare_line)
        if not crossed and reading is not None and getattr(reading.layout, "numbers_incoherent", False):
            # numbers read that contradict the order of the words: read again as one list with the words of every
            # document (a part of the sheet taken for a card); kept if its numbers are read and agree with its order,
            # otherwise the first reading is kept with every word uncertain
            merged_pool = []

            def find_merged(image, step, fingerprint):
                return detector.find_words(image, step, fingerprint=fingerprint, merged=True)

            other, _, _ = search(find_merged, gray, progress, on_word, merged_pool, spare_line)
            if other is not None and not other.layout.numbers_incoherent \
                    and other.layout.numbers_checked >= detector.MIN_NUMBERS_READ:
                reading = other
                pool[:] = merged_pool
            else:
                reading.numbers_incoherent = True
                reading.mark_disputed()
        if not crossed and (reading is not None and getattr(reading.layout, "extra_lines", False)
                            and reading.fingerprint_crop is None and not getattr(reading, "verified", False)):
            # a list numbered on most of its lines, with lines without a number: read again without one or two rows
            # at its start or end that read as a fingerprint or a label (detector.extra_line_sets); one fingerprint check
            # at most per attempt, as on the other paths (none was made here)
            extra_pool = []

            def find_extra(image, step, fingerprint):
                return detector.find_words(image, step, fingerprint=fingerprint, extra=True)

            other, _, _ = search(find_extra, gray, progress, on_word, extra_pool, spare_line)
            if other is not None:
                if other.analyze().status == "valid":
                    guard(other)
                if other.fingerprint_crop is not None:
                    other = check_fingerprint(other, extra_pool, progress)
                if getattr(other, "verified", False) \
                        or reading is None and not other.layout.fingerprint_hypothesis:
                    return other, None
                if reading is not None and other.layout.extra_lines_strong:
                    # rows read as a fingerprint (strict thresholds) or a label, not verified without them
                    reading.extra_lines = True
        if reading is None:
            return None, not_found
        if reading.analyze().status == "valid":
            guard(reading)
        if reading.fingerprint_crop is not None and getattr(reading.layout, "fingerprint_optional", False):
            # a row left out across the columns tried as the fingerprint: kept only if it matches a seed read
            doubtful = set(reading.doubtful)
            checked = check_fingerprint(reading, pool, progress)
            if getattr(checked, "verified", False):
                reading = checked
            else:
                reading.doubtful, reading._last = doubtful, None
                reading.fingerprint_crop, reading.fingerprint = None, None
        elif reading.fingerprint_crop is not None:
            reading = check_fingerprint(reading, pool, progress)
        if getattr(reading.layout if hasattr(reading, "layout") else None, "fingerprint_hypothesis", False) \
                and not getattr(reading, "verified", False):
            # a line left out as a fingerprint hypothesis whose fingerprint matches no seed read: the list has one
            # line too many, as without the hypothesis
            return None, reading.layout.not_found
        return reading, None

    def rank(reading):
        if reading is None:
            return (False, -1, 0.0)
        analysis = reading.analyze()
        return (bool(getattr(reading, "verified", False)), STATUS_RANK[analysis.status], analysis.confidence)

    try:
        reading, not_found = attempt(False)
        if not (reading is not None and (getattr(reading, "verified", False)
                                                           or reading.analyze().status == "valid")):
            # neither verified nor valid: a second search without the words that may be scribbled over (see
            # detector.CROSSED_OUT), kept if it reads better
            other, _ = attempt(True)
            if rank(other) > rank(reading):
                reading = other
    finally:
        detector._detections.clear()
    if reading is None:
        raise not_found
    progress.report(1.0)
    return reading



def check_fingerprint(reading: SeedReading, pool: list, progress: Progress) -> SeedReading:
    """
        Checks the fingerprint line of `reading`, read by the fingerprint
        network (see FINGERPRINT_CANDIDATES)
        against the candidate seeds of `pool` (see read_layout), most probable
        first. Returns the reading of the matching seed, every word certain
        (`verified`), or `reading` with every word uncertain; the check is in
        `fingerprint` of the reading returned.
    """
    crops = []
    for crop in [reading.fingerprint_crop] + list(getattr(reading, "fingerprint_others", [])):
        # two fingerprint lines (with and without a passphrase): the mean of their probabilities also sums to 1
        w, h = crop.size
        trim = round(FINGERPRINT_TRIM * h)
        crops += [crop] + ([crop.crop((0, trim, w, h)), crop.crop((0, 0, w, h - trim))] if trim else [])
    inks = [ink for ink in (normalize_word(c) for c in crops) if ink is not None]
    if not inks:
        return reading
    check = fingerprint.FingerprintCheck([fingerprint.FingerprintReading(fingerprint_network().log_probs(ink))
                                          for ink in inks])
    candidates, seen = [], set()
    for _, seed, words in sorted(pool, key=lambda item: item[0]):
        if tuple(seed) not in seen and len(candidates) < FINGERPRINT_CANDIDATES:
            seen.add(tuple(seed))
            candidates.append((seed, words))
    progress.expect(len(candidates) * FINGERPRINT_COST)
    matched = None
    # the chosen seed first, then the others together (one pass of the networks' outputs for all of them)
    for group in (candidates[:1], candidates[1:]):
        if matched or not group:
            continue
        fingerprints = []
        for seed, _ in group:
            fingerprints.append(fingerprint.seed_fingerprint(seed))
            progress.step(FINGERPRINT_COST)
        check.checked += len(group)
        for i, ((seed, words), fp, probability) in enumerate(zip(group, fingerprints, check.probabilities(fingerprints))):
            check.match_probability = max(check.match_probability, probability)
            if probability >= fingerprint.MIN_MATCH_PROBABILITY:
                matched = (seed, words)
                check.matched, check.rank = fp, check.checked - len(group) + i
                break
    if matched is None:
        check.status = "mismatch"
        reading.mark_fingerprint_mismatch()
        reading.fingerprint = check
        return reading
    seed, words = matched
    check.status = "verified"
    verified = SeedReading(reading.kind, [WordReading(w.image, [(word, 0.0)]) for word, w in zip(seed, words)],
                           reading.min_seed_probability)
    verified.fingerprint_crop = reading.fingerprint_crop
    verified.fingerprint_others = list(getattr(reading, "fingerprint_others", []))
    verified.fingerprint = check
    verified.verified = True
    verified.numbered = reading.numbered
    return verified


