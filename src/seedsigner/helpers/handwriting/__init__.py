"""
    Reading of a seed handwritten on paper, from a photo: the official
    SeedSigner 12- or 24-word card, or a free list of 12 to 24 words (blank or
    ruled paper, one or two columns, with or without numbers).

        reading = read_seed(image, progress, on_word)
        analysis = reading.analyze(confirmed)

    Each word is cut out of the photo (page.py), normalized (preprocess.py),
    read by a small neural network (word_model.py) and scored against the 2 048
    BIP39 words (decoder.py); the checksum then chooses the seed among the
    plausible readings and flags uncertain words (checksum.py). The network is
    Haller-1, made of two sets of weights: a seed that is not valid with the
    first one is read again by the second one, trained with more lowercase and
    cursive writing. The result must always be reviewed by the user against
    the paper.
"""
import math
import threading

import numpy as np
from PIL import Image

from .checksum import Analysis, MIN_PLAUSIBLE_SEED, MIN_SEED_PROBABILITY
from .decoder import number_probabilities
from .page import RECTIFY_COSTS, SEARCH_COST, WordsNotFound, find_words
from .preprocess import normalize_word
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

# Expected duration of the steps of a reading, in seconds on a Pi Zero (see
# Progress): loading of the models, reading of a word by the first network
# (crop, normalization, network, decoder), by the second network or the blend
# (the normalized word is reused), re-reading of a card word on its other crops.
# The second reading is counted for half its cost before it is known to be needed.
LOAD_COST = 0.8
WORD_COST = 0.2
OTHER_WORD_COST = 0.06
REREAD_COST = 0.4

_engine = []
_engine_lock = threading.Lock()



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



class WordReading:
    """
        image: grayscale crop of the handwritten word
        candidates: (word, log-likelihood) pairs, most likely first
    """
    def __init__(self, image: Image.Image, candidates: list[tuple[str, float]]):
        self.image = image
        self.candidates = candidates



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
        self._last = None


    def mark_ambiguous_order(self):
        """
            The words may be in another order: none is certain unless confirmed.
        """
        self.ambiguous_order = True
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
                on_word=None) -> SeedReading:
    """
        Reads the words of `reader.layout` with `weights`, reporting each one to
        `progress` and to `on_word` (see read_seed).
    """
    layout = reader.layout
    words = []

    def seed_reading(order):
        return SeedReading(layout.kind, [words[k] for k in order], min_seed_probability, layout.complete)

    for k in range(layout.count):
        words.append(reader.read(k, 0, weights))
        progress.step(WORD_COST if weights == FIRST else OTHER_WORD_COST)
        if on_word:
            on_word(layout, k, words[-1])
    reading = seed_reading(range(layout.count))

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
                reading = seed_reading(range(layout.count))

    # Two columns of equal length: column by column, or row by row if only
    # that order gives a valid seed. A permutation of a seed passes the
    # checksum once in 16 for 12 words: when both orders give a plausible
    # seed, the order written is unknown, unless the line numbers tell it.
    readings = [reading] + [seed_reading(order) for order in layout.orders[1:]]
    if any(r.analyze().status != "invalid" for r in readings[1:]):
        order = reader.numbered_order()
        if order is not None:
            reading = readings[order]
            reading.numbered = True
            return reading
    seeds = [r.analyze().mnemonic for r in readings if r.is_plausible()]
    for alternative in readings[1:]:
        if reading.analyze().status == "invalid" and alternative.analyze().status != "invalid":
            reading = alternative
    if len(seeds) > 1:
        reading.mark_ambiguous_order()
    return reading



def read_page(reader: LayoutReader, progress: Progress, weights: tuple[float, float], min_seed_probability: float,
              on_word=None) -> SeedReading:
    """
        Reads `reader.layout` with `weights`, and its list reading if it is a
        card found on part of its lines that gives no valid seed (the better
        one is kept).
    """
    reading = read_layout(reader, progress, weights, min_seed_probability, on_word)
    if reading.analyze().status != "valid" and reader.alternative():
        other = read_layout(reader.alternative(), progress, weights, min_seed_probability, on_word)
        if STATUS_RANK[other.analyze().status] > STATUS_RANK[reading.analyze().status]:
            reading = other
    return reading



def reading_cost(layout) -> float:
    """
        Expected cost of the reading of the words of `layout` (see Progress).
    """
    rectify = sum(RECTIFY_COSTS) if layout.kind != "list" else 0.0
    return rectify + layout.count * (WORD_COST + OTHER_WORD_COST)



def read_seed(image: Image.Image, progress=None, on_word=None) -> SeedReading:
    """
        Reads the seed written on `image` (upright photo, full resolution).
        `progress(fraction)` is called as the reading goes, at least every
        second or so on a Pi Zero; `on_word(layout, k, word_reading)` after the
        first reading of each word. Raises WordsNotFound if no card or list with
        a valid number of words is found.
    """
    progress = Progress(progress or (lambda fraction: None), SEARCH_COST + 12 * (WORD_COST + OTHER_WORD_COST))
    if not _engine:
        progress.expect(progress.remaining + LOAD_COST)
        load_engine()
        progress.step(LOAD_COST)
    layout = find_words(image.convert("L"), progress.step)
    progress.expect(reading_cost(layout))
    reader = LayoutReader(layout)
    reading = read_page(reader, progress, FIRST, MIN_SEED_PROBABILITY, on_word)

    # Not valid: read again by the second network, then by the blend of both;
    # a reading is kept if its status is better, or as good with a more
    # probable seed
    best = reading.analyze()
    if best.status != "valid":
        progress.expect(2 * layout.count * OTHER_WORD_COST)
        second = ((LOWERCASE, LOWERCASE_MIN_SEED_PROBABILITY), (BLEND, BLEND_MIN_SEED_PROBABILITY))
        for weights, min_seed_probability in second:
            other = read_page(reader, progress, weights, min_seed_probability)
            analysis = other.analyze()
            if (STATUS_RANK[analysis.status], analysis.confidence) > (STATUS_RANK[best.status], best.confidence):
                reading, best = other, analysis
    progress.report(1.0)
    return reading
