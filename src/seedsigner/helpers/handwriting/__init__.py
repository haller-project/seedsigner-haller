"""
    Reading of a seed handwritten on paper, from a photo: the official
    SeedSigner 12- or 24-word card, another printed card, or a free list of 12
    to 24 words (any paper, one or more columns, with or without numbers).

        reading = read_seed(image, progress, on_word)
        analysis = reading.analyze(confirmed)

    Each word is cut out of the photo (page.py), normalized (preprocess.py),
    read by a small neural network (word_model.py) and scored against the 2 048
    BIP39 words (decoder.py); the checksum then chooses the seed among the
    plausible readings and flags uncertain words (checksum.py). The network is
    Haller-1, made of two sets of weights: a seed that is not valid with the
    first one is read again by the second one, trained with more lowercase and
    cursive writing. The words are searched by the rules of page.py first; a
    photo on which they find no valid seed is searched again by the word
    detector of Haller-1 (detector.py), whose seed is only valid if the rules
    do not contradict it, then, if no list was found, as a list of two columns
    written too close together (page.find_aligned_words). The result must
    always be reviewed by the user against the paper.
"""
import math
import threading

import numpy as np
from PIL import Image

from . import detector, fingerprint
from .checksum import Analysis, MIN_PLAUSIBLE_SEED, MIN_SEED_PROBABILITY
from .decoder import number_probabilities
from .page import LIST_SEARCH_COST, RECTIFY_COSTS, SEARCH_COST, WordsNotFound, find_aligned_words, find_words
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

# Each search checks the seed of the other one (see agrees): a word it read
# with a probability of at least AGREEMENT_PROBABILITY must be the same.
AGREEMENT_PROBABILITY = 0.9

# A list of two columns too close for the rules to tell apart is cut where the
# right column starts (page.find_aligned_words), when neither the rules nor the
# detector give a valid seed. Cutting a page that is not such a list gives
# random words: the reading is kept only if its seed probability reaches
# ALIGNED_MIN_SEED_PROBABILITY (1e-9 to 1e-16 for random words).
ALIGNED_MIN_SEED_PROBABILITY = 1e-4

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

# The fingerprint line is checked against the FINGERPRINT_CANDIDATES most
# probable valid seeds of the readings (every reading order, network and
# layout), in order of rank: the first one whose fingerprint matches (see
# fingerprint.MIN_MATCH_PROBABILITY) is the seed written, every word certain.
# When none matches, the seed is never called wrong (a word or a digit may be
# misread, or the fingerprint is that of the seed with a passphrase): every
# word is uncertain. FINGERPRINT_COST: computation of one fingerprint on a Pi
# Zero (PBKDF2-HMAC-SHA512, 2048 rounds, 0.062 s; master key and HASH160,
# 0.005 s; measured with embit on 20/11/2026) and its check, seconds.
FINGERPRINT_CANDIDATES = 16
FINGERPRINT_COST = 0.1

# A seed of the detector is only valid, when the rules found no valid seed,
# if its probability is at least DETECTOR_MIN_SEED_PROBABILITY, and, when the
# rules read as many words, if they read every word the same with a
# probability of at least RULES_DOUBT: the detector mostly reads the photos
# where the rules hesitate (crossed-out or faded words, other documents in the
# field), where wrong readings that pass the checksum are more frequent.
# Chosen on simulated pages kept apart from the tests (the lowest probability
# with no wrong seed declared valid).
DETECTOR_MIN_SEED_PROBABILITY = 0.99
RULES_DOUBT = 0.5

# A word of a valid detector seed whose box is at least STRETCH times longer
# per letter of the word read than the median of the page is uncertain: part
# of it was probably not read (faded or broken letters, see stretched_words).
# Chosen on simulated pages kept apart from the tests.
import os  # noqa: E402  (experiment switch, removed before release)
STRETCH = float(os.environ.get("HW_STRETCH", "1.5"))

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
        self.fingerprint = None
        self.verified = False
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
    rectify = sum(RECTIFY_COSTS) if layout.kind != "list" else 0.0
    return rectify + layout.count * (WORD_COST + OTHER_WORD_COST)



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
    return reading



def agrees(reading: SeedReading, other: SeedReading | None, plausible: list, weak: bool = False) -> bool:
    """
        True if the valid seed of `reading` may stay valid given `other`, the
        reading of the same photo by the other search (None if it found no
        words), and `plausible`, the plausible seeds of `other`.
        * No plausible seed of `other` differs from it (weak: no valid seed).
        * No word read by `other` (same number of words) with a probability of
          at least AGREEMENT_PROBABILITY differs from it (weak: unless `other`
          is invalid, which happens when it read another document, the page in
          the wrong direction or the words shifted).
    """
    if other is None:
        return True
    seed = reading.analyze().mnemonic
    analysis = other.analyze()
    if weak:
        if analysis.status == "valid" and analysis.mnemonic != seed:
            return False
    elif any(s != seed for s in plausible):
        return False
    if len(other.words) == len(seed) and not (weak and analysis.status == "invalid"):
        for word, chosen in zip(other.words, seed):
            if word.candidates[0][0] != chosen and top_probability(word.candidates) >= AGREEMENT_PROBABILITY:
                return False
    return True



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
    return fingerprint_accepted(fingerprint_margins(crop))



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



def confirms(rules: SeedReading | None, seed: list[str]) -> bool:
    """
        True if `rules` found another number of words, or read every word of
        `seed` as its most probable word with a probability of at least
        RULES_DOUBT.
    """
    if rules is None or len(rules.words) != len(seed):
        return True
    return all(w.candidates[0][0] == s and top_probability(w.candidates) >= RULES_DOUBT
               for w, s in zip(rules.words, seed))



def search(find, gray: Image.Image, progress: Progress, on_word, pool: list, fingerprint) \
        -> tuple[SeedReading | None, list, Exception]:
    """
        (reading, its plausible seeds, None) of the words found by `find`
        (`fingerprint`: see page.find_words), or (None, [], WordsNotFound) if
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



def read_seed(image: Image.Image, progress=None, on_word=None) -> SeedReading:
    """
        Reads the seed written on `image` (upright photo, full resolution).
        `progress(fraction)` is called as the reading goes, at least every
        second or so on a Pi Zero; `on_word(layout, k, word_reading)` after the
        first reading of each word. Raises WordsNotFound if no card or list with
        a valid number of words is found.

        The words are searched by the rules first (page.py); a valid seed of
        the rules is the result. Otherwise the words are searched again by the
        detector (detector.py). Its seed is only valid if the rules read no
        other plausible seed (see agrees), and read every word the same if
        they found as many (see confirms); if the numbers of its first and last
        words do not contradict their positions; and if its probability is at
        least DETECTOR_MIN_SEED_PROBABILITY. The detector reading is kept
        unless the rules one has a better status, or the same with a more
        probable seed.
    """
    progress = Progress(progress or (lambda fraction: None), SEARCH_COST + 12 * (WORD_COST + OTHER_WORD_COST))
    if not _engine:
        progress.expect(progress.remaining + LOAD_COST)
        load_engine()
        progress.step(LOAD_COST)
    gray = image.convert("L")
    plausible = []
    pool = []

    def spare_line(crop):
        progress.step(WORD_COST + OTHER_WORD_COST)
        return is_fingerprint(crop)

    rules, plausible, not_found = search(find_words, gray, progress, on_word, pool, spare_line)
    reading = rules
    if rules is None or rules.analyze().status != "valid":
        progress.expect(detector.DETECTOR_COST + 12 * (WORD_COST + OTHER_WORD_COST))
        first, _, _ = search(detector.find_words, gray, progress, on_word, pool, spare_line)
        if first is not None and first.analyze().status == "valid":
            if not agrees(first, rules, plausible) or not confirms(rules, first.analyze().mnemonic) \
                    or first.layout.check(first.order):
                first.mark_disputed()
            else:
                if first.min_seed_probability < DETECTOR_MIN_SEED_PROBABILITY:
                    first.min_seed_probability = DETECTOR_MIN_SEED_PROBABILITY
                stretched = stretched_words(first)
                if stretched:
                    first.doubtful |= stretched
                first._last = None
        if first is not None:
            a, b = first.analyze(), rules.analyze() if rules is not None else None
            if b is None or (STATUS_RANK[a.status], a.confidence) >= (STATUS_RANK[b.status], b.confidence):
                reading = first

    # No valid seed and no layout found by the rules on a list: its columns may be too close to be told apart
    if not_found is not None and not not_found.card and (reading is None or reading.analyze().status != "valid"):
        progress.expect(LIST_SEARCH_COST + 12 * (WORD_COST + OTHER_WORD_COST))
        try:
            layout = find_aligned_words(gray, progress.step, fingerprint=spare_line)
        except WordsNotFound:
            layout = None
        if layout is not None:
            progress.expect(reading_cost(layout))
            other = read_words(layout, progress, on_word, pool=pool)
            analysis = other.analyze()
            if analysis.confidence >= ALIGNED_MIN_SEED_PROBABILITY and (reading is None or (
                    STATUS_RANK[analysis.status], analysis.confidence)
                    > (STATUS_RANK[reading.analyze().status], reading.analyze().confidence)):
                reading = other
    if reading is None:
        raise not_found
    if reading.fingerprint_crop is not None:
        reading = check_fingerprint(reading, pool, progress)
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
    ink = normalize_word(reading.fingerprint_crop)
    if ink is None:
        return reading
    check = fingerprint.FingerprintCheck([fingerprint.FingerprintReading(fingerprint_network().log_probs(ink))])
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
    verified.fingerprint = check
    verified.verified = True
    verified.numbered = reading.numbered
    return verified
