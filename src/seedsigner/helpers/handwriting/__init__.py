"""
    Reading of a seed handwritten on paper, from a photo: the official
    SeedSigner 12- or 24-word card, or a free list of 12 to 24 words (blank or
    ruled paper, one or two columns, with or without numbers).

        reading = read_seed(image, progress)
        analysis = reading.analyze(confirmed)

    Each word is cut out of the photo (page.py), normalized (preprocess.py),
    read by a small neural network (word_model.py) and scored against the 2 048
    BIP39 words (decoder.py); the checksum then chooses the seed among the
    plausible readings and flags uncertain words (checksum.py). A seed that is
    not valid is read again by a second network, trained with more lowercase
    and cursive writing. The result must always be reviewed by the user against
    the paper.
"""
import math
import threading

import numpy as np
from PIL import Image

from .checksum import Analysis, MIN_SEED_PROBABILITY
from .page import WordsNotFound, find_words
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
LOWERCASE_MODEL_PATH = MODEL_PATH.replace("word_model.npz", "word_model_lowercase.npz")
LOWERCASE_MIN_SEED_PROBABILITY = 0.85

# The same photos are also read with a blend of both networks: weighted sum of
# their word log-probabilities, BLEND_WEIGHT for the first one. It reuses the
# scores already computed. Scores of words dropped by the decoder beam are
# raised to MIN_WORD_SCORE first, so that the weighted sum stays finite.
BLEND_WEIGHT = 0.65
BLEND_MIN_SEED_PROBABILITY = 0.5
MIN_WORD_SCORE = -1e4

FIRST, LOWERCASE, BLEND = (1.0, 0.0), (0.0, 1.0), (BLEND_WEIGHT, 1 - BLEND_WEIGHT)

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
    """
    def __init__(self, kind: str, words: list[WordReading], min_seed_probability: float = MIN_SEED_PROBABILITY):
        self.kind = kind
        self.words = words
        self.min_seed_probability = min_seed_probability


    def analyze(self, confirmed: dict[int, str] = None) -> Analysis:
        """
            Seed chosen from the readings, the words confirmed by the user
            ({position: word}) being fixed.
        """
        confirmed = confirmed or {}
        return Analysis([[(confirmed[k], 0.0)] if k in confirmed else w.candidates for k, w in enumerate(self.words)],
                        self.min_seed_probability)



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
            self.scores[(k, variant, model)] = decoder.scores(network.log_probs(ink))
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
        best = np.argsort(-scores, kind="stable")[:CANDIDATES]
        return WordReading(crop, [(words[i], float(scores[i])) for i in best])



def combine(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
        Sum of the log-probabilities of two readings of the same word.
    """
    def log_probabilities(candidates):
        top = candidates[0][1]
        total = top + math.log(sum(math.exp(s - top) for _, s in candidates))
        return {w: s - total for w, s in candidates}
    la, lb = log_probabilities(a), log_probabilities(b)
    combined = {w: la.get(w, FUSION_FLOOR) + lb.get(w, FUSION_FLOOR) for w in set(la) | set(lb)}
    return sorted(combined.items(), key=lambda item: -item[1])[:CANDIDATES]



def read_layout(reader: LayoutReader, report, weights: tuple[float, float], min_seed_probability: float) -> SeedReading:
    """
        Reads the words of `reader.layout` with `weights`; `report(fraction)` is
        called as the reading goes.
    """
    layout = reader.layout
    words = []
    for k in range(layout.count):
        report(0.9 * k / layout.count)
        words.append(reader.read(k, 0, weights))
    reading = SeedReading(layout.kind, words, min_seed_probability)

    # Re-reads of the least probable words, on their other crops
    if layout.variants > 1:
        least_probable = sorted(range(layout.count), key=lambda k: top_probability(words[k].candidates))
        for k in least_probable[:MAX_REREADS]:
            if reading.analyze().status == "valid":
                break
            report(0.9)
            others = [r for r in (reader.read(k, v, weights) for v in range(1, layout.variants)) if r is not None]
            if others:
                best = max((r.candidates for r in others), key=top_probability)
                words[k] = WordReading(words[k].image, combine(words[k].candidates, best))
                reading = SeedReading(layout.kind, words, min_seed_probability)

    # Two columns of equal length: column by column, or row by row if only
    # that order gives a valid seed
    for order in layout.orders[1:]:
        if reading.analyze().status == "invalid":
            alternative = SeedReading(layout.kind, [words[i] for i in order], min_seed_probability)
            if alternative.analyze().status != "invalid":
                reading = alternative
    return reading



def read_page(reader: LayoutReader, report, weights: tuple[float, float], min_seed_probability: float) -> SeedReading:
    """
        Reads `reader.layout` with `weights`, and its list reading if it is a
        card found on part of its lines that gives no valid seed (the better
        one is kept).
    """
    reading = read_layout(reader, lambda f: report(0.85 * f), weights, min_seed_probability)
    if reading.analyze().status != "valid" and reader.alternative():
        other = read_layout(reader.alternative(), lambda f: report(0.85 + 0.15 * f), weights, min_seed_probability)
        if STATUS_RANK[other.analyze().status] > STATUS_RANK[reading.analyze().status]:
            reading = other
    return reading



def read_seed(image: Image.Image, progress=None) -> SeedReading:
    """
        Reads the seed written on `image` (upright photo, full resolution).
        `progress(fraction)` is called as the reading goes. Raises WordsNotFound
        if no card or list with a valid number of words is found.
    """
    def report(fraction):
        if progress:
            progress(fraction)

    report(0.0)
    load_engine()
    reader = LayoutReader(find_words(image.convert("L")))
    reading = read_page(reader, lambda f: report(0.1 + 0.5 * f), FIRST, MIN_SEED_PROBABILITY)

    # Not valid: read again by the second network, then by the blend of both;
    # a reading is kept if its status is better, or as good with a more
    # probable seed
    best = reading.analyze()
    if best.status != "valid":
        second = ((LOWERCASE, LOWERCASE_MIN_SEED_PROBABILITY, 0.6), (BLEND, BLEND_MIN_SEED_PROBABILITY, 0.8))
        for weights, min_seed_probability, start in second:
            other = read_page(reader, lambda f, s=start: report(s + 0.2 * f), weights, min_seed_probability)
            analysis = other.analyze()
            if (STATUS_RANK[analysis.status], analysis.confidence) > (STATUS_RANK[best.status], best.confidence):
                reading, best = other, analysis
    report(1.0)
    return reading
