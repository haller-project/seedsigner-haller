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



def read_word(image: Image.Image, model=None) -> list[tuple[str, float]]:
    """
        The CANDIDATES most likely words written on `image` according to
        `model` (default: the first word model), with their log-likelihood; if
        no ink is found, CANDIDATES words with equal scores.
    """
    first, decoder, _ = load_engine()
    model = model or first
    ink = normalize_word(image)
    if ink is None:
        return [(w, 0.0) for w in decoder.words[:CANDIDATES]]
    scores = decoder.scores(model.log_probs(ink))
    best = np.argsort(-scores)[:CANDIDATES]
    return [(decoder.words[i], float(scores[i])) for i in best]



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



def read_layout(layout, report, model=None, min_seed_probability: float = MIN_SEED_PROBABILITY) -> SeedReading:
    """
        Reads the words of `layout` with `model`; `report(fraction)` is called
        as the reading goes.
    """
    words = []
    for k in range(layout.count):
        report(0.9 * k / layout.count)
        crop = layout.crop(k)
        words.append(WordReading(crop, read_word(crop, model)))
    reading = SeedReading(layout.kind, words, min_seed_probability)

    # Re-reads of the least probable words, on their other crops
    if layout.variants > 1:
        least_probable = sorted(range(layout.count), key=lambda k: top_probability(words[k].candidates))
        for k in least_probable[:MAX_REREADS]:
            if reading.analyze().status == "valid":
                break
            report(0.9)
            crops = [c for c in (layout.crop(k, v) for v in range(1, layout.variants)) if c is not None]
            if crops:
                best = max((read_word(c, model) for c in crops), key=top_probability)
                words[k].candidates = combine(words[k].candidates, best)

    # Two columns of equal length: column by column, or row by row if only
    # that order gives a valid seed
    for order in layout.orders[1:]:
        if reading.analyze().status == "invalid":
            alternative = SeedReading(layout.kind, [words[i] for i in order], min_seed_probability)
            if alternative.analyze().status != "invalid":
                reading = alternative
    return reading



def read_page(layout, report, model=None, min_seed_probability: float = MIN_SEED_PROBABILITY) -> SeedReading:
    """
        Reads `layout` with `model`, and its list reading if it is a card found
        on part of its lines that gives no valid seed (the better one is kept).
    """
    reading = read_layout(layout, lambda f: report(0.85 * f), model, min_seed_probability)
    if layout.alternative and reading.analyze().status != "valid":
        try:
            other = read_layout(layout.alternative(), lambda f: report(0.85 + 0.15 * f), model, min_seed_probability)
        except WordsNotFound:
            other = None
        if other and STATUS_RANK[other.analyze().status] > STATUS_RANK[reading.analyze().status]:
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
    _, _, lowercase = load_engine()
    layout = find_words(image.convert("L"))
    reading = read_page(layout, lambda f: report(0.1 + 0.5 * f))

    # Not valid: read again by the second model, kept if its status is better,
    # or as good with a more probable seed
    first = reading.analyze()
    if first.status != "valid":
        other = read_page(layout, lambda f: report(0.6 + 0.4 * f), lowercase, LOWERCASE_MIN_SEED_PROBABILITY)
        second = other.analyze()
        if (STATUS_RANK[second.status], second.confidence) > (STATUS_RANK[first.status], first.confidence):
            reading = other
    report(1.0)
    return reading
