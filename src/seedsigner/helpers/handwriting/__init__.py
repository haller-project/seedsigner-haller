"""
    Reading of a seed handwritten on paper, from a photo: the official
    SeedSigner 12- or 24-word card, or a free list of 12 to 24 words (blank or
    ruled paper, one or two columns, with or without numbers).

        reading = read_seed(image, progress)
        analysis = reading.analyze(confirmed)

    Each word is cut out of the photo (page.py), normalized (preprocess.py),
    read by a small neural network (word_model.py) and scored against the 2 048
    BIP39 words (decoder.py); the checksum then chooses the seed among the
    plausible readings and flags uncertain words (checksum.py). The result must
    always be reviewed by the user against the paper.
"""
import math
import threading

import numpy as np
from PIL import Image

from .checksum import Analysis
from .page import WordsNotFound, find_words
from .preprocess import normalize_word

# Readings kept per word (the following ones are negligible)
CANDIDATES = 30

# While the seed is not valid, at most MAX_REREADS words, least probable
# first, are read again on their other crop variants; both readings are then
# combined, a word missing from one of them getting FUSION_FLOOR there
MAX_REREADS = 4
FUSION_FLOOR = math.log(1e-4)

STATUS_RANK = {"invalid": 0, "uncertain": 1, "valid": 2}

_engine = []
_engine_lock = threading.Lock()



def load_engine():
    """
        Loads the word model and the dictionary decoder once (3 to 4 seconds on
        a Pi Zero). Their work buffers are shared: one reading at a time.
    """
    with _engine_lock:
        if not _engine:
            from .decoder import Decoder
            from .word_model import WordModel
            _engine.extend((WordModel(), Decoder()))
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
    """
    def __init__(self, kind: str, words: list[WordReading]):
        self.kind = kind
        self.words = words


    def analyze(self, confirmed: dict[int, str] = None) -> Analysis:
        """
            Seed chosen from the readings, the words confirmed by the user
            ({position: word}) being fixed.
        """
        confirmed = confirmed or {}
        return Analysis([[(confirmed[k], 0.0)] if k in confirmed else w.candidates for k, w in enumerate(self.words)])



def top_probability(candidates: list[tuple[str, float]]) -> float:
    """
        Probability of the most likely word among the candidates.
    """
    return 1 / sum(math.exp(s - candidates[0][1]) for _, s in candidates)



def read_word(image: Image.Image) -> list[tuple[str, float]]:
    """
        The CANDIDATES most likely words written on `image`, with their
        log-likelihood; if no ink is found, CANDIDATES words with equal scores.
    """
    model, decoder = load_engine()
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



def read_layout(layout, report) -> SeedReading:
    """
        Reads the words of `layout`; `report(fraction)` is called as the reading
        goes.
    """
    words = []
    for k in range(layout.count):
        report(0.9 * k / layout.count)
        crop = layout.crop(k)
        words.append(WordReading(crop, read_word(crop)))
    reading = SeedReading(layout.kind, words)

    # Re-reads of the least probable words, on their other crops
    if layout.variants > 1:
        least_probable = sorted(range(layout.count), key=lambda k: top_probability(words[k].candidates))
        for k in least_probable[:MAX_REREADS]:
            if reading.analyze().status == "valid":
                break
            report(0.9)
            crops = [c for c in (layout.crop(k, v) for v in range(1, layout.variants)) if c is not None]
            if crops:
                best = max((read_word(c) for c in crops), key=top_probability)
                words[k].candidates = combine(words[k].candidates, best)

    # Two columns of equal length: column by column, or row by row if only
    # that order gives a valid seed
    for order in layout.orders[1:]:
        if reading.analyze().status == "invalid":
            alternative = SeedReading(layout.kind, [words[i] for i in order])
            if alternative.analyze().status != "invalid":
                reading = alternative
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
    layout = find_words(image.convert("L"))
    reading = read_layout(layout, lambda f: report(0.1 + 0.8 * f))

    # A card found on part of its lines and giving no valid seed: the list
    # reading of the photo replaces it if it ranks better
    if layout.alternative and reading.analyze().status != "valid":
        try:
            other = read_layout(layout.alternative(), lambda f: report(0.85 + 0.15 * f))
        except WordsNotFound:
            other = None
        if other and STATUS_RANK[other.analyze().status] > STATUS_RANK[reading.analyze().status]:
            reading = other
    report(1.0)
    return reading
