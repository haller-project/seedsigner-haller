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
import hashlib
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

# Words abbreviated to their first 4 letters (decoder.ABBREVIATION_LETTERS). Every word may be read abbreviated,
# with the a priori penalty decoder.ABBREVIATION_PENALTY, high so that words written in full read as before. A
# layout on which at least ABBREVIATED_MIN_WORDS words, and ABBREVIATED_SHARE of its words, read better as some
# abbreviation than as any word written in full by ABBREVIATION_EVIDENCE (decoder.abbreviation_evidence, first
# crops, networks of the reading) is an abbreviated list: its words are read again with ABBREVIATED_PAGE_PENALTY (the
# outputs of the networks and of the decoder are reused). Lists may mix both forms (words of 3 or 4 letters are always
# written in full). Chosen on the tuning pages only (real photos DEV and site, DEVR, FPDEV, ETIQDEV, 200 simulated
# abbreviated pages; 22/11/2026): an evidence of 0 lost 4 exact seeds of DEVR, 3 none.
ABBREVIATED_PAGE_PENALTY = 0.0
ABBREVIATION_EVIDENCE = 3.0
ABBREVIATED_MIN_WORDS = 3
ABBREVIATED_SHARE = 0.25

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
# A fingerprint hypothesis whose fingerprint matches no seed read: with HYPOTHESIS_KEPT, the list read without that
# line is kept as a reading whose fingerprint does not match (every word uncertain, TRUST_LOW), as for a line that
# passes the thresholds; without it, the list is not found. The line taken out is the only one
# left out of a list of one line too many, so that no other layout of the same words has a valid count (a grid of
# 24 words in 3 numbered columns, its fingerprint above it and misread, was not found; 22/11/2026).
# Frozen benches against 978f1fa: 46 exact seeds gained (1 real), none lost, no verified reading changed.
HYPOTHESIS_KEPT = True
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
# misread): every word is uncertain. FINGERPRINT_CANDIDATES is also the number
# of seeds compared with a fingerprint line over the whole reading of a photo,
# both searches of read_seed and every path included (FingerprintBudget): with
# the threshold of fingerprint.py, a wrong seed is verified by chance with a
# probability of at most 3 x 3.3e-7 = 1e-6 per photo (the right seed was among
# the first 3 on all but 6 of 1 698 pages with a fingerprint read,
# 21/11/2026). FINGERPRINT_COST: computation of one fingerprint on a Pi
# Zero (PBKDF2-HMAC-SHA512, 2048 rounds, 0.062 s; master key and HASH160,
# 0.005 s; measured with embit on 20/11/2026) and its check, seconds.
FINGERPRINT_CANDIDATES = 3
# Seeds of the first search compared at once with its fingerprint line, before a second search may be made (see
# FingerprintBudget): a match spares that search; the others are left for the end, with those of the second search.
# Chosen on the 19 frozen benches (22/11/2026): with 1 or 2, the same readings, verified and exact seeds on every page
# as with 3 seeds per search; with 1, 163 pages verified by their second or third seed are searched a second time
# before it is compared (+3 to +7 s on a Pi Zero), with 2, 18 pages (+3.8 s at most), none of them a real photo.
FIRST_SEARCH_CHECKS = 2
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
        abbreviated: candidates read in their abbreviated form (see
            decoder.ABBREVIATION_LETTERS)
    """
    def __init__(self, image: Image.Image, candidates: list[tuple[str, float]], abbreviated: frozenset = frozenset()):
        self.image = image
        self.candidates = candidates
        self.abbreviated = abbreviated


    def letters(self, word: str) -> int:
        """
            Number of letters of `word` as written: ABBREVIATION_LETTERS if it
            was read abbreviated.
        """
        from .decoder import ABBREVIATION_LETTERS
        return ABBREVIATION_LETTERS if word in self.abbreviated else len(word)



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
        if self.ambiguous_order and len(confirmed or {}) < len(self.words):
            # several orders of the lines give a plausible seed and neither the numbers nor the fingerprint chose one:
            # the probability of the seed measures its words only, not its order
            return Trust(TRUST_LOW)
        if analysis.status == "valid":
            return Trust(TRUST_HIGH)
        if analysis.status == "uncertain" and analysis.confidence >= MIN_MEDIUM_SEED_PROBABILITY:
            return Trust(TRUST_MEDIUM)
        return Trust(TRUST_LOW)



# Stages reported to Progress.on_stage while reading, for display only:
# "load" (word models), "find" (detection on the whole photo), "closer" (detection
# on the region of the document), "lines" (lines and columns), "numbers" (line
# numbers read, done/total), "order" (order
# told by the line numbers), "guard" (guards of a valid seed),
# "spare" (lines outside the words tried as a fingerprint, done), "rows" (rows at
# the start or end of a list tried as a label or a fingerprint, done), "words",
# "words2", "words3" (words read by the first network, the second, their blend:
# done/total, layout, k), "reread" (re-reading of the least probable words:
# done/total, layout, k), "fp_read" (fingerprint line read, done/total), "fp_check"
# (fingerprints of the candidate seeds computed, done/total). The start of an
# attempt is reported as "attempt:" + one of ATTEMPTS: "first" (first search),
# "list" (a card read as a list), "extra" (a list read without rows at its start
# or end), "crossed" (search without the words scribbled over).
ATTEMPTS = ("first", "list", "extra", "crossed")



class Progress:
    """
        Progress of a reading, reported as a fraction to `report(fraction)`.
        Each step done moves the fraction by its share of the cost (seconds on
        a Pi Zero) expected to remain: the progress is steady when the expected
        costs are right, and never goes back nor stops long when they are not.
    """
    def __init__(self, report, remaining: float, on_stage=None):
        self.report = report
        self.remaining = remaining
        self.fraction = 0.0
        self.on_stage = on_stage
        self.attempt = None
        report(0.0)


    def __call__(self, cost: float):
        """
            Same as step: a Progress is the `step` given to detector.find_words.
        """
        self.step(cost)


    def stage(self, name: str, done: int = None, total: int = None, layout=None, k: int = None):
        """
            What the reading does now, reported to `on_stage(name, done, total,
            layout, k)` (see STAGES) for display only: `done` of `total` (either
            may be None), `layout` and `k` the word concerned. Nothing of the
            reading depends on it.
        """
        if self.on_stage:
            self.on_stage(name, done, total, layout, k)


    def start_attempt(self, name: str):
        """
            A new hypothesis on the page is tried (see ATTEMPTS).
        """
        self.attempt = name
        self.stage("attempt:" + name)


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



# Word crops read during one reading of a photo (read_seed), by content: a crop read again by another search of the
# same photo (crossed out, extra lines, merged) reuses its ink map and the outputs of the networks, which depend on
# the crop alone. Emptied at the start and the end of read_seed.
WORD_CACHE = True
_crop_cache = {}   # key of a crop: ink map or None
_score_cache = {}  # (key, model): (outputs of the network, of the decoder (decoder.Decoder.raw), combined scores)
_fingerprint_cache = {}  # key of a crop: outputs of the fingerprint network


def fingerprint_log_probs(crop: Image.Image, ink: np.ndarray) -> np.ndarray:
    """
        Outputs of the fingerprint network on `ink`, the ink map of `crop`,
        computed once per crop during a reading (WORD_CACHE).
    """
    if not WORD_CACHE:
        return fingerprint_network().log_probs(ink)
    key = crop_key(crop)
    if key not in _fingerprint_cache:
        _fingerprint_cache[key] = fingerprint_network().log_probs(ink)
    return _fingerprint_cache[key]


def crop_key(crop: Image.Image) -> bytes:
    return hashlib.blake2b(crop.tobytes(), digest_size=16, person=b"%dx%d" % crop.size).digest()



class LayoutReader:
    """
        Reads the words of `layout` with either network or their blend; the
        scores of each crop are computed once per network.
    """
    def __init__(self, layout):
        self.layout = layout
        self.inks = {}     # (k, variant): (crop, ink map or None)
        self.scores = {}   # (k, variant, model): word scores
        self.leads = {}    # (k, variant, model): lead of the abbreviated forms (decoder.Decoder.forms)
        self.raw = {}      # (k, variant, model): outputs of the decoder (decoder.Decoder.raw)
        self.combined = {}  # (k, variant, model): {penalty: (scores, leads)}, shared through _score_cache
        self.penalty = None  # penalty of the abbreviated forms: decoder.ABBREVIATION_PENALTY, or that of a list
        #                      read as abbreviated (ABBREVIATED_PAGE_PENALTY, see abbreviated_list)
        self.log_probs = {}  # (k, variant, model): outputs of the network
        self.numbers = {}  # k: (number, probability) or None
        self.keys = {}     # (k, variant): key of the crop (WORD_CACHE)
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
            from .decoder import combine_forms
            cached = (self.keys[(k, variant)], model) if WORD_CACHE and (k, variant) in self.keys else None
            key = (k, variant, model)
            if key in self.raw:
                pass
            elif cached is not None and cached in _score_cache:
                self.log_probs[key], self.raw[key], self.combined[key] = _score_cache[cached]
            else:
                first, decoder, lowercase = load_engine()
                network = lowercase if model else first
                self.log_probs[key] = network.log_probs(ink)
                self.raw[key] = decoder.raw(self.log_probs[key])
                self.combined[key] = {}
                if cached is not None:
                    _score_cache[cached] = self.log_probs[key], self.raw[key], self.combined[key]
            if self.penalty not in self.combined[key]:
                self.combined[key][self.penalty] = combine_forms(*self.raw[key], self.penalty)
            self.scores[key], self.leads[key] = self.combined[key][self.penalty]
        return self.scores[(k, variant, model)]


    def abbreviated_list(self, weights: tuple[float, float]) -> bool:
        """
            True if the words of the layout, first crops, read with `weights`,
            make an abbreviated list (see ABBREVIATED_PAGE_PENALTY): the penalty
            of the abbreviated forms is then lowered for every later reading
            (the scores computed are dropped). Decided once per layout.
        """
        from .decoder import ABBREVIATIONS, abbreviation_evidence
        if self.penalty is not None or not ABBREVIATIONS:
            return False
        models = [m for m in (0, 1) if weights[m]]
        found = 0
        for k in range(self.layout.count):
            if self.model_scores(k, 0, models[0]) is None:
                continue
            evidence = np.mean([abbreviation_evidence(*self.raw[(k, 0, m)]) for m in models
                                if self.model_scores(k, 0, m) is not None])
            found += evidence >= ABBREVIATION_EVIDENCE
        if found < max(ABBREVIATED_MIN_WORDS, ABBREVIATED_SHARE * self.layout.count):
            return False
        self.penalty = ABBREVIATED_PAGE_PENALTY
        self.scores.clear()
        self.leads.clear()
        return True


    def read(self, k: int, variant: int, weights: tuple[float, float]) -> WordReading | None:
        """
            The CANDIDATES most likely words of crop (k, variant), the scores of
            the networks being weighted by `weights`; if no ink is found,
            CANDIDATES words with equal scores. None if there is no such crop.
        """
        words = load_engine()[1].words
        if (k, variant) not in self.inks:
            crop = self.layout.crop(k, variant)
            if crop is not None and WORD_CACHE:
                key = self.keys[(k, variant)] = crop_key(crop)
                if key not in _crop_cache:
                    _crop_cache[key] = normalize_word(crop)
                self.inks[(k, variant)] = (crop, _crop_cache[key])
            else:
                self.inks[(k, variant)] = (crop, None if crop is None else normalize_word(crop))
        crop, ink = self.inks[(k, variant)]
        if crop is None:
            return None
        if ink is None:
            return WordReading(crop, [(w, 0.0) for w in words[:CANDIDATES]])
        if weights[1] == 0:
            scores = self.model_scores(k, variant, 0)
            lead = self.leads[(k, variant, 0)]
        elif weights[0] == 0:
            scores = self.model_scores(k, variant, 1)
            lead = self.leads[(k, variant, 1)]
        else:
            scores = weights[0] * log_softmax(self.model_scores(k, variant, 0)) \
                + weights[1] * log_softmax(self.model_scores(k, variant, 1))
            lead = weights[0] * self.leads[(k, variant, 0)] + weights[1] * self.leads[(k, variant, 1)]
        scores = scores / TEMPERATURES[weights]
        best = np.argsort(-scores, kind="stable")[:CANDIDATES]
        return WordReading(crop, [(words[i], float(scores[i])) for i in best],
                           frozenset(words[i] for i in best if lead[i] > 0))



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
        seeds to `pool`, as (rank, words, word readings, log-probability of the words).
    """
    layout = reader.layout
    words = []

    def seed_reading(order):
        reading = SeedReading(layout.kind, [words[k] for k in order], min_seed_probability, layout.complete)
        reading.fingerprint_crop = getattr(layout, "fingerprint", None)
        reading.fingerprint_others = list(getattr(layout, "fingerprint_others", None) or [])
        reading.layout, reading.order = layout, list(order)
        return reading

    words_stage = "words" if weights == FIRST else "words2" if weights == LOWERCASE else "words3"
    for k in range(layout.count):
        progress.stage(words_stage, k + 1, layout.count, layout, k)
        words.append(reader.read(k, 0, weights))
        progress.step(WORD_COST if weights == FIRST else OTHER_WORD_COST)
        if on_word:
            on_word(layout, k, words[-1])
    if getattr(reader, "abbreviated_list", None) and reader.abbreviated_list(weights):   # (readers of the tests)
        for k in range(layout.count):
            words[k] = reader.read(k, 0, weights)
            if on_word:
                on_word(layout, k, words[k])
    reading = seed_reading(layout.orders[0])

    # Re-reads of the least probable words, on their other crops
    if layout.variants > 1:
        least_probable = sorted(range(layout.count), key=lambda k: top_probability(words[k].candidates))
        rereads = least_probable[:MAX_REREADS]
        for i, k in enumerate(rereads):
            if reading.analyze().status == "valid":
                break
            progress.stage("reread", i + 1, len(rereads), layout, k)
            others = [r for r in (reader.read(k, v, weights) for v in range(1, layout.variants)) if r is not None]
            progress.step(REREAD_COST)
            if others:
                best = max(others, key=lambda r: top_probability(r.candidates))
                words[k] = WordReading(words[k].image, combine(words[k].candidates, best.candidates),
                                       words[k].abbreviated | (best.abbreviated - {w for w, _ in words[k].candidates}))
                reading = seed_reading(layout.orders[0])

    # Two columns of equal length: column by column, or row by row if only
    # that order gives a valid seed. A permutation of a seed passes the
    # checksum once in 16 for 12 words: when both orders give a plausible
    # seed, the order written is unknown, unless the line numbers tell it.
    readings = [reading] + [seed_reading(order) for order in layout.orders[1:]]
    if pool is not None and getattr(layout, "fingerprint", None) is not None:
        for r in readings:
            analysis = r.analyze()
            for rank, (seed, log_p) in enumerate(zip(analysis.valid_seeds(FINGERPRINT_CANDIDATES),
                                                     analysis.valid_seed_log_probabilities(FINGERPRINT_CANDIDATES))):
                pool.append((rank, seed, r.words, log_p))
    if any(r.analyze().status != "invalid" for r in readings[1:]):
        progress.stage("order")
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
        previous = progress.attempt
        progress.start_attempt("list")
        other = read_layout(reader.alternative(), progress, weights, min_seed_probability, on_word, plausible, pool)
        progress.start_attempt(previous)
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



def fingerprint_margins(crop: Image.Image, on_network=None) -> list[tuple[float, float]] | None:
    """
        fingerprint.margins of the line on `crop` for the first and the
        second network, or None if no ink is found. `on_network()` is called
        before each network (display only).
    """
    ink = normalize_word(crop)
    if ink is None:
        return None
    first, decoder, lowercase = load_engine()
    found = []
    for network in (first, lowercase):
        if on_network is not None:
            on_network()
        log_probs = network.log_probs(ink)
        found.append(fingerprint.margins(log_probs, decoder.scores(log_probs)))
    return found



def is_fingerprint(crop: Image.Image, on_network=None) -> bool:
    """
        True if the line on `crop` reads as a fingerprint rather than a word
        (see FINGERPRINT_MARGIN). `on_network()` is called before each network
        (display only).
    """
    found = fingerprint_margins(crop, on_network)
    if fingerprint_accepted(found):
        return True
    if found is None or sum(m for m, _ in found) / len(found) < FINGERPRINT_RESCUE_MARGIN:
        return False
    # near the threshold: the fingerprint network has the last word, with the same thresholds
    if on_network is not None:
        on_network()
    _, decoder, _ = load_engine()
    log_probs = fingerprint_network().log_probs(normalize_word(crop))
    return fingerprint_accepted([fingerprint.margins(log_probs, decoder.scores(log_probs))])



def fingerprint_hypothesis_margin(crop: Image.Image, on_network=None) -> float | None:
    """
        Margin of the line on `crop` read as a fingerprint by the fingerprint
        network (fingerprint.margins), if it is at least
        LOOSE_FINGERPRINT_MARGIN with a fit of at least FINGERPRINT_FIT, else
        None. Used only when no line of a list with one line too many passes
        is_fingerprint: the line is then left out as a hypothesis, kept only
        if its fingerprint matches a seed read. `on_network()` is called before
        the network (display only).
    """
    ink = normalize_word(crop)
    if ink is None:
        return None
    _, decoder, _ = load_engine()
    if on_network is not None:
        on_network()
    log_probs = fingerprint_log_probs(crop, ink)
    margin, fit = fingerprint.margins(log_probs, decoder.scores(log_probs))
    return margin if margin >= LOOSE_FINGERPRINT_MARGIN and fit >= FINGERPRINT_FIT else None



def line_role(crops: list[Image.Image], on_network=None) -> tuple[str | None, Image.Image | None]:
    """
        What a row of a list left out of the words reads as, on its crops
        (widest first): ("fingerprint", crop) if a crop passes is_fingerprint,
        ("label", None) if one of the first two reads as a label (see
        LABEL_MARGIN), ("loose", crop) if a crop passes
        fingerprint_hypothesis_margin, ("other", None) if the first reads as
        no BIP39 word (NO_WORD_FIT), else (None, None). `on_network()` is
        called before each network (display only).
    """
    for crop in crops:
        if is_fingerprint(crop, on_network):
            return "fingerprint", crop
    first, decoder, lowercase = load_engine()
    other = False
    for i, crop in enumerate(crops[:2]):
        ink = normalize_word(crop)
        if ink is None:
            continue
        found = []
        for network in (first, lowercase):
            if on_network is not None:
                on_network()
            log_probs = network.log_probs(ink)
            found.append(fingerprint.label_margins(log_probs, decoder.scores(log_probs)))
        margin, fit, word_fit = (sum(f[j] for f in found) / len(found) for j in range(3))
        if margin >= LABEL_MARGIN and fit >= LABEL_FIT:
            return "label", None
        other = other or (i == 0 and word_fit <= NO_WORD_FIT)
    for crop in crops:
        if fingerprint_hypothesis_margin(crop, on_network) is not None:
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
        layout = find(gray, progress, fingerprint=fingerprint)
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
    per_letter = [lengths[k] / w.letters(w.candidates[0][0]) for k, w in zip(reading.order, reading.words)]
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



def read_seed(image: Image.Image, progress=None, on_word=None, on_stage=None) -> SeedReading:
    """
        Reads the seed written on `image` (upright photo, full resolution).
        `progress(fraction)` is called as the reading goes, at least every
        second or so on a Pi Zero; `on_word(layout, k, word_reading)` after the
        first reading of each word; `on_stage` receives what the reading does
        (see Progress.stage and STAGES), for display only. Raises WordsNotFound
        if no document with a valid number of words is found.

        The words are found by the detector (detector.find_words) and read;
        a valid seed then goes through the guards (see guard), and a
        fingerprint line written with the words is checked
        (check_fingerprint). A reading neither verified nor valid is tried
        again without the words that may be scribbled over (see
        detector.CROSSED_OUT); the second one is kept if it is verified, or
        else has a better status, or else a more probable seed.
    """
    progress = Progress(progress or (lambda fraction: None), detector.DETECTOR_COST + 12 * (WORD_COST + OTHER_WORD_COST),
                        on_stage)
    if not _engine:
        progress.stage("load")
        progress.expect(progress.remaining + LOAD_COST)
        load_engine()
        progress.step(LOAD_COST)
    gray = image.convert("L")
    detector._detections.clear()
    _crop_cache.clear()
    _score_cache.clear()
    _fingerprint_cache.clear()

    tried = [0, 0]  # readings of lines tried as a fingerprint, rows tried as a label or a fingerprint (shown only)

    def network_read():
        # each network reading a line tried as a fingerprint moves the counter (2-4s per line on a Pi Zero)
        tried[0] += 1
        progress.stage("spare", tried[0])

    def spare_line(crop):
        progress.step(WORD_COST + OTHER_WORD_COST)
        return is_fingerprint(crop, network_read)

    def hypothesis_margin(crop):
        progress.step(WORD_COST)
        tried[0] += 1
        progress.stage("spare", tried[0])
        return fingerprint_hypothesis_margin(crop)

    spare_line.margin = hypothesis_margin

    def row_read():
        # each network reading a row tried as a label or a fingerprint moves the counter
        tried[1] += 1
        progress.stage("rows", tried[1])

    def role(crops):
        progress.step(2 * WORD_COST)
        progress.stage("rows", tried[1] or None)
        return line_role(crops, row_read)

    spare_line.role = role

    budget = FingerprintBudget()
    # the first search compares its FIRST_SEARCH_CHECKS most probable seeds at once (a match spares the second
    # search), the second one none:
    # their other candidates are compared together at the end (see FingerprintBudget)
    now = {False: FIRST_SEARCH_CHECKS, True: 0}

    def attempt(crossed: bool):
        pool = []
        progress.start_attempt("crossed" if crossed else "first")

        def find(image, step, fingerprint):
            return detector.find_words(image, step, fingerprint=fingerprint, crossed=crossed)

        reading, _, not_found = search(find, gray, progress, on_word, pool, spare_line)
        if not crossed and reading is not None and getattr(reading.layout, "numbers_incoherent", False) \
                and not getattr(reading.layout, "cards_held", True):
            # no card kernel holds words: the list read without them is the same one
            reading.numbers_incoherent = True
            reading.mark_disputed()
        elif not crossed and reading is not None and getattr(reading.layout, "numbers_incoherent", False):
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
            progress.start_attempt("extra")

            def find_extra(image, step, fingerprint):
                return detector.find_words(image, step, fingerprint=fingerprint, extra=True)

            other, _, _ = search(find_extra, gray, progress, on_word, extra_pool, spare_line)
            if other is not None:
                if other.analyze().status == "valid":
                    progress.stage("guard")
                    guard(other)
                if other.fingerprint_crop is not None:
                    other = check_fingerprint(other, extra_pool, progress, budget=budget, now=now[crossed])
                if getattr(other, "verified", False) \
                        or reading is None and not other.layout.fingerprint_hypothesis:
                    return other, None
                if reading is not None and other.layout.extra_lines_strong:
                    # rows read as a fingerprint (strict thresholds) or a label, not verified without them
                    reading.extra_lines = True
            progress.start_attempt("first")
        if reading is None:
            return None, not_found
        if reading.analyze().status == "valid":
            progress.stage("guard")
            guard(reading)
        if reading.fingerprint_crop is not None and getattr(reading.layout, "fingerprint_optional", False):
            # a row left out across the columns tried as the fingerprint: kept only if it matches a seed read
            doubtful = set(reading.doubtful)
            # a single line outside the rows across the columns (detector.FINGERPRINT_OPTIONAL_ANY): read on its crop
            # only, whose outputs were computed for the loose margin
            checked = check_fingerprint(reading, pool, progress,
                                        trim=not getattr(reading.layout, "fingerprint_single", False),
                                        budget=budget, now=now[crossed])
            if getattr(checked, "verified", False):
                reading = checked
            else:
                reading.doubtful, reading._last = doubtful, None
                reading.fingerprint_crop, reading.fingerprint = None, None
        elif reading.fingerprint_crop is not None:
            reading = check_fingerprint(reading, pool, progress, budget=budget, now=now[crossed])
        if not HYPOTHESIS_KEPT \
                and getattr(reading.layout if hasattr(reading, "layout") else None, "fingerprint_hypothesis", False) \
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
        verified = resolve_fingerprints(budget, progress)
        if verified is not None:
            reading = verified
    finally:
        detector._detections.clear()
        _crop_cache.clear()
        _score_cache.clear()
        _fingerprint_cache.clear()
    if reading is None:
        raise not_found
    progress.report(1.0)
    return reading



class FingerprintBudget:
    """
        Seeds checked against a fingerprint line during one reading of a photo
        (read_seed): at most FINGERPRINT_CANDIDATES (seed, line) pairs in all,
        over every search and every path. Each pair verifies a wrong seed by
        chance with a probability of at most 1 / (MIN_MATCH_PROBABILITY * 2^32)
        (fingerprint.py), so that a photo does with at most
        FINGERPRINT_CANDIDATES times that.

        A check made while a second search may follow (`now`) compares its
        `now` most probable seeds only (FIRST_SEARCH_CHECKS); its other
        candidates are `pending`. If that
        check had matched, the second search would not have been made: a
        candidate compared then could stop it, and the pairs that may verify
        a photo would be those of both searches. The pending candidates of
        both searches are compared at the end (resolve), the most probable
        ones first, within what is left of the budget.
    """
    def __init__(self, total: int = None):
        self.left = FINGERPRINT_CANDIDATES if total is None else total
        self.pending = []
        # (seed, line) pairs already compared: a pending one is not compared again
        self.compared = set()



class PendingCheck:
    """
        Candidates of a fingerprint check left for FingerprintBudget.resolve,
        with what the reading verified by one of them is made of (taken when
        the check was made: the reading may change afterwards).
    """
    def __init__(self, reading: SeedReading, check, candidates: list, line):
        self.check = check
        self.line = line
        self.candidates = candidates
        self.kind = reading.kind
        self.min_seed_probability = reading.min_seed_probability
        self.numbered = reading.numbered
        self.fingerprint_crop = reading.fingerprint_crop
        self.fingerprint_others = list(getattr(reading, "fingerprint_others", []))


    def verified_reading(self, seed: list[str], words: list) -> SeedReading:
        verified = SeedReading(self.kind, [WordReading(w.image, [(word, 0.0)], w.abbreviated & {word})
                                           for word, w in zip(seed, words)],
                               self.min_seed_probability)
        verified.fingerprint_crop = self.fingerprint_crop
        verified.fingerprint_others = list(self.fingerprint_others)
        verified.fingerprint = self.check
        verified.verified = True
        verified.numbered = self.numbered
        return verified



def fingerprint_candidates(pool: list) -> list:
    """
        Distinct seeds of `pool` (see read_layout), by rank (the most probable
        seed of every reading first), as (seed, word readings, log-probability).
    """
    candidates, seen = [], set()
    for _, seed, words, log_p in sorted(pool, key=lambda item: item[0]):
        if tuple(seed) not in seen:
            seen.add(tuple(seed))
            candidates.append((seed, words, log_p))
    return candidates



def match_fingerprints(check, group: list, progress: Progress, offset: int = 0, total: int = None):
    """
        Fingerprints of the seeds of `group` (seed, word readings, ...)
        compared with the line of `check`, in one pass of the networks'
        outputs: the index of the first one that matches, or None. Each seed
        is reported as stage "fp_check", numbered from `offset` + 1 out of
        `total` (default: the size of `group`), for display only.
    """
    fingerprints = []
    for candidate in group:
        progress.stage("fp_check", offset + len(fingerprints) + 1, total or len(group))
        fingerprints.append(fingerprint.seed_fingerprint(candidate[0]))
        progress.step(FINGERPRINT_COST)
    check.checked += len(group)
    for i, (fp, probability) in enumerate(zip(fingerprints, check.probabilities(fingerprints))):
        check.match_probability = max(check.match_probability, probability)
        if probability >= fingerprint.MIN_MATCH_PROBABILITY:
            check.matched, check.rank = fp, check.checked - len(group) + i
            return i
    return None



def check_fingerprint(reading: SeedReading, pool: list, progress: Progress, trim: bool = True,
                      budget: FingerprintBudget = None, now: int = None) -> SeedReading:
    """
        Checks the fingerprint line of `reading`, read by the fingerprint
        network (see FINGERPRINT_CANDIDATES)
        against the candidate seeds of `pool` (see read_layout), most probable
        first. Returns the reading of the matching seed, every word certain
        (`verified`), or `reading` with every word uncertain; the check is in
        `fingerprint` of the reading returned. Without `trim`, the line is
        read on its crop only (not also with FINGERPRINT_TRIM off its top and
        bottom).

        With `budget`, the seeds compared come out of it; with `now` as well,
        at most `now` of them are compared here and the others are left in
        `budget.pending` (see FingerprintBudget), the reading being returned
        as if none of them matched.
    """
    progress.stage("fp_read")
    crops = []
    for crop in [reading.fingerprint_crop] + list(getattr(reading, "fingerprint_others", [])):
        # two fingerprint lines (with and without a passphrase): the mean of their probabilities also sums to 1
        w, h = crop.size
        cut = round(FINGERPRINT_TRIM * h) if trim else 0
        crops += [crop] + ([crop.crop((0, cut, w, h)), crop.crop((0, 0, w, h - cut))] if cut else [])
    inks = []
    for i, c in enumerate(crops):
        progress.stage("fp_read", i + 1, len(crops))
        ink = normalize_word(c)
        if ink is not None:
            inks.append((c, ink))
    if not inks:
        return reading
    readings = []
    for i, (c, ink) in enumerate(inks):
        progress.stage("fp_read", i + 1, len(inks))
        readings.append(fingerprint.FingerprintReading(fingerprint_log_probs(c, ink)))
    check = fingerprint.FingerprintCheck(readings)
    budget = budget or FingerprintBudget()
    line = tuple((c.size, hash(c.tobytes())) for c in crops)
    candidates = [c for c in fingerprint_candidates(pool) if (tuple(c[0]), line) not in budget.compared]
    count = min(len(candidates), budget.left if now is None else min(now, budget.left))
    later = candidates[count:]
    candidates = candidates[:count]
    budget.left -= count
    budget.compared.update((tuple(c[0]), line) for c in candidates)
    pending = PendingCheck(reading, check, later, line)
    progress.expect(len(candidates) * FINGERPRINT_COST)
    matched = None
    # the chosen seed first, then the others together (one pass of the networks' outputs for all of them)
    for offset, group in ((0, candidates[:1]), (1, candidates[1:])):
        if matched or not group:
            continue
        i = match_fingerprints(check, group, progress, offset, len(candidates))
        if i is not None:
            matched = group[i]
    if matched is not None:
        check.status = "verified"
        return pending.verified_reading(matched[0], matched[1])
    if now is not None and later and budget.left:
        budget.pending.append(pending)
    check.status = "mismatch"
    reading.mark_fingerprint_mismatch()
    reading.fingerprint = check
    return reading



def resolve_fingerprints(budget: FingerprintBudget, progress: Progress):
    """
        Compares the pending candidates of `budget` (see FingerprintBudget)
        within what is left of it: those of a single check in its order (as
        if compared when it was made), those of several checks the most
        probable first (log-probability of their words). Returns the reading
        verified by the first one that matches, or None.
    """
    entries = [(-log_p, n, j, pending, (seed, words))
               for n, pending in enumerate(budget.pending)
               for j, (seed, words, log_p) in enumerate(pending.candidates)]
    if len(budget.pending) > 1:
        entries.sort(key=lambda e: e[:3])
    chosen = []
    for e in entries:
        key = (tuple(e[4][0]), e[3].line)
        if key not in budget.compared and len(chosen) < budget.left:
            budget.compared.add(key)
            chosen.append(e)
    budget.left -= len(chosen)
    budget.pending = []
    if not chosen:
        return None
    progress.expect(len(chosen) * FINGERPRINT_COST)
    first = None
    for pending in dict.fromkeys(e[3] for e in chosen):
        group = [e for e in chosen if e[3] is pending]
        i = match_fingerprints(pending.check, [e[4] for e in group], progress)
        if i is not None and (first is None or chosen.index(group[i]) < chosen.index(first)):
            first = group[i]
    if first is None:
        return None
    pending = first[3]
    pending.check.status = "verified"
    return pending.verified_reading(*first[4])
