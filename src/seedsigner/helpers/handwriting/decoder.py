"""
    Dictionary decoding: CTC log-likelihood of every BIP39 word given the
    log-probabilities computed by the word recognizer, by the C library
    (native/handwriting.c).

    A prefix of blanks and digits is allowed before the word, so that a line
    number written before the word does not need to be removed. The words share
    a prefix tree (12 491 nodes instead of 24 184 CTC states). States that fall
    more than `beam` below the best state of the previous frame are dropped,
    which only affects words far behind the best one.

    Each word of more than ABBREVIATION_LETTERS letters may also be written
    abbreviated to its first ABBREVIATION_LETTERS letters, as on many metal
    and card backups: the English BIP39 list is built so that these letters
    identify the word (the 2 048 forms are distinct, and no abbreviated form is
    another word). The abbreviated forms are nodes of the same prefix tree, so
    that they cost no state of the decoder. The score of such a word is the sum
    of the likelihoods of both forms, the abbreviated one weighted by
    exp(-penalty): an a priori odds, ABBREVIATION_PENALTY by default, high so
    that words written in full read as before; a list read as abbreviated gets
    a lower one (see handwriting.ABBREVIATED_PAGE_PENALTY).
"""
import ctypes

import numpy as np
from embit.bip39 import WORDLIST

from . import native

DEFAULT_BEAM = 30.0

# Abbreviated forms (see above); ABBREVIATIONS False reads full words only.
ABBREVIATIONS = True
ABBREVIATION_LETTERS = 4
ABBREVIATION_PENALTY = 9.0



class Decoder:
    """
        `scores(log_probs)`: log-likelihood of every word of `words`, in order,
        written in full or abbreviated; `forms(log_probs)`: the same and the
        lead of the abbreviated form over the full one.
        Raises native.LibraryNotBuilt if the C library has not been built.
    """
    def __init__(self, words: list[str] = WORDLIST, beam: float = DEFAULT_BEAM):
        self.lib = native.load()
        self.words = list(words)
        self.beam = beam
        self.abbreviable = np.array([i for i, w in enumerate(self.words) if len(w) > ABBREVIATION_LETTERS], np.intp)
        abbreviated = [self.words[i][:ABBREVIATION_LETTERS] for i in self.abbreviable]
        self.trie = native.trie(self.words + abbreviated)
        self.work = np.zeros(self.lib.hw_decoder_work_size(ctypes.byref(self.trie)) // 4, np.float32)


    def raw(self, log_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
            log_probs: (NUM_CLASSES, frames) float32. Returns the log-likelihood
            of the full form of every word, and that of its abbreviated form
            (-inf for words that have none), both (len(words),) float32.
        """
        lp = np.ascontiguousarray(log_probs, np.float32)
        n = len(self.words)
        out = np.empty(n + len(self.abbreviable), np.float32)
        self.lib.hw_word_scores(ctypes.byref(self.trie), lp.ctypes.data, lp.shape[1], self.beam,
                                self.work.ctypes.data, out.ctypes.data)
        short = np.full(n, -np.inf, np.float32)
        short[self.abbreviable] = out[n:]
        return out[:n].copy(), short


    def forms(self, log_probs: np.ndarray, penalty: float = None) -> tuple[np.ndarray, np.ndarray]:
        """
            log_probs: (NUM_CLASSES, frames) float32. Returns the scores of
            `scores` and, for every word, the log-likelihood of its abbreviated
            form, penalized, less that of its full form (-inf for words that
            have none, or when ABBREVIATIONS is False): positive when the word
            reads as abbreviated. Both (len(words),) float32. `penalty`:
            ABBREVIATION_PENALTY by default.
        """
        return combine_forms(*self.raw(log_probs), penalty)


    def scores(self, log_probs: np.ndarray) -> np.ndarray:
        """
            log_probs: (NUM_CLASSES, frames) float32. Returns (len(words),) float32;
            words dropped by the beam get about -1e30.
        """
        return self.forms(log_probs)[0]



def combine_forms(full: np.ndarray, short: np.ndarray, penalty: float = None) -> tuple[np.ndarray, np.ndarray]:
    """
        Scores and leads of Decoder.forms from the outputs of Decoder.raw, the
        abbreviated forms weighted by exp(-penalty) (ABBREVIATION_PENALTY by
        default).
    """
    full = full.copy()
    lead = np.full(len(full), -np.inf, np.float32)
    if ABBREVIATIONS:
        some = np.isfinite(short)
        penalized = short[some] - np.float32(ABBREVIATION_PENALTY if penalty is None else penalty)
        lead[some] = penalized - full[some]
        full[some] = np.logaddexp(full[some], penalized)
    return full, lead



def abbreviation_evidence(full: np.ndarray, short: np.ndarray) -> float:
    """
        Log-likelihood of the best abbreviated form less that of the best full
        word (outputs of Decoder.raw): positive when the word reads better as
        some abbreviation than as any word written in full.
    """
    return float(short.max() - full.max())



def number_probabilities(log_probs: np.ndarray, count: int) -> np.ndarray:
    """
        Probability that the digits read on an ink map (blank, letters and
        punctuation being ignored) are exactly those of each number 1 to
        `count` (array of `count` values): the sum over every frame alignment of
        the network outputs, log_probs (NUM_CLASSES, frames). The probabilities
        of all digit strings, the empty one included, sum to 1, so that a
        number is only probable if no other digits are. Numbers of one or two
        digits.
    """
    p = np.exp(log_probs.astype(np.float64))
    digits = p[native.FIRST_DIGIT_CLASS:]                   # (10, frames)
    other = p.sum(0) - digits.sum(0)                        # blank and letters
    numbers = np.arange(1, count + 1)
    two = numbers >= 10
    first, second = np.where(two, numbers // 10, numbers), numbers % 10
    repeat = two & (first == second)
    start = 1.0
    run1, gap1, run2, gap2 = (np.zeros(count) for _ in range(4))  # in or after the run of the first, second digit
    for t in range(p.shape[1]):
        d, o = digits[:, t], other[t]
        run1, gap1, run2, gap2, start = (
            (start + run1) * d[first],
            (gap1 + run1) * o,
            np.where(two, (run2 + gap1 + np.where(repeat, 0.0, run1)) * d[second], 0.0),
            (gap2 + run2) * o,
            start * o)
    return np.where(two, run2 + gap2, run1 + gap1)
