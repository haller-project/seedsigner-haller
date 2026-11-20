"""
    Dictionary decoding: CTC log-likelihood of every BIP39 word given the
    log-probabilities computed by the word recognizer, by the C library
    (native/handwriting.c).

    A prefix of blanks and digits is allowed before the word, so that a line
    number written before the word does not need to be removed. The words share
    a prefix tree (12 491 nodes instead of 24 184 CTC states). States that fall
    more than `beam` below the best state of the previous frame are dropped,
    which only affects words far behind the best one.
"""
import ctypes

import numpy as np
from embit.bip39 import WORDLIST

from . import native

DEFAULT_BEAM = 30.0



class Decoder:
    """
        `scores(log_probs)`: log-likelihood of every word of `words`, in order.
        Raises native.LibraryNotBuilt if the C library has not been built.
    """
    def __init__(self, words: list[str] = WORDLIST, beam: float = DEFAULT_BEAM):
        self.lib = native.load()
        self.words = list(words)
        self.beam = beam
        self.trie = native.trie(self.words)
        self.work = np.zeros(self.lib.hw_decoder_work_size(ctypes.byref(self.trie)) // 4, np.float32)


    def scores(self, log_probs: np.ndarray) -> np.ndarray:
        """
            log_probs: (NUM_CLASSES, frames) float32. Returns (len(words),) float32;
            words dropped by the beam get about -1e30.
        """
        lp = np.ascontiguousarray(log_probs, np.float32)
        out = np.empty(len(self.words), np.float32)
        self.lib.hw_word_scores(ctypes.byref(self.trie), lp.ctypes.data, lp.shape[1], self.beam,
                                self.work.ctypes.data, out.ctypes.data)
        return out



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
