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
