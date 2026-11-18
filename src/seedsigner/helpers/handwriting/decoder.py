"""
    Dictionary decoding: CTC log-likelihood of every BIP39 word given the
    log-probabilities computed by the word recognizer.

    A prefix of blanks and digits is allowed before the word, so that a line
    number written before the word does not need to be removed. The words share
    a prefix tree (12 491 nodes instead of 24 184 CTC states). States that fall
    more than `beam` below the best state of the previous frame are dropped,
    which only affects words far behind the best one.

    The C library (native/handwriting.c) is used when it has been built; the
    numpy code below gives the same scores within about 2e-3 (the C code uses an
    interpolated table for the log-add).
"""
import ctypes

import numpy as np
from embit.bip39 import WORDLIST

from . import native
from .word_model import ALPHABET, FIRST_DIGIT_CLASS

DEFAULT_BEAM = 30.0
NEG = np.float32(-1e30)



def build_trie(words: list[str]) -> dict[str, np.ndarray]:
    """
        Prefix tree of `words`, nodes stored parent before child (node 0: root):
        parent and letter (class index) of each node, last node of each word.
    """
    index = {"": 0}
    parent, letter = [0], [0]
    for prefix in sorted({w[:i] for w in words for i in range(1, len(w) + 1)}, key=lambda s: (len(s), s)):
        index[prefix] = len(parent)
        parent.append(index[prefix[:-1]])
        letter.append(ALPHABET.index(prefix[-1]) + 1)
    return dict(parent=np.array(parent, np.int32), letter=np.array(letter, np.int16),
                word_end=np.array([index[w] for w in words], np.int32))



class Decoder:
    """
        `scores(log_probs)`: log-likelihood of every word of `words`, in order.
    """
    def __init__(self, words: list[str] = WORDLIST, beam: float = DEFAULT_BEAM, use_native: bool = True):
        self.words = list(words)
        self.beam = beam
        self.trie = build_trie(self.words)
        self.lib = native.load() if use_native else None
        if self.lib is not None:
            t = self.trie
            self.c_trie = native.Trie(len(t["parent"]), len(t["word_end"]), t["parent"].ctypes.data,
                                      t["letter"].ctypes.data, t["word_end"].ctypes.data)
            self.work = np.zeros(self.lib.hw_decoder_work_size(ctypes.byref(self.c_trie)) // 4, np.float32)


    def scores(self, log_probs: np.ndarray) -> np.ndarray:
        """
            log_probs: (NUM_CLASSES, frames) float32. Returns (len(words),) float32;
            words dropped by the beam get about -1e30.
        """
        lp = np.ascontiguousarray(log_probs, np.float32)
        if self.lib is None:
            return self._scores_numpy(lp)
        out = np.empty(len(self.words), np.float32)
        self.lib.hw_word_scores(ctypes.byref(self.c_trie), lp.ctypes.data, lp.shape[1], self.beam,
                                self.work.ctypes.data, out.ctypes.data)
        return out


    def _scores_numpy(self, lp: np.ndarray) -> np.ndarray:
        """
            Same recursion as hw_word_scores, all nodes of a frame at once.
        """
        parent, letter = self.trie["parent"][1:], self.trie["letter"][1:]
        from_root = parent == 0
        same_letter = self.trie["letter"][parent] == letter
        emit = lp[letter]
        prefix = np.logaddexp.reduce(np.concatenate([lp[:1], lp[FIRST_DIGIT_CLASS:]]), axis=0)
        n = len(self.trie["parent"])
        L, B = np.full(n, NEG), np.full(n, NEG)
        L[1:] = np.where(from_root, emit[:, 0], NEG)
        P = prefix[0]
        best = max(P, L.max())
        for t in range(1, lp.shape[1]):
            floor = best - self.beam if self.beam > 0 else NEG
            Lp, Bp = L[parent], B[parent]
            v = np.where(from_root, np.logaddexp(L[1:], P), np.logaddexp(L[1:], Bp))
            v = np.where(from_root | same_letter, v, np.logaddexp(v, Lp))
            new_L, new_B = np.full(n, NEG), np.full(n, NEG)
            new_L[1:] = v + emit[:, t]
            new_B[1:] = np.logaddexp(B[1:], L[1:]) + lp[0, t]
            dropped = (L[1:] < floor) & (B[1:] < floor) & np.where(from_root, P < floor, (Lp < floor) & (Bp < floor))
            new_L[1:][dropped] = NEG
            new_B[1:][dropped] = NEG
            P = P + prefix[t]
            L, B = new_L, new_B
            best = max(P, L.max(), B.max())
        end = self.trie["word_end"]
        return np.logaddexp(L[end], B[end]).astype(np.float32)
