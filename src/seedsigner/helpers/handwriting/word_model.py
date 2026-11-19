"""
    Word recognizer: a small convolutional network computed in 16-bit integer
    arithmetic by the C library (native/handwriting.c).

    Input: ink map of INPUT_HEIGHT x width (preprocess.normalize_word).
    Output: for every 4 columns, log-probabilities of NUM_CLASSES classes: 0 is
    the CTC blank, 1-26 the letters a-z, 27-36 the digits 0-9.

    The quantized weights of the Haller-1 model (resources/handwriting/
    haller-1-*.npz) are written by haller.quantize, which also documents the
    format, together with reference vectors (haller-1-vectors.json) that the C
    library must reproduce bit for bit (tests/test_handwriting_vectors.py).
"""
import ctypes
import os

import numpy as np

from . import native
from .native import ALPHABET, FIRST_DIGIT_CLASS, NUM_CLASSES  # noqa: F401
from .preprocess import INPUT_HEIGHT, MAX_INPUT_WIDTH

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "handwriting",
                          "haller-1-capitals.npz")



class WordModel:
    """
        Loads the quantized network once; `log_probs(ink)` reads one word.
        Raises native.LibraryNotBuilt if the C library has not been built.
    """
    def __init__(self, path: str = MODEL_PATH):
        self.lib = native.load()
        with np.load(path) as data:
            self.network = native.network({k: np.ascontiguousarray(data[k]) for k in data.files})
        self.work = np.zeros(0, np.int32)


    def logits(self, ink: np.ndarray) -> np.ndarray:
        """
            (NUM_CLASSES, frames) float32 logits of an ink map (uint8, INPUT_HEIGHT x width).
        """
        ink = np.ascontiguousarray(ink, np.uint8)
        if ink.shape[0] != INPUT_HEIGHT or not 0 < ink.shape[1] <= MAX_INPUT_WIDTH:
            raise ValueError(f"ink map of shape {ink.shape}")
        width = ink.shape[1]
        size = self.lib.hw_network_work_size(ctypes.byref(self.network), width)
        if self.work.nbytes < size:
            self.work = np.zeros(-(-size // 4), np.int32)
        out = np.empty((NUM_CLASSES, max(1, width // 4)), np.float32)
        self.lib.hw_network_logits(ctypes.byref(self.network), ink.ctypes.data, width, self.work.ctypes.data,
                                   out.ctypes.data)
        return out


    def log_probs(self, ink: np.ndarray) -> np.ndarray:
        """
            (NUM_CLASSES, frames) float32 log-probabilities of an ink map.
        """
        z = self.logits(ink)
        self.lib.hw_log_softmax(z.ctypes.data, z.shape[1])
        return z
