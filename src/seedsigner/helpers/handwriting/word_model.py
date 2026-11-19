"""
    Word recognizer: a small convolutional network computed in 16-bit integer
    arithmetic.

    Input: ink map of INPUT_HEIGHT x width (preprocess.normalize_word).
    Output: for every 4 columns, log-probabilities of 37 classes: 0 is the CTC
    blank, 1-26 the letters a-z, 27-36 the digits 0-9.

    The quantized weights of the Haller-1 model (resources/handwriting/
    haller-1-*.npz) are written by haller.quantize, which also documents the
    format. The C library (native/handwriting.c) is used when it has been
    built; the numpy code below computes bit-identical logits.
"""
import ctypes
import os

import numpy as np

from . import native
from .preprocess import INPUT_HEIGHT, MAX_INPUT_WIDTH

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "handwriting",
                          "haller-1-capitals.npz")

LAYERS = ("b1", "b2", "b3", "b4", "t1", "t2")

# Output classes: 0 is the CTC blank, then the letters and the digits
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
NUM_CLASSES = 1 + len(ALPHABET)
FIRST_DIGIT_CLASS = 1 + ALPHABET.index("0")

# Activations are 12-bit integers; requantization multipliers have SHIFT fractional bits
AMAX = 2047
SHIFT = 16



def frames(width: int) -> int:
    return max(1, width // 4)



def requantize(acc: np.ndarray, mul: np.ndarray) -> np.ndarray:
    """
        ReLU and requantization of int32 accumulators to the 12-bit range.
    """
    q = (acc.astype(np.int64) * mul + (1 << (SHIFT - 1))) >> SHIFT
    return np.where(acc <= 0, 0, np.minimum(q, AMAX))



def wrap32(a: np.ndarray) -> np.ndarray:
    """
        int64 sums to int32 with two's complement wrap-around, as in C.
    """
    return a.astype(np.int32)



class WordModel:
    """
        Loads the quantized network once; `log_probs(ink)` reads one word.
    """
    def __init__(self, path: str = MODEL_PATH, use_native: bool = True):
        with np.load(path) as data:
            self.weights = {k: np.ascontiguousarray(data[k]) for k in data.files}
        p = self.weights
        self.c1, self.c2, self.c3 = (p[f"b{i}.weight"].shape[0] for i in (1, 2, 3))
        self.hidden = p["t1.weight"].shape[0]
        self.lib = native.load() if use_native else None
        if self.lib is not None:
            net = native.Network(self.c1, self.c2, self.c3, self.hidden)
            for i, layer in enumerate(LAYERS):
                net.w[i] = p[layer + ".weight"].ctypes.data
                net.b[i] = p[layer + ".bias"].ctypes.data
                net.m[i] = p[layer + ".mul"].ctypes.data
            net.wo, net.so, net.bo = (p[k].ctypes.data for k in ("out.weight", "out.scale", "out.bias"))
            self.network = net
            self.work = np.zeros(0, np.int32)


    def logits(self, ink: np.ndarray) -> np.ndarray:
        """
            (NUM_CLASSES, frames) float32 logits of an ink map (uint8, INPUT_HEIGHT x width).
        """
        ink = np.ascontiguousarray(ink, np.uint8)
        if ink.shape[0] != INPUT_HEIGHT or not 0 < ink.shape[1] <= MAX_INPUT_WIDTH:
            raise ValueError(f"ink map of shape {ink.shape}")
        if self.lib is None:
            return self._logits_numpy(ink)
        width = ink.shape[1]
        size = self.lib.hw_network_work_size(ctypes.byref(self.network), width)
        if self.work.nbytes < size:
            self.work = np.zeros(-(-size // 4), np.int32)
        out = np.empty((NUM_CLASSES, frames(width)), np.float32)
        self.lib.hw_network_logits(ctypes.byref(self.network), ink.ctypes.data, width, self.work.ctypes.data,
                                   out.ctypes.data)
        return out


    def log_probs(self, ink: np.ndarray) -> np.ndarray:
        """
            (NUM_CLASSES, frames) float32 log-probabilities of an ink map.
        """
        z = self.logits(ink)
        if self.lib is not None:
            self.lib.hw_log_softmax(z.ctypes.data, z.shape[1])
            return z
        m = z.max(0)
        return z - (m + np.log(np.exp(z - m).sum(0)))


    def _logits_numpy(self, ink: np.ndarray) -> np.ndarray:
        """
            Same integer arithmetic as hw_network_logits.
        """
        p = self.weights
        width = ink.shape[1]
        padded = (width + 16 + 3) // 4 * 4
        x = np.zeros((INPUT_HEIGHT, padded, 1), np.int64)
        x[:, :width, 0] = ink.astype(np.int64) << 3
        pools = ((2, 2), (2, 2), (2, 1), (2, 1))
        for i, (ph, pw) in enumerate(pools):
            layer = LAYERS[i]
            w = p[layer + ".weight"].reshape(p[layer + ".weight"].shape[0], -1).astype(np.int64)
            h, wd, c = x.shape
            xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
            windows = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(0, 1))  # h, wd, c, 3, 3
            cols = windows.transpose(0, 1, 3, 4, 2).reshape(h * wd, 9 * c)
            acc = wrap32(cols @ w.T + p[layer + ".bias"]).reshape(h, wd, -1)
            acc = acc.reshape(h // ph, ph, wd // pw, pw, -1).max(axis=(1, 3))
            x = requantize(acc, p[layer + ".mul"])
        x = x.transpose(1, 0, 2).reshape(x.shape[1], -1)  # columns: (t, h * c3 + c)
        for i, dilation in ((4, 1), (5, 2)):
            layer = LAYERS[i]
            w = p[layer + ".weight"].reshape(p[layer + ".weight"].shape[0], -1).astype(np.int64)
            t = x.shape[0]
            xp = np.pad(x, ((dilation, dilation), (0, 0)))
            cols = np.concatenate([xp[j * dilation:j * dilation + t] for j in range(3)], 1)
            x = requantize(wrap32(cols @ w.T + p[layer + ".bias"]), p[layer + ".mul"])
        acc = wrap32(x @ p["out.weight"].astype(np.int64).T)[:frames(width)]
        return np.ascontiguousarray((acc.astype(np.float32) * p["out.scale"] + p["out.bias"]).T)
