"""
    Master key fingerprint written with the seed: 8 hexadecimal digits, the
    first 4 bytes of the HASH160 of the BIP32 master public key (no
    passphrase), as shown by wallets and SeedSigner. Many users write it under
    their words; it is not secret.

    Reading: the outputs of the word network on the line (see word_model.py)
    are decoded by CTC constrained to an optional prefix (PREFIXES) followed
    by exactly 8 hexadecimal digits. The sum over every string of 8 digits is
    the likelihood of the line being a fingerprint, compared with the
    likelihood of its best BIP39 word and with that of the free reading of
    the network (`margins`). A backward pass gives the probability of each
    digit at each position; the forward pass restricted to one string, the
    probability of that fingerprint.
"""
import numpy as np

from .native import ALPHABET

HEX = "0123456789abcdef"
LENGTH = 8
HEX_CLASSES = np.array([1 + ALPHABET.index(c) for c in HEX])
# Words written before the digits ("FP: 22a2b678"); the colon and spaces are
# not in the alphabet of the network and read as blanks
PREFIXES = ("", "fp", "xfp", "mfp", "fingerprint")



class Lattice:
    """
        CTC states of a line read as one of `prefixes` (fixed letters)
        followed by LENGTH hexadecimal digits, for every prefix at once and a
        batch of digit masks; probabilities rescaled at every frame
        (`log_scale`).
    """
    def __init__(self, prefixes: tuple[str, ...], batch: int):
        count, longest = len(prefixes), max(1, max(len(p) for p in prefixes))
        self.labels = np.zeros((count, longest), int)
        self.valid = np.zeros((count, longest))            # positions holding a letter of the prefix
        self.differs = np.zeros((count, longest))          # letter i differs from letter i - 1
        self.last = np.full(count, -1)                     # index of the last letter, -1 for no prefix
        for k, prefix in enumerate(prefixes):
            for i, c in enumerate(prefix):
                self.labels[k, i] = 1 + ALPHABET.index(c)
                self.valid[k, i] = 1.0
                self.differs[k, i] = float(i > 0 and c != prefix[i - 1])
            self.last[k] = len(prefix) - 1
        rows = np.arange(count)
        last = np.maximum(self.last, 0)
        self.no_prefix = (self.last < 0)[:, None]
        # a first digit equal to the last letter of the prefix needs a blank between them
        self.same_as_last = (HEX_CLASSES[None, :] == self.labels[rows, last][:, None]) & ~self.no_prefix
        self._rows, self._last = rows, last
        self.start = np.ones(count)                        # blanks only so far
        self.q_on = np.zeros((count, longest))             # in prefix letter i
        self.q_off = np.zeros((count, longest))            # in a blank after prefix letter i
        self.h_on = np.zeros((count, batch, LENGTH, 16))   # in digit j, by digit value
        self.h_off = np.zeros((count, batch, LENGTH, 16))  # in a blank after digit j
        self.log_scale = 0.0


    def step(self, e: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        """
            Moves on by one frame of outputs `e` (NUM_CLASSES,) with the digit
            masks `mask` (batch, LENGTH, 16); returns the mass entering each
            digit state (batch, LENGTH, 16), summed over the prefixes, and the
            log scale it is given in.
        """
        blank, digits = e[0], e[HEX_CLASSES]
        q_on, q_off = self.q_on, self.q_off
        first = np.where(self.no_prefix, self.start[:, None],
                         q_off[self._rows, self._last][:, None]
                         + np.where(self.same_as_last, 0.0, q_on[self._rows, self._last][:, None]))   # (count, 16)
        enter = np.empty_like(self.h_on)
        enter[:, :, 0] = digits * mask[:, 0] * first[:, None, :]
        sums = self.h_on[:, :, :-1].sum(3, keepdims=True) + self.h_off[:, :, :-1].sum(3, keepdims=True)
        enter[:, :, 1:] = digits * mask[:, 1:] * (sums - self.h_on[:, :, :-1])
        self.h_on, self.h_off = enter + digits * mask * self.h_on, blank * (self.h_on + self.h_off)
        before = np.concatenate((self.start[:, None], q_off[:, :-1]), axis=1)                # blank before letter i
        previous = np.concatenate((np.zeros((len(q_on), 1)), q_on[:, :-1]), axis=1) * self.differs
        self.q_on = e[self.labels] * (q_on + before + previous) * self.valid
        self.q_off = blank * (q_on + q_off)
        self.start = self.start * blank
        scale = max(self.start.max(), self.q_on.max(), self.q_off.max(), self.h_on.max(), self.h_off.max())
        previous_scale = self.log_scale
        if scale <= 0:
            self.log_scale = -np.inf
            return enter.sum(0), -np.inf
        self.start, self.q_on, self.q_off = self.start / scale, self.q_on / scale, self.q_off / scale
        self.h_on, self.h_off = self.h_on / scale, self.h_off / scale
        self.log_scale += np.log(scale)
        return enter.sum(0), previous_scale


    def log_total(self) -> np.ndarray:
        """
            (batch,) log-likelihood of the line ending after the last digit.
        """
        total = self.h_on[:, :, -1].sum((0, 2)) + self.h_off[:, :, -1].sum((0, 2))
        with np.errstate(divide="ignore"):
            return np.log(total) + self.log_scale



def forward(log_probs: np.ndarray, allowed: np.ndarray, prefixes=PREFIXES) -> np.ndarray:
    """
        Log-likelihood, for each mask of `allowed` (B, LENGTH, 16) booleans,
        that the line reads as one of `prefixes` followed by LENGTH
        hexadecimal digits, digit j being one allowed by mask[j].
        log_probs: (NUM_CLASSES, frames) network outputs. Returns (B,).
    """
    p = np.exp(log_probs.astype(np.float64))
    mask = allowed.astype(np.float64)
    lattice = Lattice(prefixes, allowed.shape[0])
    for t in range(p.shape[1]):
        lattice.step(p[:, t], mask)
    return lattice.log_total()



def digit_posteriors(log_probs: np.ndarray, prefixes=PREFIXES) -> tuple[float, np.ndarray]:
    """
        Log-likelihood of the line being one of `prefixes` and 8 hexadecimal
        digits, and (LENGTH, 16) probability of each digit value at each
        position given that reading. Forward pass keeping the mass entering
        each digit state at each frame; backward pass giving the probability
        of completing the string from that state; their product summed over
        the frames is the probability of the digit (each string enters each
        of its digits once).
    """
    p = np.exp(log_probs.astype(np.float64))
    frames = p.shape[1]
    every = np.ones((1, LENGTH, 16))
    lattice = Lattice(prefixes, 1)
    entering = np.zeros((frames, LENGTH, 16))
    scales = np.zeros(frames)
    for t in range(frames):
        enter, scales[t] = lattice.step(p[:, t], every)
        entering[t] = enter[0]
    log_likelihood = float(lattice.log_total()[0])
    if not np.isfinite(log_likelihood):
        return -np.inf, np.full((LENGTH, 16), 1 / 16)

    # backward: b_on[j, c], b_off[j, c] = probability of the rest of the frames from that state
    b_on, b_off = np.zeros((LENGTH, 16)), np.zeros((LENGTH, 16))
    b_on[-1] = b_off[-1] = 1.0
    back_scale = 0.0
    posterior = entering[-1] * b_on * np.exp(scales[-1] + back_scale - log_likelihood)
    for t in range(frames - 2, -1, -1):
        e = p[:, t + 1]
        blank, digits = e[0], e[HEX_CLASSES]
        onward = np.zeros((LENGTH, 16))
        onward[:-1] = digits * b_on[1:]   # next frame starts digit j + 1 with value c' (not c without a blank)
        total_onward = onward.sum(1, keepdims=True)
        b_on, b_off = digits * b_on + blank * b_off + total_onward - onward, blank * b_off + total_onward
        scale = max(b_on.max(), b_off.max())
        if scale <= 0:
            break
        b_on, b_off = b_on / scale, b_off / scale
        back_scale += np.log(scale)
        if np.isfinite(scales[t]):
            posterior += entering[t] * b_on * np.exp(scales[t] + back_scale - log_likelihood)
    return log_likelihood, posterior



def string_masks(strings: list[str]) -> np.ndarray:
    """
        (len(strings), LENGTH, 16) masks allowing exactly each string.
    """
    masks = np.zeros((len(strings), LENGTH, 16), bool)
    for b, s in enumerate(strings):
        for j, c in enumerate(s.lower()):
            masks[b, j, HEX.index(c)] = True
    return masks



class FingerprintReading:
    """
        Reading of a line as a fingerprint.

        log_likelihood: log-likelihood of the line being an optional prefix and
            8 hexadecimal digits (any of them)
        digits: (LENGTH, 16) probability of each digit value at each position
        text: most probable digit at each position
        probability(fp): probability of the fingerprint `fp` (8 hexadecimal
            digits) given the line, among all strings of 8 digits
    """
    def __init__(self, log_probs: np.ndarray):
        self.log_probs = log_probs
        self.log_likelihood, self.digits = digit_posteriors(log_probs)
        self.text = "".join(HEX[i] for i in self.digits.argmax(1))


    def probabilities(self, fingerprints: list[str]) -> list[float]:
        """
            Probability of each fingerprint given the line.
        """
        if not fingerprints or not np.isfinite(self.log_likelihood):
            return [0.0] * len(fingerprints)
        scores = forward(self.log_probs, string_masks(fingerprints))
        return [float(np.exp(s - self.log_likelihood)) for s in scores]



def margins(log_probs: np.ndarray, word_scores: np.ndarray) -> tuple[float, float]:
    """
        (word margin, fit) of the line read as a fingerprint:
        * word margin: its log-likelihood minus that of its best BIP39 word
          (word_scores: Decoder.scores of the same outputs);
        * fit: its log-likelihood minus that of the best path of the network
          with no constraint (most probable class at every frame), a lower
          bound of the probability of the network's own reading. Near 0 or
          above when the line is read as 8 digits; far below for two words
          written on one line, which no single word explains either.
    """
    every = np.ones((1, LENGTH, 16), bool)
    hexadecimal = float(forward(log_probs, every)[0])
    free = float(log_probs.max(0).sum())
    return hexadecimal - float(np.max(word_scores)), hexadecimal - free

