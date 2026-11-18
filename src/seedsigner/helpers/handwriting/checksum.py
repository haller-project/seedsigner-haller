"""
    Choice of the seed among the possible readings of its words.

    The BIP39 checksum has only 4 bits for 12 words (8 for 24): a random
    combination of words passes it once in 16. It rules readings out, but only
    designates the right one when no other plausible combination passes it.
    The analysis therefore enumerates the plausible combinations, keeps the
    valid ones, and marks as uncertain the positions where another valid seed is
    almost as probable as the best one, or where the chosen word is unlikely;
    every word is uncertain when the seed is invalid or itself improbable.
"""
import hashlib
import math

from embit import bip39

# Readings of a word considered: probability at least MIN_ALTERNATIVE, at most
# MAX_READINGS per word; at most BEAM_WIDTH combinations are followed
MIN_ALTERNATIVE = 0.01
MAX_READINGS = 4
BEAM_WIDTH = 4096

# A valid seed less than RIVAL_RATIO times less probable than the best one makes
# the positions where they differ uncertain
RIVAL_RATIO = 10.0

# A chosen word below this probability is uncertain, even if the checksum agrees
MIN_WORD_PROBABILITY = 0.5

# Calibration of the word scores (log-likelihoods of the word model) and
# minimum probability of the whole seed for it to be declared valid without
# uncertain word; if it is lower, its words below MIN_CONFIDENT_WORD become
# uncertain (at least the least probable one). Chosen on simulated and real
# photos: most valid and exact seeds with at most one wrong seed declared valid.
TEMPERATURE = 1.2
MIN_SEED_PROBABILITY = 0.5
MIN_CONFIDENT_WORD = 0.95

# A valid seed this improbable is often a chance agreement of misread words
# with the checksum (1 in 16 for 12 words): a misread word can then have a high
# probability of its own. Every word not confirmed by the user is then
# uncertain, as for an invalid seed.
MIN_PLAUSIBLE_SEED = 0.1

WORD_INDEX = {w: i for i, w in enumerate(bip39.WORDLIST)}



def checksum_ok(words: list[str]) -> bool:
    """
        True if the BIP39 checksum of `words` (12 to 24 English words) is valid.
    """
    checksum_bits = len(words) // 3
    value = 0
    for w in words:
        value = value << 11 | WORD_INDEX[w]
    entropy = (value >> checksum_bits).to_bytes(len(words) * 11 // 33 * 4, "big")
    return value & ((1 << checksum_bits) - 1) == hashlib.sha256(entropy).digest()[0] >> (8 - checksum_bits)



def probabilities(scores: list[float]) -> list[float]:
    """
        Softmax of calibrated scores.
    """
    top = max(scores)
    e = [math.exp((s - top) / TEMPERATURE) for s in scores]
    total = sum(e)
    return [v / total for v in e]



class Analysis:
    """
        Result of the analysis of the readings of a seed.

        mnemonic: most probable valid seed, or the best reading of each word if
            no plausible combination is valid
        status: "valid", "uncertain" (valid seed with uncertain words) or "invalid"
        confidence: reading probability of the chosen seed (product of the
            probabilities of its words; 0 if invalid)
        uncertain: {position: [other possible words]}
    """
    def __init__(self, readings: list[list[tuple[str, float]]]):
        """
            readings: for each position, (word, score) pairs, scores being
            log-likelihoods; a single pair for a word confirmed by the user.
        """
        self.readings = [dict(zip([w for w, _ in r], probabilities([s for _, s in r]))) for r in readings]
        best = [max(p, key=p.get) for p in self.readings]
        valid = self._valid_seeds()
        self.mnemonic = valid[0][1] if valid else best
        self.uncertain = {}
        for k, p in enumerate(self.readings):
            if self.mnemonic[k] != best[k] or p[self.mnemonic[k]] < MIN_WORD_PROBABILITY:
                self._mark(k)
        if not valid:
            self.status, self.confidence = "invalid", 0.0
            self._mark_all()
            return
        for log_p, rival in valid[1:]:
            if valid[0][0] - log_p >= math.log(RIVAL_RATIO):
                break
            for k, (a, b) in enumerate(zip(self.mnemonic, rival)):
                if a != b:
                    self._mark(k, b)
        self.confidence = math.exp(valid[0][0])
        if self.confidence < MIN_PLAUSIBLE_SEED:
            self._mark_all()
        elif not self.uncertain and self.confidence < MIN_SEED_PROBABILITY:
            word_p = [p[w] for p, w in zip(self.readings, self.mnemonic)]
            weak = [k for k, q in enumerate(word_p) if q < MIN_CONFIDENT_WORD] or [word_p.index(min(word_p))]
            for k in weak:
                self._mark(k)
        self.status = "uncertain" if self.uncertain else "valid"


    def _valid_seeds(self) -> list[tuple[float, list[str]]]:
        """
            (log-probability, words) of the valid plausible combinations, most
            probable first.
        """
        beam = [(0.0, [])]
        for p in self.readings:
            options = sorted(p.items(), key=lambda item: -item[1])[:MAX_READINGS]
            options = [(w, math.log(q)) for w, q in options if q >= MIN_ALTERNATIVE] or [(options[0][0], math.log(options[0][1]))]
            beam = sorted(((log_p + v, words + [w]) for log_p, words in beam for w, v in options), key=lambda b: -b[0])[:BEAM_WIDTH]
        return [b for b in beam if checksum_ok(b[1])]


    def _mark_all(self):
        """
            Marks as uncertain every position with more than one reading (the
            words confirmed by the user have a single one).
        """
        for k, p in enumerate(self.readings):
            if len(p) > 1:
                self._mark(k)


    def _mark(self, k: int, rival: str = None):
        """
            Marks position k as uncertain, with `rival` or its other likely readings.
        """
        others = self.uncertain.setdefault(k, [])
        if rival is None:
            others += [w for w, _ in self.options(k, 4) if w != self.mnemonic[k] and w not in others]
        elif rival not in others:
            others.append(rival)


    def options(self, k: int, n: int = 3) -> list[tuple[str, float]]:
        """
            Words proposed for position k, with their reading probability, most
            probable first: the chosen word, its rivals, then the other readings
            of probability at least MIN_ALTERNATIVE; at most n.
        """
        p = self.readings[k]
        words = [self.mnemonic[k]] + [w for w in self.uncertain.get(k, []) if w != self.mnemonic[k]]
        for w in sorted(p, key=p.get, reverse=True):
            if len(words) >= n or p[w] < MIN_ALTERNATIVE:
                break
            if w not in words:
                words.append(w)
        return sorted(((w, p.get(w, 0.0)) for w in words[:n]), key=lambda item: -item[1])
