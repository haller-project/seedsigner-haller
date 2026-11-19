"""
    Choice of the seed among the possible readings of its words.

    The BIP39 checksum has only 4 bits for 12 words (8 for 24): a random
    combination of words passes it once in 16. It rules readings out, but only
    designates the right one when no other plausible combination passes it.
    The analysis therefore enumerates the plausible combinations and keeps the
    valid ones. Word scores are calibrated log-likelihoods (see
    handwriting.TEMPERATURES), so that the probability of a seed given the
    photo and the checksum can be estimated: its reading probability divided
    by that of all the valid seeds, enumerated or not. A position is uncertain
    when the probability of its word given the checksum is low; every word is
    uncertain when the seed is invalid or itself improbable.
"""
import hashlib
import math

from embit import bip39

# Readings of a word considered: probability at least MIN_ALTERNATIVE, at most
# MAX_READINGS per word; at most BEAM_WIDTH combinations are followed
MIN_ALTERNATIVE = 0.01
MAX_READINGS = 4
BEAM_WIDTH = 4096

# A chosen word whose probability given the checksum is below MIN_CONFIDENT_WORD
# is uncertain. A seed less probable than MIN_SEED_PROBABILITY is not declared
# valid: if none of its words is uncertain, the least probable one becomes so.
# Below MIN_PLAUSIBLE_SEED, the seed is often a chance agreement of misread
# words with the checksum: every word not confirmed by the user is uncertain,
# as for an invalid seed. Chosen on simulated photos kept apart from the test
# photos: no wrong seed declared valid, as many right ones as possible.
MIN_CONFIDENT_WORD = 0.95
MIN_SEED_PROBABILITY = 0.5
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
    e = [math.exp(s - top) for s in scores]
    total = sum(e)
    return [v / total for v in e]



class Analysis:
    """
        Result of the analysis of the readings of a seed.

        mnemonic: most probable valid seed, or the best reading of each word if
            no plausible combination is valid
        status: "valid", "uncertain" (valid seed with uncertain words) or "invalid"
        confidence: probability of the chosen seed given the readings and the
            checksum (0 if invalid)
        uncertain: {position: [other possible words]}
    """
    def __init__(self, readings: list[list[tuple[str, float]]], min_seed_probability: float = MIN_SEED_PROBABILITY,
                 doubtful=()):
        """
            readings: for each position, (word, score) pairs, scores being
            calibrated log-likelihoods; a single pair for a word confirmed by
            the user.
            min_seed_probability: minimum probability of a seed declared valid
            without uncertain word.
            doubtful: positions whose crop may miss part of the word; they are
            uncertain unless confirmed.
        """
        self.readings = [dict(zip([w for w, _ in r], probabilities([s for _, s in r]))) for r in readings]
        best = [max(p, key=p.get) for p in self.readings]
        beam = self._beam()
        self._valid = [b for b in beam if checksum_ok(b[1])]
        self.mnemonic = self._valid[0][1] if self._valid else best
        self.uncertain = {}
        self._marginals = {}
        if not self._valid:
            self.status, self.confidence = "invalid", 0.0
            self._mark_all()
            return

        # Probability of the valid seeds: the enumerated ones, and the share of
        # the combinations left out that passes the checksum by chance
        self._beam_words = [words for _, words in beam]
        self._beam_p = [math.exp(log_p) for log_p, _ in beam]
        self._chance = 2.0 ** -(len(readings) // 3)
        self._total = sum(math.exp(log_p) for log_p, _ in self._valid) \
            + self._chance * max(0.0, 1.0 - sum(self._beam_p))
        self.confidence = math.exp(self._valid[0][0]) / self._total
        if self.confidence < MIN_PLAUSIBLE_SEED:
            self._mark_all()
        else:
            word_p = [self.word_probability(k, w) for k, w in enumerate(self.mnemonic)]
            for k, q in enumerate(word_p):
                if self.mnemonic[k] != best[k] or q < MIN_CONFIDENT_WORD:
                    self._mark(k)
            if not self.uncertain and self.confidence < min_seed_probability:
                self._mark(word_p.index(min(word_p)))
        for k in doubtful:
            if len(self.readings[k]) > 1:
                self._mark(k)
        self.status = "uncertain" if self.uncertain else "valid"


    def _beam(self) -> list[tuple[float, list[str]]]:
        """
            (log-probability, words) of the most probable combinations of the
            plausible readings, most probable first.
        """
        beam = [(0.0, [])]
        for p in self.readings:
            options = sorted(p.items(), key=lambda item: -item[1])[:MAX_READINGS]
            options = [(w, math.log(q)) for w, q in options if q >= MIN_ALTERNATIVE] or [(options[0][0], math.log(options[0][1]))]
            beam = sorted(((log_p + v, words + [w]) for log_p, words in beam for w, v in options), key=lambda b: -b[0])[:BEAM_WIDTH]
        return beam


    def word_probability(self, k: int, word: str) -> float:
        """
            Probability that `word` is at position k, given the readings and the
            checksum (its reading probability if no plausible seed is valid).
        """
        p = self.readings[k].get(word, 0.0)
        if not self._valid:
            return p
        if k not in self._marginals:
            valid, enumerated = {}, {}
            for log_p, words in self._valid:
                valid[words[k]] = valid.get(words[k], 0.0) + math.exp(log_p)
            for q, words in zip(self._beam_p, self._beam_words):
                enumerated[words[k]] = enumerated.get(words[k], 0.0) + q
            self._marginals[k] = (valid, enumerated)
        valid, enumerated = self._marginals[k]
        return min(1.0, (valid.get(word, 0.0) + self._chance * max(0.0, p - enumerated.get(word, 0.0))) / self._total)


    def _mark_all(self):
        """
            Marks as uncertain every position with more than one reading (the
            words confirmed by the user have a single one).
        """
        for k, p in enumerate(self.readings):
            if len(p) > 1:
                self._mark(k)


    def _mark(self, k: int):
        """
            Marks position k as uncertain, with its other likely readings.
        """
        others = self.uncertain.setdefault(k, [])
        others += [w for w, _ in self.options(k, 4) if w != self.mnemonic[k] and w not in others]


    def options(self, k: int, n: int = 3) -> list[tuple[str, float]]:
        """
            Words proposed for position k with their probability (see
            word_probability), most probable first: the chosen word and the
            most probable of its other readings of reading probability at least
            MIN_ALTERNATIVE; at most n.
        """
        p = self.readings[k]
        chosen = self.mnemonic[k]
        others = [w for w in p if w != chosen and p[w] >= MIN_ALTERNATIVE]
        q = {w: self.word_probability(k, w) for w in [chosen] + others}
        others = sorted(others, key=lambda w: (-q[w], -p[w]))[:n - 1]
        return sorted(((w, q[w]) for w in [chosen] + others), key=lambda item: -item[1])
