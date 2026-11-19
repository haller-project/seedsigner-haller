import math

import pytest
from embit import bip39

from seedsigner.helpers.handwriting import checksum
from seedsigner.helpers.handwriting.checksum import Analysis

VALID_12 = "obscure bone gas open exotic abuse virus bunker shuffle nasty ship dash".split()
VALID_24 = ["abandon"] * 23 + ["art"]



def test_checksum_ok():
    """
        The checksum matches embit's on valid and invalid mnemonics of every length.
    """
    assert checksum.checksum_ok(VALID_12)
    assert checksum.checksum_ok(VALID_24)
    assert not checksum.checksum_ok(VALID_12[:-1] + ["able"])
    for n in (12, 15, 18, 21, 24):
        for last in bip39.WORDLIST[:64]:
            words = ["zoo"] * (n - 1) + [last]
            assert checksum.checksum_ok(words) == bip39.mnemonic_is_valid(" ".join(words))



def test_analysis_valid():
    """
        Clear readings of a valid seed: valid, no uncertain word.
    """
    readings = [[(w, 0.0), ("zoo", -40.0)] for w in VALID_12]
    analysis = Analysis(readings)
    assert analysis.mnemonic == VALID_12
    assert analysis.status == "valid"
    assert analysis.uncertain == {}
    assert analysis.confidence > 0.99



def test_analysis_checksum_fixes_a_word():
    """
        The best reading of a word fails the checksum; the second one passes:
        the seed is chosen, but that word is uncertain since the photo read
        another word. Given the checksum, the word chosen is certain and the
        word read impossible: the options say so.
    """
    readings = [[(w, 0.0)] for w in VALID_12[:-1]] + [[("able", 0.0), (VALID_12[-1], -1.0)]]
    analysis = Analysis(readings)
    assert analysis.mnemonic == VALID_12
    assert analysis.status == "uncertain"
    assert list(analysis.uncertain) == [11]
    assert analysis.options(11) == [(VALID_12[-1], 1.0), ("able", 0.0)]



def test_analysis_probabilities_given_the_checksum():
    """
        Two readings of the last word both pass the checksum: their
        probabilities given the checksum are their reading probabilities, and
        the less probable rival makes the word uncertain.
    """
    rival = next(w for w in bip39.WORDLIST if w != VALID_12[-1] and checksum.checksum_ok(VALID_12[:-1] + [w]))
    readings = [[(w, 0.0)] for w in VALID_12[:-1]] + [[(VALID_12[-1], 0.0), (rival, -1.0)]]
    analysis = Analysis(readings)
    p = 1 / (1 + math.exp(-1.0))
    assert analysis.mnemonic == VALID_12
    assert analysis.confidence == pytest.approx(p)
    assert analysis.options(11) == [(VALID_12[-1], pytest.approx(p)), (rival, pytest.approx(1 - p))]
    assert analysis.status == "uncertain"
    assert list(analysis.uncertain) == [11]



def test_analysis_invalid():
    readings = [[(w, 0.0)] for w in VALID_12[:-1]] + [[("able", 0.0)]]
    analysis = Analysis(readings)
    assert analysis.status == "invalid"
    assert analysis.confidence == 0.0
    assert analysis.mnemonic[-1] == "able"



def test_analysis_invalid_marks_every_word():
    """
        No plausible combination passes the checksum: any word may be the misread
        one, so every word not confirmed by the user is uncertain.
    """
    readings = [[(w, 0.0), ("zoo", -40.0)] for w in VALID_12[:-1]] + [[("able", 0.0)]]
    analysis = Analysis(readings)
    assert analysis.status == "invalid"
    assert sorted(analysis.uncertain) == list(range(11))



def test_analysis_improbable_seed_marks_every_word():
    """
        Each word is read with probability 0.6 only: the valid seed found is
        improbable (0.6^12) and every word is uncertain, until enough words are
        confirmed.
    """
    readings = [[(w, 0.0), ("zoo", -0.5)] for w in VALID_12]
    analysis = Analysis(readings)
    assert analysis.mnemonic == VALID_12
    assert analysis.confidence < checksum.MIN_PLAUSIBLE_SEED
    assert sorted(analysis.uncertain) == list(range(12))

    confirmed = [[(w, 0.0)] for w in VALID_12[:10]] + readings[10:]
    analysis = Analysis(confirmed)
    assert analysis.confidence > checksum.MIN_PLAUSIBLE_SEED
    assert set(analysis.uncertain) <= {10, 11}



def test_analysis_min_seed_probability():
    """
        Every word has a slightly less probable rival that also passes the
        checksum: each word is above MIN_CONFIDENT_WORD given the checksum, but
        the seed itself has a probability of about 0.7. It is valid with the
        default minimum (0.5); with a minimum of 0.85, as for the second model,
        its least probable word becomes uncertain.
    """
    rivals = [next(w for w in bip39.WORDLIST
                   if w != VALID_12[k] and checksum.checksum_ok(VALID_12[:k] + [w] + VALID_12[k + 1:]))
              for k in range(12)]
    readings = [[(w, 0.0), (r, -3.3)] for w, r in zip(VALID_12, rivals)]
    analysis = Analysis(readings)
    assert 0.5 < analysis.confidence < 0.85
    assert min(analysis.word_probability(k, w) for k, w in enumerate(VALID_12)) > checksum.MIN_CONFIDENT_WORD
    assert analysis.status == "valid"
    strict = Analysis(readings, min_seed_probability=0.85)
    assert strict.status == "uncertain"
    assert len(strict.uncertain) == 1
