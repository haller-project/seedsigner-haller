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
        the seed is valid but that word is uncertain.
    """
    readings = [[(w, 0.0)] for w in VALID_12[:-1]] + [[("able", 0.0), (VALID_12[-1], -1.0)]]
    analysis = Analysis(readings)
    assert analysis.mnemonic == VALID_12
    assert analysis.status == "uncertain"
    assert list(analysis.uncertain) == [11]
    assert analysis.options(11)[0][0] == "able"



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
        A seed of probability 0.7 is valid with the default minimum (0.5) and
        gets uncertain words with a minimum of 0.85, as for the second model.
    """
    readings = [[(w, 0.0)] for w in VALID_12[:-1]] + [[(VALID_12[-1], 0.0), ("zoo", -1.0)]]
    assert 0.5 < Analysis(readings).confidence < 0.85
    assert Analysis(readings).status == "valid"
    strict = Analysis(readings, min_seed_probability=0.85)
    assert strict.status == "uncertain"
    assert list(strict.uncertain) == [11]
