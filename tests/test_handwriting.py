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
