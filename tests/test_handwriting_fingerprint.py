import itertools
import os
import random
import types

import pytest

np = pytest.importorskip("numpy")  # in the Raspi requirements only
if not isinstance(np, types.ModuleType):  # replaced by a mock in tests/base.py
    pytest.skip("numpy is not installed", allow_module_level=True)
from PIL import Image, ImageDraw, ImageFont

from seedsigner.helpers.handwriting import fingerprint, native
from seedsigner.helpers.handwriting.word_model import ALPHABET, NUM_CLASSES

try:
    native.load()
except native.LibraryNotBuilt:
    if os.environ.get("CI"):
        raise
    pytest.skip("C library not built (make -C src/seedsigner/helpers/handwriting/native)", allow_module_level=True)

FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "seedsigner", "resources", "fonts",
                         "OpenSans-Regular.ttf")



def ctc_probability(log_probs: np.ndarray, labels: list[int]) -> float:
    """ Reference CTC: probability of the label sequence `labels` (classes, no blank) over every alignment. """
    p = np.exp(log_probs.astype(np.float64))
    extended = [0]
    for label in labels:
        extended += [label, 0]
    alpha = np.zeros(len(extended))
    alpha[0] = p[0, 0]
    if len(extended) > 1:
        alpha[1] = p[extended[1], 0]
    for t in range(1, p.shape[1]):
        previous = alpha.copy()
        for s, label in enumerate(extended):
            total = previous[s] + (previous[s - 1] if s >= 1 else 0.0)
            if s >= 2 and label != 0 and label != extended[s - 2]:
                total += previous[s - 2]
            alpha[s] = total * p[label, t]
    return alpha[-1] + (alpha[-2] if len(extended) > 1 else 0.0)



def classes(text: str) -> list[int]:
    return [1 + ALPHABET.index(c) for c in text]



def random_log_probs(frames: int, active: str, seed: int) -> np.ndarray:
    """ Random frame outputs with most of the mass on the blank and the characters of `active`. """
    rng = np.random.default_rng(seed)
    z = np.full((NUM_CLASSES, frames), -60.0)
    rows = [0] + classes(active)
    z[rows] = rng.normal(0, 1.5, (len(rows), frames))
    z -= np.log(np.exp(z).sum(0, keepdims=True))
    return z.astype(np.float32)



def test_constrained_ctc_matches_the_reference():
    """ Without prefix, the probability of one string is the CTC probability of its digits; with prefixes, the sum
    of the CTC probabilities of each prefix followed by the digits (prefix « ga » before a first digit a: the two a
    need a blank between them). The probabilities of the digits match the forward pass restricted to each digit. """
    for seed, string in ((1, "1a2a3a4a"), (2, "aa1111aa"), (3, "a1a1a1a1")):
        lp = random_log_probs(14, "ag1234", seed)
        mask = fingerprint.string_masks([string])
        alone = float(np.exp(fingerprint.forward(lp, mask, prefixes=("",))[0]))
        assert alone == pytest.approx(ctc_probability(lp, classes(string)), rel=1e-6)
        both = float(np.exp(fingerprint.forward(lp, mask, prefixes=("", "ga", "g"))[0]))
        expected = sum(ctc_probability(lp, classes(p + string)) for p in ("", "ga", "g"))
        assert both == pytest.approx(expected, rel=1e-6)
        log_likelihood, digits = fingerprint.digit_posteriors(lp, prefixes=("", "ga"))
        total = fingerprint.forward(lp, np.ones((1, fingerprint.LENGTH, 16), bool), prefixes=("", "ga"))[0]
        assert log_likelihood == pytest.approx(total)
        j, c = 0, fingerprint.HEX.index("a")
        one = np.ones((1, fingerprint.LENGTH, 16), bool)
        one[0, j] = False
        one[0, j, c] = True
        restricted = fingerprint.forward(lp, one, prefixes=("", "ga"))[0]
        assert digits[j, c] == pytest.approx(np.exp(restricted - total), rel=1e-6)



def test_digit_probabilities_sum_to_one():
    """ The probabilities of each digit at each position, by the forward-backward pass, are those of the forward
    pass restricted to that digit, divided by the total over every string of 8 digits; they sum to 1. """
    lp = random_log_probs(30, "0123456789abcdefxp", 4)
    reading = fingerprint.FingerprintReading(lp)
    every = np.ones((1, fingerprint.LENGTH, 16), bool)
    single = np.repeat(every, fingerprint.LENGTH * 16, axis=0).reshape(fingerprint.LENGTH, 16, fingerprint.LENGTH, 16)
    for j in range(fingerprint.LENGTH):
        single[j, :, j, :] = np.eye(16, dtype=bool)
    scores = fingerprint.forward(lp, np.concatenate([every, single.reshape(-1, fingerprint.LENGTH, 16)]))
    assert reading.log_likelihood == pytest.approx(scores[0])
    assert np.allclose(reading.digits, np.exp(scores[1:].reshape(fingerprint.LENGTH, 16) - scores[0]), atol=1e-12)
    assert np.allclose(reading.digits.sum(1), 1.0)
    p = reading.probabilities([reading.text, "00000000"])
    assert 0 < p[1] < p[0] <= 1



def written_line(text: str) -> Image.Image:
    font = ImageFont.truetype(FONT_PATH, 46)
    image = Image.new("L", (60 + 30 * len(text), 90), 215)
    ImageDraw.Draw(image).text((30, 15), text, fill=35, font=font)
    return image



def test_fingerprint_line_told_from_words():
    """ A printed fingerprint, with or without prefix, reads as a fingerprint; words, numbered or not, do not,
    including a word made of hexadecimal letters after its number. """
    from seedsigner.helpers.handwriting import fingerprint_margins, is_fingerprint
    for text in ("22a2b678", "73C5DA0A", "fp 0f8e1d2c"):
        assert is_fingerprint(written_line(text)), (text, fingerprint_margins(written_line(text)))
    for text in ("abandon", "12. decade", "zoo", "3 face", "ACCESS"):
        assert not is_fingerprint(written_line(text)), (text, fingerprint_margins(written_line(text)))



def rendered_list(lines: list[str]) -> Image.Image:
    """ Photo of a list printed in capitals on a light sheet, one line per entry. """
    font = ImageFont.truetype(FONT_PATH, 46)
    image = Image.new("L", (1200, 1700), 90)
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 120, 1050, 1600), fill=225)
    for i, line in enumerate(lines):
        draw.text((260, 180 + i * 100), line.upper(), fill=35, font=font)
    return image



def test_fingerprint_line_left_out_of_a_list():
    """ 12 words and their fingerprint under them: the 13th line is left out of the words and kept as the
    fingerprint crop; the 12 words are read. Above the words, the same. (A fingerprint of digits only, as
    04643705, is read less surely by the second network and may be kept as a line.) """
    from embit import bip39
    from seedsigner.helpers.handwriting import page, read_seed
    words = bip39.mnemonic_from_bytes(bytes(random.Random(5).getrandbits(8) for _ in range(16))).split()
    fp = fingerprint.seed_fingerprint(words)
    for lines in (words + [fp], [fp] + words):
        photo = rendered_list(lines)
        with pytest.raises(page.WordsNotFound):
            page.find_words(photo)
        from seedsigner.helpers.handwriting import is_fingerprint
        layout = page.find_words(photo, fingerprint=is_fingerprint)
        assert layout.count == 12 and layout.fingerprint is not None and layout.complete
        analysis = read_seed(photo).analyze()
        assert analysis.mnemonic == words and analysis.status == "valid"



def test_word_line_never_left_out():
    """ A 13th line that is a word stays a word: the list is not a seed. """
    from embit import bip39
    from seedsigner.helpers.handwriting import is_fingerprint, page
    words = bip39.mnemonic_from_bytes(bytes(random.Random(8).getrandbits(8) for _ in range(16))).split()
    with pytest.raises(page.WordsNotFound):
        page.find_words(rendered_list(words + ["decade"]), fingerprint=is_fingerprint)



def test_fingerprint_of_a_test_vector():
    """ BIP32 fingerprint of the BIP39 test mnemonic « abandon … about », without passphrase. """
    assert fingerprint.seed_fingerprint(["abandon"] * 11 + ["about"]) == "73c5da0a"



def test_fingerprint_verifies_the_seed():
    """ The fingerprint written under the words is that of the seed read: the seed is verified, every word certain,
    and the check says so. """
    from embit import bip39
    from seedsigner.helpers.handwriting import read_seed
    words = bip39.mnemonic_from_bytes(bytes(random.Random(5).getrandbits(8) for _ in range(16))).split()
    reading = read_seed(rendered_list(words + [fingerprint.seed_fingerprint(words)]))
    analysis = reading.analyze()
    assert reading.verified and reading.fingerprint.status == "verified" and reading.fingerprint.checked == 1
    assert analysis.mnemonic == words and analysis.status == "valid" and not analysis.uncertain
    assert reading.fingerprint.matched == fingerprint.seed_fingerprint(words)



def test_fingerprint_of_another_seed_never_makes_it_wrong():
    """ A fingerprint that matches none of the seeds read (a passphrase, a misread digit) does not call the seed
    wrong: the words read are kept, every one of them to check. """
    from embit import bip39
    from seedsigner.helpers.handwriting import read_seed
    words = bip39.mnemonic_from_bytes(bytes(random.Random(5).getrandbits(8) for _ in range(16))).split()
    other = bip39.mnemonic_from_bytes(bytes(random.Random(6).getrandbits(8) for _ in range(16))).split()
    reading = read_seed(rendered_list(words + [fingerprint.seed_fingerprint(other)]))
    analysis = reading.analyze()
    assert not reading.verified and reading.fingerprint.status == "mismatch" and reading.fingerprint.checked >= 1
    assert analysis.mnemonic == words and analysis.status == "uncertain" and len(analysis.uncertain) == 12
    confirmed = reading.analyze({k: w for k, w in enumerate(words)})
    assert confirmed.status == "valid"



def test_chance_match_bound():
    """ The probabilities of all strings of 8 digits sum to 1: at most 1 / MIN_MATCH_PROBABILITY of them match. """
    lp = random_log_probs(40, "0123456789abcdef", 9)
    reading = fingerprint.FingerprintReading(lp)
    assert np.allclose(reading.digits.sum(1), 1.0)
    assert 1 / (fingerprint.MIN_MATCH_PROBABILITY * 2 ** 32) < 3e-6
