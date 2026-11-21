"""
    Rows of a list that may not be words (a label such as "Fingerprint", a
    fingerprint, a second fingerprint): sets of rows tried at the start or the
    end of a list, reading of a label by CTC, trust of a reading with such
    rows left in.
"""
import math

import pytest

np = pytest.importorskip("numpy")

from seedsigner.helpers import handwriting  # noqa: E402
from seedsigner.helpers.handwriting import detector, fingerprint  # noqa: E402
from seedsigner.helpers.handwriting.native import ALPHABET, NUM_CLASSES  # noqa: E402


def box(a0, a1, c, h=10.0, pixels=100):
    b = object.__new__(detector.Box)
    b.ux, b.uy = 1.0, 0.0
    b.a0, b.a1, b.c, b.h = float(a0), float(a1), float(c), float(h)
    b.score, b.pixels = 0.9, pixels
    return b


def two_columns(rows=6):
    return [box(x, x + 60, 30 + 20 * r) for r in range(rows) for x in (0, 150)]


def test_label_and_fingerprint_rows_under_two_columns():
    # 12 words in two columns, a label row and a fingerprint row split in two pieces below them: 15 lines (a valid
    # count) for group_lines; leaving out the last two rows leaves the 12 words
    words = two_columns()
    label = box(40, 110, 160)
    pieces = [box(60, 120, 182), box(150, 190, 181)]
    sets = detector.extra_line_sets(words + [label] + pieces)
    found = [(rows, columns) for rows, columns in sets if sum(len(c) for c in columns) == 12]
    assert len(found) == 1
    rows, columns = found[0]
    assert [p for _, p in rows] == [[label], pieces]
    assert [len(c) for c in columns] == [6, 6]


def test_no_extra_line_set_for_words_alone():
    assert all(sum(len(c) for c in columns) in (12, 15, 18, 21, 24)
               for _, columns in detector.extra_line_sets(two_columns()))
    assert not [s for s in detector.extra_line_sets(two_columns()) if len(s[0]) == 2]


def outputs(text, frames_per_char=3, sharp=8.0):
    # network outputs reading `text`: each character over a few frames, blanks between
    classes = []
    for c in text:
        classes += [1 + ALPHABET.index(c)] * frames_per_char + [0]
    log_probs = np.full((NUM_CLASSES, len(classes)), -sharp)
    for t, k in enumerate(classes):
        log_probs[k, t] = 0.0
    return log_probs - np.log(np.exp(log_probs).sum(0, keepdims=True))


def test_text_log_likelihood_prefers_the_text_read():
    log_probs = outputs("fingerprint")
    assert fingerprint.text_log_likelihood(log_probs, "fingerprint") > -1.0
    assert fingerprint.text_log_likelihood(log_probs, "fingerprin") < -5.0
    assert fingerprint.text_log_likelihood(log_probs, "fp") < -20.0


def test_text_log_likelihood_of_repeated_letters():
    # a double letter needs a blank between its two characters
    log_probs = outputs("off")
    assert fingerprint.text_log_likelihood(log_probs, "off") > -1.0
    assert fingerprint.text_log_likelihood(log_probs, "of") < -5.0


def test_label_margins_of_a_label_line():
    log_probs = outputs("seedwords")
    word_scores = np.array([fingerprint.text_log_likelihood(log_probs, "seed")])
    margin, fit, word_fit = fingerprint.label_margins(log_probs, word_scores)
    assert margin > 10 and fit > -1.0 and word_fit < -10


def test_no_label_is_a_bip39_word():
    from seedsigner.helpers.handwriting.decoder import WORDLIST
    assert not set(fingerprint.LABELS) & set(WORDLIST)


def test_trust_of_a_reading_with_extra_lines_is_low():
    words = [handwriting.WordReading(None, [(w, 0.0)]) for w in
             "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about".split()]
    reading = handwriting.SeedReading("list", words)
    assert reading.trust().level == handwriting.TRUST_HIGH
    reading.extra_lines = True
    assert reading.trust().level == handwriting.TRUST_LOW
