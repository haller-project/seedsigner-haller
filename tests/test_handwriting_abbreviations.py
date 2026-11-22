"""
    Words abbreviated to their first 4 letters (decoder.ABBREVIATION_LETTERS).
"""
import os
import types

import pytest

np = pytest.importorskip("numpy")  # in the Raspi requirements only
if not isinstance(np, types.ModuleType):  # replaced by a mock in tests/base.py
    pytest.skip("numpy is not installed", allow_module_level=True)
from embit.bip39 import WORDLIST

from seedsigner.helpers.handwriting import decoder as decoder_module, native
from seedsigner.helpers.handwriting.decoder import ABBREVIATION_LETTERS, Decoder
from seedsigner.helpers.handwriting.word_model import ALPHABET, NUM_CLASSES

try:
    native.load()
except native.LibraryNotBuilt:
    if os.environ.get("CI"):
        raise
    pytest.skip("C library not built (make -C src/seedsigner/helpers/handwriting/native)", allow_module_level=True)



def perfect_log_probs(text: str) -> np.ndarray:
    """ One frame per character, a blank between letters, 0.98 on the class of every frame. """
    classes = [0]
    for c in text:
        classes += [ALPHABET.index(c) + 1, 0]
    lp = np.full((NUM_CLASSES, len(classes)), np.log(0.02 / (NUM_CLASSES - 1)), np.float32)
    lp[classes, np.arange(len(classes))] = np.log(0.98)
    return lp



def test_abbreviations_are_a_bijection():
    """ The abbreviated forms of the 2 048 words are distinct and none is another word. """
    forms = [w[:ABBREVIATION_LETTERS] for w in WORDLIST]
    assert len(set(forms)) == len(WORDLIST)
    assert not [w for w in WORDLIST if len(w) > ABBREVIATION_LETTERS and w[:ABBREVIATION_LETTERS] in set(WORDLIST)]



def test_abbreviated_word():
    """ A word written abbreviated, with or without a line number, is read as the full word, abbreviated (penalty of an
        abbreviated list). """
    decoder = Decoder()
    for text, word in (("aban", "abandon"), ("zone", "zone"), ("12wris", "wrist"), ("7acti", "action"),
                       ("abandon", "abandon"), ("act", "act")):
        full, lead = decoder.forms(perfect_log_probs(text), penalty=0.0)
        best = int(np.argmax(full))
        assert decoder.words[best] == word, text
        assert (lead[best] > 0) == (len(text.lstrip("0123456789")) == ABBREVIATION_LETTERS and len(word) > 4), text



def test_full_words_unchanged_without_abbreviations(monkeypatch):
    """ With ABBREVIATIONS False the scores are those of a decoder of the full words only. """
    monkeypatch.setattr(decoder_module, "ABBREVIATIONS", False)
    rng = np.random.default_rng(0)
    decoder = Decoder()
    lib = native.load()
    trie = native.trie(list(WORDLIST))
    import ctypes
    work = np.zeros(lib.hw_decoder_work_size(ctypes.byref(trie)) // 4, np.float32)
    for _ in range(5):
        z = rng.normal(0, 3, (NUM_CLASSES, 40)).astype(np.float32)
        lp = (z - np.log(np.exp(z).sum(0))).astype(np.float32)
        out = np.empty(len(WORDLIST), np.float32)
        lib.hw_word_scores(ctypes.byref(trie), lp.ctypes.data, lp.shape[1], decoder.beam, work.ctypes.data,
                           out.ctypes.data)
        assert np.array_equal(decoder.scores(lp), out)
        assert np.all(np.isneginf(decoder.forms(lp)[1]))



def test_abbreviation_evidence():
    """ An abbreviated word reads better as an abbreviation than as any full word; a full word does not. """
    from seedsigner.helpers.handwriting.decoder import abbreviation_evidence
    decoder = Decoder()
    for text, abbreviated in (("aban", True), ("wris", True), ("abandon", False), ("exit", False), ("zoo", False)):
        assert (abbreviation_evidence(*decoder.raw(perfect_log_probs(text))) > 0) == abbreviated, text
