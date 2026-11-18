import os
import types

import pytest

np = pytest.importorskip("numpy")  # in the Raspi requirements only
if not isinstance(np, types.ModuleType):  # replaced by a mock in tests/base.py
    pytest.skip("numpy is not installed", allow_module_level=True)
from PIL import Image, ImageDraw, ImageFont

from seedsigner.helpers.handwriting import native
from seedsigner.helpers.handwriting.decoder import ALPHABET, Decoder
from seedsigner.helpers.handwriting.preprocess import INPUT_HEIGHT, normalize_word
from seedsigner.helpers.handwriting.word_model import NUM_CLASSES, WordModel

needs_native = pytest.mark.skipif(native.load() is None,
                                  reason="native library not built (make -C src/seedsigner/helpers/handwriting/native)")

FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "seedsigner", "resources", "fonts",
                         "OpenSans-Regular.ttf")



def ink_map_of(text: str) -> np.ndarray:
    """ Ink map of `text` drawn on a gray background, as cut from a photo. """
    font = ImageFont.truetype(FONT_PATH, 40)
    image = Image.new("L", (40 + 26 * len(text), 70), 200)
    ImageDraw.Draw(image).text((20, 5), text, fill=40, font=font)
    return normalize_word(image)



def perfect_log_probs(classes: list[int]) -> np.ndarray:
    """ Log-probabilities putting 0.98 on the given class of every frame. """
    lp = np.full((NUM_CLASSES, len(classes)), np.log(0.02 / (NUM_CLASSES - 1)), np.float32)
    lp[classes, np.arange(len(classes))] = np.log(0.98)
    return lp



def frames_of(text: str) -> list[int]:
    """ One frame per character, a blank between letters and around the word. """
    out = [0]
    for c in text:
        out += [ALPHABET.index(c) + 1, 0]
    return out



@needs_native
def test_network_native_equals_numpy():
    """ The C library and the numpy code give bit-identical logits. """
    rng = np.random.default_rng(1)
    inks = [ink_map_of(w) for w in ("abandon", "zoo", "wrist", "17 ocean")]
    inks += [rng.integers(0, 256, (INPUT_HEIGHT, w), dtype=np.uint8) for w in (8, 9, 63, 256)]
    assert all(ink is not None for ink in inks)
    c_model, np_model = WordModel(), WordModel(use_native=False)
    for ink in inks:
        a, b = c_model.logits(ink), np_model.logits(ink)
        assert a.shape == b.shape == (NUM_CLASSES, max(1, ink.shape[1] // 4))
        assert np.array_equal(a, b)
        np.testing.assert_allclose(c_model.log_probs(ink), np_model.log_probs(ink), atol=1e-5)



def test_perfect_word():
    """ A clean reading of a word, with or without a line number, is decoded. """
    for use_native in (False, True):
        if use_native and native.load() is None:
            continue
        decoder = Decoder(use_native=use_native)
        for text, word in (("abandon", "abandon"), ("add", "add"), ("12zoo", "zoo"), ("7wrist", "wrist")):
            scores = decoder.scores(perfect_log_probs(frames_of(text)))
            assert decoder.words[int(np.argmax(scores))] == word



@needs_native
def test_decoder_native_equals_numpy():
    """ Same scores, within float rounding, for words close to the best one. """
    rng = np.random.default_rng(2)
    c_decoder, np_decoder = Decoder(), Decoder(use_native=False)
    for text in ("abandon", "3zoo", "letter"):
        lp = perfect_log_probs(frames_of(text)) + rng.normal(0, 1.0, (NUM_CLASSES, len(frames_of(text))))
        lp = (lp - np.log(np.exp(lp).sum(0))).astype(np.float32)
        a, b = c_decoder.scores(lp), np_decoder.scores(lp)
        assert np.argmax(a) == np.argmax(b)
        near = b > b.max() - 20
        np.testing.assert_allclose(a[near], b[near], atol=2e-3)
        assert ((a < -1e29) == (b < -1e29)).all()



def test_read_rendered_words():
    """ The shipped model reads printed words (a sanity check, not a benchmark). """
    model, decoder = WordModel(), Decoder()
    for word in ("abandon", "zoo", "wrist", "letter"):
        scores = decoder.scores(model.log_probs(ink_map_of(word)))
        assert decoder.words[int(np.argmax(scores))] == word
