import hashlib
import os
import types

import pytest

np = pytest.importorskip("numpy")  # in the Raspi requirements only
if not isinstance(np, types.ModuleType):  # replaced by a mock in tests/base.py
    pytest.skip("numpy is not installed", allow_module_level=True)
from PIL import Image, ImageDraw, ImageFont

from seedsigner.helpers.handwriting import native
from seedsigner.helpers.handwriting.decoder import Decoder
from seedsigner.helpers.handwriting.preprocess import normalize_word
from seedsigner.helpers.handwriting.word_model import ALPHABET, MODEL_PATH, NUM_CLASSES, WordModel

try:
    native.load()
except native.LibraryNotBuilt:
    if os.environ.get("CI"):
        raise
    pytest.skip("C library not built (make -C src/seedsigner/helpers/handwriting/native)", allow_module_level=True)

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



def test_perfect_word():
    """ A clean reading of a word, with or without a line number, is decoded. """
    decoder = Decoder()
    for text, word in (("abandon", "abandon"), ("add", "add"), ("12zoo", "zoo"), ("7wrist", "wrist")):
        scores = decoder.scores(perfect_log_probs(frames_of(text)))
        assert decoder.words[int(np.argmax(scores))] == word



def test_haller_1_weights():
    """ The shipped weights are those of Haller-1 listed in docs/handwritten_seeds.md. """
    expected = {
        "haller-1-capitals.npz": "1faede0402e7e3b48e922f8a82a3d72fcbb7e7354b4909298e31616431e7206a",
        "haller-1-lowercase.npz": "55c2aba311f7e31e4fdbe6e82d17fc820165064c23c1792e1059fa6bf4e3863f",
    }
    for name, sha256 in expected.items():
        with open(os.path.join(os.path.dirname(MODEL_PATH), name), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == sha256



def test_read_rendered_words():
    """ Haller-1 reads printed words (a sanity check, not a benchmark). """
    model, decoder = WordModel(), Decoder()
    for word in ("abandon", "zoo", "wrist", "letter"):
        scores = decoder.scores(model.log_probs(ink_map_of(word)))
        assert decoder.words[int(np.argmax(scores))] == word



def word_crop(text: str) -> Image.Image:
    """ `text` drawn on a gray background, as cut from a photo. """
    font = ImageFont.truetype(FONT_PATH, 40)
    image = Image.new("L", (40 + 26 * len(text), 70), 200)
    ImageDraw.Draw(image).text((20, 5), text, fill=40, font=font)
    return image



def fake_layout(words: list[str]):
    """ Layout of one column of the given words, one crop variant. """
    crops = [word_crop(w) for w in words]
    return types.SimpleNamespace(kind="list", count=len(words), orders=[list(range(len(words)))], variants=1,
                                 crop=lambda k, variant=0: crops[k] if variant == 0 else None, alternative=None)



def counting_networks(monkeypatch) -> dict[str, int]:
    """ Counts the calls to the two networks of the engine. """
    import seedsigner.helpers.handwriting as hw
    first, decoder, lowercase = hw.load_engine()
    calls = {"first": 0, "lowercase": 0}
    for name, network in (("first", first), ("lowercase", lowercase)):
        original = network.log_probs
        def counted(ink, name=name, original=original):
            calls[name] += 1
            return original(ink)
        monkeypatch.setattr(network, "log_probs", counted)
    return calls



def test_blend_reuses_the_scores_of_both_networks(monkeypatch):
    """ A crop is read once per network; the blend is the weighted sum of their log-probabilities. """
    import seedsigner.helpers.handwriting as hw
    calls = counting_networks(monkeypatch)
    reader = hw.LayoutReader(fake_layout(["letter"]))
    assert reader.read(0, 0, hw.FIRST).candidates[0][0] == "letter"
    assert reader.read(0, 0, hw.LOWERCASE).candidates[0][0] == "letter"
    blend = reader.read(0, 0, hw.BLEND).candidates
    assert calls == {"first": 1, "lowercase": 1}
    assert reader.read(0, 1, hw.BLEND) is None
    expected = hw.BLEND_WEIGHT * hw.log_softmax(reader.model_scores(0, 0, 0)) \
        + (1 - hw.BLEND_WEIGHT) * hw.log_softmax(reader.model_scores(0, 0, 1))
    assert blend[0][0] == "letter"
    assert blend[0][1] == pytest.approx(expected.max())



def test_second_readings_only_when_the_seed_is_not_valid(monkeypatch):
    """ A valid seed is read by the first network only; an invalid one also by the second network and the blend. """
    import seedsigner.helpers.handwriting as hw
    valid = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about".split()
    for words, second in ((valid, 0), (valid[:-1] + ["zoo"], len(valid))):
        calls = counting_networks(monkeypatch)
        monkeypatch.setattr(hw, "find_words", lambda image, step, words=words: fake_layout(words))
        reading = hw.read_seed(Image.new("L", (10, 10)))
        assert [w.candidates[0][0] for w in reading.words] == words
        assert calls == {"first": len(words), "lowercase": second}



def test_progress_is_steady_and_moves_forward():
    """ With exact costs the progress is proportional to the work done; with wrong ones it still moves forward. """
    import seedsigner.helpers.handwriting as hw
    fractions = []
    progress = hw.Progress(fractions.append, 10.0)
    for _ in range(4):
        progress.step(1.0)
    assert fractions == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    progress.expect(30.0)
    for _ in range(40):
        progress.step(1.0)
    assert all(b > a for a, b in zip(fractions, fractions[1:])) and fractions[-1] < 1.0



def test_read_seed_reports_progress(monkeypatch):
    """ The progress goes forward from 0 to 1, and each word read is reported. """
    import seedsigner.helpers.handwriting as hw
    words = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about".split()
    monkeypatch.setattr(hw, "find_words", lambda image, step: fake_layout(words))
    fractions, read = [], []
    hw.read_seed(Image.new("L", (10, 10)), progress=fractions.append,
                 on_word=lambda layout, k, reading: read.append((k, reading.candidates[0][0])))
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert all(b > a for a, b in zip(fractions, fractions[1:]))
    assert read == list(enumerate(words))



def test_border_connected():
    """ The C flood fill marks the pixels that a breadth-first search from the border reaches. """
    from seedsigner.helpers.handwriting.page import border_connected
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = rng.random((rng.integers(1, 50), rng.integers(1, 50))) < rng.uniform(0.2, 0.9)
        h, w = mask.shape
        reached = np.zeros_like(mask)
        queue = [(y, x) for y in range(h) for x in range(w) if mask[y, x] and (y in (0, h - 1) or x in (0, w - 1))]
        for y, x in queue:
            reached[y, x] = True
        while queue:
            y, x = queue.pop()
            for v, u in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= v < h and 0 <= u < w and mask[v, u] and not reached[v, u]:
                    reached[v, u] = True
                    queue.append((v, u))
        assert np.array_equal(border_connected(mask), reached)
