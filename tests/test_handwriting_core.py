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
        "haller-1-capitals.npz": "2e9275f1acd0d347437e856160430e96a283ec484a48ab96a2ea1ca91531fe14",
        "haller-1-lowercase.npz": "5d340b354ced032992cb474f03bcc0b063f24ee8e1446d2953203b778652e509",
        "haller-1-detector.npz": "8ddddf19ee0088e8dcfe71ba12377167feeb6a626cd6edb17b0bed3b5f69fc5b",
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
                                 crop=lambda k, variant=0: crops[k] if variant == 0 else None, alternative=None,
                                 complete=True, number_crop=None, doubtful=set(), fingerprint=None, extra_words=[],
                                 lengths=[0.6 * len(w) for w in words])



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
    assert blend[0][1] == pytest.approx(expected.max() / hw.TEMPERATURES[hw.BLEND])



def test_second_readings_only_when_the_seed_is_not_valid(monkeypatch):
    """ A valid seed is read by the first network only; an invalid one also by the second network and the blend. """
    import seedsigner.helpers.handwriting as hw
    valid = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about".split()
    for words, second in ((valid, 0), (valid[:-1] + ["zoo"], len(valid))):
        calls = counting_networks(monkeypatch)
        monkeypatch.setattr(hw.detector, "find_words", lambda image, step, words=words, **_: fake_layout(words))
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
    monkeypatch.setattr(hw.detector, "find_words", lambda image, step, **_: fake_layout(words))
    fractions, read = [], []
    hw.read_seed(Image.new("L", (10, 10)), progress=fractions.append,
                 on_word=lambda layout, k, reading: read.append((k, reading.candidates[0][0])))
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert all(b > a for a, b in zip(fractions, fractions[1:]))
    assert read == list(enumerate(words))



def test_homography_maps_the_corners():
    """ The perspective coefficients (PIL convention: output to input) map each corner onto its source. """
    from seedsigner.helpers.handwriting.page import homography
    card = [(0, 0), (100, 0), (100, 140), (0, 140)]
    photo = [(12, 30), (95, 22), (105, 160), (8, 150)]
    a, b, c, d, e, f, g, h = homography(photo, card)
    for (x, y), (u, v) in zip(card, photo):
        w = g * x + h * y + 1
        assert np.allclose(((a * x + b * y + c) / w, (d * x + e * y + f) / w), (u, v), atol=1e-6)



def test_two_column_orders_both_valid():
    """ when the seed and its row by row permutation both pass the checksum, no word is certain """
    import random
    from seedsigner.helpers.handwriting import FIRST, Progress, WordReading, read_layout
    from seedsigner.helpers.handwriting.checksum import checksum_ok
    from embit import bip39
    rows = [[i for r in range(6) for i in (r, 6 + r)]]
    rng = random.Random(4)
    while True:
        seed = bip39.mnemonic_from_bytes(bytes(rng.getrandbits(8) for _ in range(16))).split()
        if checksum_ok([seed[i] for i in rows[0]]):
            break
    layout = types.SimpleNamespace(kind="list", count=12, orders=[list(range(12))] + rows, variants=1, complete=True)
    reader = types.SimpleNamespace(layout=layout, read=lambda k, variant, weights: WordReading(
        None, [(seed[k], 0.0)] + [(w, -30.0) for w in ("abandon", "ability", "able")]), numbered_order=lambda: None)
    reading = read_layout(reader, Progress(lambda p: None, 1.0), FIRST, 0.5)
    analysis = reading.analyze()
    assert reading.ambiguous_order and analysis.status == "uncertain" and len(analysis.uncertain) == 12

    # the line numbers tell the order written: no ambiguity
    for order in (0, 1):
        reader.numbered_order = lambda order=order: order
        reading = read_layout(reader, Progress(lambda p: None, 1.0), FIRST, 0.5)
        assert reading.numbered and not reading.ambiguous_order and reading.analyze().status == "valid"
        assert reading.analyze().mnemonic == [seed[i] for i in layout.orders[order]]
    reader.numbered_order = lambda: None

    layout.orders = [list(range(12))]
    reading = read_layout(reader, Progress(lambda p: None, 1.0), FIRST, 0.5)
    assert not reading.ambiguous_order and reading.analyze().status == "valid"



def test_number_probabilities():
    """ The digits of a line number are read, whatever punctuation follows them. """
    from seedsigner.helpers.handwriting.decoder import number_probabilities
    def frames(text):
        out = [0]
        for c in text:
            out += [ALPHABET.index(c) + 1 if c in ALPHABET else 0, 0]
        return out
    for text, number in (("12", 12), ("1", 1), ("7)", 7), ("11", 11), ("3j", 3)):
        p = number_probabilities(perfect_log_probs(frames(text)), 24)
        assert max(range(24), key=lambda i: p[i]) + 1 == number and p[number - 1] > 0.9
    # digits that are no number of the list: no number is probable
    assert max(number_probabilities(perfect_log_probs(frames("71")), 24)) < 0.05
    assert max(number_probabilities(perfect_log_probs(frames("ab")), 24)) < 0.05



def test_numbered_order_votes():
    """ Numbers read with assurance choose the order in which their words have that position, if no other
        order gets a vote. """
    from seedsigner.helpers.handwriting import LayoutReader
    columns, rows = list(range(12)), [i for r in range(6) for i in (r, 6 + r)]
    layout = types.SimpleNamespace(count=12, orders=[columns, rows], number_crop=None)
    reader = LayoutReader(layout)

    def written(order, k):
        return order.index(k) + 1

    reader.numbers = {k: (written(rows, k), 0.99) for k in range(12)}
    assert reader.numbered_order() == 1
    reader.numbers = {k: (written(columns, k), 0.99) for k in range(12)}
    assert reader.numbered_order() == 0
    # one number of the other order: no decision
    reader.numbers[3] = (written(rows, 3), 0.99)
    assert reader.numbered_order() is None
    # too few numbers read with assurance (the first one is the same in both orders)
    reader.numbers = {k: (written(rows, k), 0.99 if k < 3 else 0.5) for k in range(12)}
    assert reader.numbered_order() is None
    reader.numbers = {k: None for k in range(12)}
    assert reader.numbered_order() is None



def rendered_list(words: list[str]) -> Image.Image:
    """ Photo of a list of words printed in capitals on a light sheet. """
    font = ImageFont.truetype(FONT_PATH, 46)
    image = Image.new("L", (1200, 1600), 90)
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 150, 1050, 1450), fill=225)
    for i, word in enumerate(words):
        draw.text((260, 220 + i * 95), word.upper(), fill=35, font=font)
    return image



def test_detector_finds_the_words_of_a_list():
    """ The word detector finds the 12 lines of a list, which are then read. """
    import random
    from embit import bip39
    from seedsigner.helpers.handwriting import Progress, detector, read_words
    words = bip39.mnemonic_from_bytes(bytes(random.Random(3).getrandbits(8) for _ in range(16))).split()
    layout = detector.find_words(rendered_list(words))
    assert layout.kind == "list" and layout.count == 12 and layout.complete
    analysis = read_words(layout, Progress(lambda p: None, 1.0)).analyze()
    assert analysis.status == "valid" and analysis.mnemonic == words



def tight_two_column_list(words: list[str]) -> Image.Image:
    """
        Photo of a numbered list of two columns printed in capitals, the
        number of each word of the right column a third of a text height after
        the longest word of the left column: no empty corridor between them.
    """
    font = ImageFont.truetype(FONT_PATH, 46)
    image = Image.new("L", (1400, 1200), 90)
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 1300, 1100), fill=225)
    rows = len(words) // 2
    left = [f"{i + 1} {w.upper()}" for i, w in enumerate(words[:rows])]
    right_x = 180 + max(draw.textlength(t, font=font) for t in left) + 12
    for i in range(rows):
        draw.text((180, 230 + i * 120), left[i], fill=35, font=font)
        draw.text((right_x, 230 + i * 120), f"{rows + i + 1} {words[rows + i].upper()}", fill=35, font=font)
    return image



def test_small_list_detected_again_on_its_region():
    """ A list covering a small part of the photo is detected again on its region, then read. """
    import random
    from embit import bip39
    from seedsigner.helpers.handwriting import detector, read_seed
    words = bip39.mnemonic_from_bytes(bytes(random.Random(3).getrandbits(8) for _ in range(16))).split()
    photo = Image.new("L", (2400, 3200), 90)
    photo.paste(rendered_list(words), (300, 400))
    det = detector.detect(photo)
    assert detector.document_region(det, photo.size) is not None
    analysis = read_seed(photo).analyze()
    assert analysis.mnemonic == words and analysis.status == "valid"



def test_tight_two_column_list():
    """
        Two numbered columns with no empty corridor between them: the numbers give one slot per word. A sanity
        check on printed capitals, which the detector was not trained on (some other draws are found as 6 lines
        of 2 words); handwritten tight columns are measured by the benchmarks.
    """
    import random
    from embit import bip39
    from seedsigner.helpers.handwriting import read_seed
    words = bip39.mnemonic_from_bytes(bytes(random.Random(6).getrandbits(8) for _ in range(16))).split()
    analysis = read_seed(tight_two_column_list(words)).analyze()
    assert analysis.mnemonic == words and analysis.status in ("valid", "uncertain")



def test_trust_levels():
    """
    A reading is verified only by a matching fingerprint; otherwise its trust level
    follows its checksum analysis, and a fingerprint that matches no seed read lowers it.
    """
    from types import SimpleNamespace
    from seedsigner.helpers.handwriting import (TRUST_HIGH, TRUST_LOW, TRUST_MEDIUM, TRUST_VERIFIED, SeedReading,
        WordReading)
    words = ["abandon"] * 11 + ["about"]

    def reading(second: float, last: str = "about"):
        return SeedReading("list", [WordReading(None, [(w, 0.0), ("zoo", second)]) for w in words[:11]]
                           + [WordReading(None, [(last, 0.0), ("zoo", second)])])

    sure = reading(-20.0)
    assert sure.analyze().status == "valid" and sure.trust().level == TRUST_HIGH
    assert not sure.trust().verified

    unsure = reading(-0.5)
    analysis = unsure.analyze()
    assert analysis.status == "uncertain"
    assert unsure.trust().level == (TRUST_MEDIUM if analysis.confidence >= 0.5 else TRUST_LOW)

    invalid = reading(-20.0, last="abandon")
    assert invalid.analyze().status == "invalid" and invalid.trust().level == TRUST_LOW

    mismatch = reading(-20.0)
    mismatch.fingerprint = SimpleNamespace(status="mismatch")
    assert mismatch.trust().level == TRUST_LOW

    verified = SeedReading("list", [WordReading(None, [(w, 0.0)]) for w in words])
    verified.verified = True
    assert verified.trust().level == TRUST_VERIFIED and verified.trust().verified
    assert verified.trust({11: "about"}).verified
    assert not verified.trust({11: "actual"}).verified



def test_unread_faint_ink():
    """ Faint letters before a dark word are measured; a word written evenly has none. """
    from seedsigner.helpers.handwriting.preprocess import unread_ink
    font = ImageFont.truetype(FONT_PATH, 46)
    even = Image.new("L", (420, 90), 215)
    ImageDraw.Draw(even).text((30, 15), "BRIDGE", fill=35, font=font)
    faded = Image.new("L", (420, 90), 215)
    draw = ImageDraw.Draw(faded)
    draw.text((30, 15), "BR", fill=175, font=font)
    draw.text((30 + draw.textlength("BR", font=font), 15), "IDGE", fill=35, font=font)
    assert unread_ink(even) < 0.2
    assert unread_ink(faded) >= 1.0

