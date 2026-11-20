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
        "haller-1-detector.npz": "b586959af85a3d016551f9e9058c088b7942a00331e3fb78eef6368adc7ad0d4",
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
                                 complete=True, number_crop=None)



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
        monkeypatch.setattr(hw, "find_words", lambda image, step, words=words, **_: fake_layout(words))
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
    monkeypatch.setattr(hw, "find_words", lambda image, step, **_: fake_layout(words))
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



def test_local_max():
    from seedsigner.helpers.handwriting.page import local_max
    a = np.random.default_rng(1).integers(0, 255, (20, 30)).astype(np.float32)
    padded = np.pad(a, 2, mode="edge")
    expected = np.array([[padded[y:y + 5, x:x + 5].max() for x in range(30)] for y in range(20)])
    assert np.array_equal(local_max(a, 2), expected)



def test_row_line_ignores_handwriting_peaks():
    """ the line crossing the most strips is kept, not the strokes of the words around it """
    from seedsigner.helpers.handwriting.page import row_line
    x = np.repeat(np.arange(6) * 10.0, 3)
    strokes = [(14, 27), (25, 16), (13, 26), (27, 15), (16, 24), (24, 13)]
    y = np.array([[20 + 0.05 * s, *strokes[s]] for s in range(6)]).ravel()
    intercept, slope, columns = row_line(x, y, 1.0)
    assert abs(intercept - 20) < 0.01 and abs(slope - 0.005) < 0.001 and len(columns) == 6



def test_homography_least_squares():
    from seedsigner.helpers.handwriting.page import apply_homography, homography
    card = [(0, 0), (100, 0), (100, 140), (0, 140)]
    photo = [(12, 30), (95, 22), (105, 160), (8, 150)]
    exact = homography(photo, card)
    grid = [(x, y) for x in (0, 50, 100) for y in (0, 70, 140)]
    fitted = homography(apply_homography(exact, grid), grid)
    assert np.allclose(fitted, exact, atol=1e-9)



def test_right_column_of_a_24_word_card():
    """ the lines of the right column of the 24-word card fall on those of the 12-word card `quad` """
    from seedsigner.helpers.handwriting.page import (CARD_CORNERS, CARD_LINES, apply_homography, homography,
                                                     right_column_card24)
    quad = [(100, 80), (500, 90), (510, 650), (90, 640)]
    wide = right_column_card24(quad)
    to_photo12, to_photo24 = homography(quad, CARD_CORNERS["card12"]), homography(wide, CARD_CORNERS["card24"])
    for (_, x12, y12), (_, x24, y24) in zip(CARD_LINES["card12"], CARD_LINES["card24"][12:]):
        assert np.allclose(apply_homography(to_photo12, [(x12, y12)]), apply_homography(to_photo24, [(x24, y24)]),
                           atol=1e-6)



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



def test_each_search_checks_the_seed_of_the_other():
    """ The seed of the detector is not valid if the rules read another plausible seed or a sure word differently;
        the seed of the rules is not valid if the detector read another valid seed or, its reading not being
        invalid, a sure word differently. """
    import math
    from embit import bip39
    from seedsigner.helpers.handwriting import SeedReading, WordReading, agrees
    seed = bip39.mnemonic_from_bytes(bytes(range(16))).split()
    other = bip39.mnemonic_from_bytes(bytes(range(1, 17))).split()

    def reading(words, p_top=1.0, complete=True):
        """ Each word read first with probability p_top, then "zoo" """
        second = math.log(1 / p_top - 1) if p_top < 1 else -50.0
        return SeedReading("list", [WordReading(None, [(w, 0.0), ("zoo", second)]) for w in words], complete=complete)

    first = reading(seed)
    assert first.analyze().status == "valid"
    assert agrees(first, None, [])                                          # the other search found no words
    assert not agrees(first, reading(other, 0.5), [other])                  # another plausible seed
    assert agrees(first, reading(seed, 0.6), [seed])                        # the same seed, uncertain
    assert not agrees(first, reading(other[:1] + seed[1:], complete=False), [])   # a sure word differs
    assert agrees(first, reading(other[:1] + seed[1:], 0.6, complete=False), [])  # an unsure one
    # weak check: only a valid seed, or a sure word of a reading that is not invalid, contradicts
    assert reading(other, 0.5).analyze().status != "valid"
    assert agrees(first, reading(other, 0.5), [other], weak=True)
    assert not agrees(first, reading(other), [other], weak=True)
    garbled = reading(other[:1] + seed[1:], complete=False)
    assert garbled.analyze().status == "invalid"
    assert agrees(first, garbled, [], weak=True)



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



def test_aligned_columns_of_a_tight_list():
    """ Two columns with no empty corridor between them: the rules find one line per row; cut where the right
    column starts, the rows give the words of the seed. """
    import random
    from embit import bip39
    from seedsigner.helpers.handwriting import page, read_seed
    words = bip39.mnemonic_from_bytes(bytes(random.Random(5).getrandbits(8) for _ in range(16))).split()
    photo = tight_two_column_list(words)
    half = photo.reduce(2)
    ink = page.ink_mask(half)
    assert sum(len(c) for c in page.find_lines(half, ink)) == 6
    columns = page.aligned_columns(half, ink)
    assert [len(c) for c in columns] == [6, 6]
    for left, right in zip(*columns):
        assert left[2] <= right[0] and abs(left[1] - right[1]) <= 2
    with pytest.raises(page.WordsNotFound):
        page.find_words(photo)
    layout = page.find_aligned_words(photo)
    assert layout.kind == "list" and layout.count == 12
    analysis = read_seed(photo).analyze()
    assert analysis.mnemonic == words and analysis.status in ("valid", "uncertain")



def test_aligned_columns_not_found_in_one_column():
    """ A list of one column whose words are cut nowhere: no aligned start leaves enough text on both sides. """
    from seedsigner.helpers.handwriting import page
    half = rendered_list(["attitude", "surface", "betray", "supply", "valve", "erupt"]).reduce(2)
    assert page.aligned_columns(half, page.ink_mask(half)) is None
