"""
    Lines extended by the ink of their row (InkReach, ink_extend), empty
    numbered slots holding strokes, and a title row across the columns left
    out (bridged_columns), on synthetic boxes and images.
"""
import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

from seedsigner.helpers.handwriting import detector  # noqa: E402


def box(a0, a1, c, h=10.0, pixels=100):
    b = object.__new__(detector.Box)
    b.ux, b.uy = 1.0, 0.0
    b.a0, b.a1, b.c, b.h = float(a0), float(a1), float(c), float(h)
    b.score, b.pixels = 0.9, pixels
    return b


class Identity:
    """ Detection whose output pixels are the pixels of the photo. """
    def to_photo(self, points):
        return [(float(x), float(y)) for x, y in points]


def page(strokes, rules=(), size=(400, 100), dark_until=0):
    """ White page with vertical strokes 8 px high centered on y = 50 (x positions), ruled lines (y). """
    im = Image.new("L", size, 235)
    d = ImageDraw.Draw(im)
    if dark_until:
        d.rectangle((0, 0, dark_until, size[1]), fill=40)
    for x in strokes:
        d.rectangle((x, 46, x + 1, 54), fill=30)
    for y in rules:
        d.line((0, y, size[0], y), fill=30, width=1)
    return im


def reach(im):
    return detector.InkReach(im, Identity(), 235)


def test_reach_extends_over_letters_left_out():
    # a word of strokes every 4 px from 100 to 196, the box on 140-170 only
    im = page(range(100, 200, 4))
    a0, a1 = reach(im).reach(box(140, 170, 50), -np.inf, np.inf)
    assert a0 < 105 and a1 > 195


def test_reach_stops_at_a_word_gap_and_at_the_next_box():
    im = page(list(range(100, 160, 4)) + list(range(200, 260, 4)))
    # 40 px of white = 4 heights: another word, not taken in
    assert reach(im).reach(box(100, 160, 50), -np.inf, np.inf)[1] < 175
    # strokes up to 260 but a box starts at 180
    im = page(range(100, 260, 4))
    assert reach(im).reach(box(100, 140, 50), -np.inf, 180)[1] <= 180


def test_reach_keeps_a_whole_word():
    im = page(range(100, 160, 4))
    assert reach(im).reach(box(97, 163, 50), -np.inf, np.inf) == (97, 163)


def test_reach_ignores_ruled_lines_and_the_edge_of_the_sheet():
    im = page(range(100, 160, 4), rules=(52,), dark_until=60)
    a0, a1 = reach(im).reach(box(97, 163, 50), -np.inf, np.inf)
    assert (a0, a1) == (97, 163)


def test_ink_extend_bounded_by_the_next_column():
    im = page(range(100, 300, 4))
    left = [box(100, 140, 50)]
    right = [box(260, 300, 50)]
    detector.ink_extend(reach(im), [left, right], left + right)
    assert left[0].a1 <= 255


def test_slot_finds_a_faint_word():
    im = page(range(150, 190, 4))
    found = reach(im).slot(box(110, 300, 50))
    assert found is not None and found[0] < 152 and found[1] > 188
    assert reach(page([])).slot(box(110, 300, 50)) is None


def test_bridged_columns_leaves_the_title_out():
    rows = range(40, 160, 20)
    title = [box(20, 330, 0)]
    boxes = title + [box(0, 60, c) for c in rows] + [box(260, 320, c) for c in rows]
    assert detector.group_lines([detector.copy_box(b) for b in boxes]) and \
        len(detector.group_lines([detector.copy_box(b) for b in boxes])) == 1
    columns, removed = detector.bridged_columns(boxes)
    assert [len(c) for c in columns] == [6, 6] and removed == [title]


def test_bridged_columns_none_for_a_list_in_columns():
    rows = range(40, 160, 20)
    boxes = [box(0, 60, c) for c in rows] + [box(260, 320, c) for c in rows]
    assert detector.bridged_columns(boxes) is None


def test_ink_extend_reports_each_line():
    # display only: one call per line, in reading order, with the number of lines
    im = page(range(100, 300, 4))
    left = [box(100, 140, 50), box(100, 140, 80)]
    right = [box(260, 300, 50)]
    seen = []
    detector.ink_extend(reach(im), [left, right], left + right, on_line=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 3), (2, 3), (3, 3)]
