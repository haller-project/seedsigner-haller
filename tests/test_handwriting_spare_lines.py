"""
    Lines tried as the fingerprint and rows of words, on synthetic boxes:
    number boxes at the end of a fingerprint line, head of the second column,
    first or last row of a grid, slanted rows close to each other.
"""
import pytest

np = pytest.importorskip("numpy")

from seedsigner.helpers.handwriting import detector, page  # noqa: E402


def box(a0, a1, c, h=10.0, pixels=100):
    b = object.__new__(detector.Box)
    b.ux, b.uy = 1.0, 0.0
    b.a0, b.a1, b.c, b.h = float(a0), float(a1), float(c), float(h)
    b.score, b.pixels = 0.9, pixels
    return b


def test_fingerprint_line_widened_to_its_last_digits():
    line = box(0, 70, 0)
    end = box(66, 82, 1)       # last digits read as a number, against the end of the line
    other = box(150, 160, 0)   # number of the next column
    wide = detector.with_number(line, [end, other], after=True)
    assert (wide.a0, wide.a1) == (0, 82)
    assert detector.with_number(line, [end, other]) is None


def test_fingerprint_line_widened_on_both_sides():
    line = box(20, 80, 0)
    wide = detector.with_number(line, [box(5, 15, 0), box(78, 90, 0)], after=True)
    assert (wide.a0, wide.a1) == (5, 90)


def test_spare_lines_head_of_second_column():
    columns = [[1] * 6, [2] * 7]
    assert (1, 0) in page.spare_lines(columns)
    assert page.spare_lines([[1] * 7]) == [(0, 0), (0, 6)]


def test_extra_row_above_a_tight_grid():
    # 4 rows of 3 words whose columns are too close for group_lines, and a fingerprint row above
    words = [box(x, x + 50, 30 + 20 * r) for r in range(4) for x in (0, 62, 124)]
    fp = box(0, 90, 5)
    found = detector.extra_rows(words + [fp])
    assert len(found) == 1
    line, pieces, columns = found[0]
    assert pieces == [fp] and sum(len(c) for c in columns) == 12


def test_no_extra_row_without_valid_rest():
    words = [box(x, x + 50, 30 + 20 * r) for r in range(4) for x in (0, 62, 124)]
    assert detector.extra_rows(words) == []


def test_slanted_rows_followed_box_to_box():
    # two rows of 6 words rising along the text, 1.5 heights apart: grouped by their mean center, each row breaks
    rows = [[box(22 * k, 22 * k + 18, base + 1.6 * k, h=7.5) for k in range(6)] for base in (90, 101)]
    boxes = [b for row in rows for b in row]
    assert detector.row_words(boxes) is None or sum(len(r) for r in detector.row_words(boxes)) != 12
    chained = detector.row_words(boxes, chained=True)
    assert [len(r) for r in chained] == [6, 6]
    assert [b.a0 for b in chained[0]] == [22 * k for k in range(6)]


def test_words_not_found_small():
    e = page.WordsNotFound("words too small", small=True)
    assert e.small and not e.at_edge and e.line_count == 0


def test_first_pass_box_moved_to_the_region():
    # whole photo seen at scale 0.1, region from (100, 200) seen at scale 0.25 (output pixels: half the input)
    first = object.__new__(detector.Detection)
    first.words = [box(10, 30, 20, h=4)]
    first.to_photo = lambda points: [(2 * u / 0.1, 2 * v / 0.1) for u, v in points]
    det = object.__new__(detector.Detection)
    det.first = first
    det.from_photo = lambda points: [((x - 100) * 0.25 / 2, (y - 200) * 0.25 / 2) for x, y in points]
    (moved,) = detector.first_pass_words(det)
    # photo: along 200-600, across 400 ; region output: along (200-100)*0.125 = 12.5 .. 62.5, across 25
    assert (moved.a0, moved.a1, moved.c, moved.h) == pytest.approx((12.5, 62.5, 25.0, 10.0))
