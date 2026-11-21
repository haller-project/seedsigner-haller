"""
    Layout rules of the detector on synthetic boxes: reading order by bands,
    words written in rows, word box running over the start of the next column.
"""
import pytest

np = pytest.importorskip("numpy")

from seedsigner.helpers.handwriting import detector  # noqa: E402


def box(a0, a1, c, h=10.0, pixels=100):
    b = object.__new__(detector.Box)
    b.ux, b.uy = 1.0, 0.0
    b.a0, b.a1, b.c, b.h = float(a0), float(a1), float(c), float(h)
    b.score, b.pixels = 0.9, pixels
    return b


def column(x, centers, length=60):
    return [box(x, x + length, c) for c in centers]


def test_band_order_two_bands():
    centers = [0, 20, 40, 75, 95, 115]
    order = detector.band_order([column(0, centers), column(200, centers)])
    assert order == [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]


def test_band_order_regular_columns():
    centers = [0, 20, 40, 60, 80, 100]
    assert detector.band_order([column(0, centers), column(200, centers)]) is None


def test_band_order_gaps_at_different_rows():
    left = column(0, [0, 20, 40, 75, 95, 115])
    right = column(200, [0, 35, 55, 75, 95, 115])
    assert detector.band_order([left, right]) is None


def test_band_order_offered_only_with_numbers():
    centers = [0, 20, 40, 75, 95, 115]
    columns = [column(0, centers), column(200, centers)]
    assert len(detector.candidate_orders(columns)) == 2
    orders = detector.candidate_orders(columns, bands=True)
    assert orders[2] == [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]


def test_row_words_reads_rows_left_to_right():
    boxes = []
    for r, n in enumerate((5, 5, 2)):
        boxes += [box(100 * k, 100 * k + 50, 30 * r) for k in range(n)]
    rows = detector.row_words(list(reversed(boxes)))
    assert [len(r) for r in rows] == [5, 5, 2]
    assert [b.a0 for b in rows[0]] == [0, 100, 200, 300, 400]
    assert rows[2][0].c == 60


def test_row_words_not_a_list_in_columns():
    boxes = column(0, range(0, 240, 20)) + column(200, range(0, 240, 20))
    assert detector.row_words(boxes) is None


def test_aligned_columns_cuts_a_box_over_the_next_column():
    rows = []
    for r in range(6):
        c = 20 * r
        if r == 4:
            rows.append(box(0, 200, c, pixels=300))  # "5 BRAND11ALTER": one box over both columns
        else:
            rows += [box(0, 60, c), box(120, 190, c)]
    columns = detector.aligned_columns(rows)
    assert columns is not None and [len(c) for c in columns] == [6, 6]
    left, right = columns[0][4], columns[1][4]
    assert left.a0 == 0 and 60 < left.a1 < 120
    assert right.a0 == 120 and right.a1 == 200
