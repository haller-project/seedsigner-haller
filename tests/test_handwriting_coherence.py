"""
    Card kernels without the shape of a card (card_shaped) and numbers read
    that contradict the order of the words (numbers_contradict).
"""
import math

import pytest

np = pytest.importorskip("numpy")

from seedsigner.helpers.handwriting import detector  # noqa: E402


def pixels_of(mask):
    return np.argwhere(mask)


def test_a_filled_rectangle_is_a_card():
    mask = np.zeros((200, 200), bool)
    mask[40:110, 30:150] = True  # 8 400 pixels, compactness 1
    assert detector.card_shaped(pixels_of(mask))


def test_a_turned_rectangle_is_a_card():
    yy, xx = np.mgrid[0:300, 0:300]
    a = math.radians(30)
    u = (xx - 150) * math.cos(a) + (yy - 150) * math.sin(a)
    v = -(xx - 150) * math.sin(a) + (yy - 150) * math.cos(a)
    assert detector.card_shaped(pixels_of((abs(u) < 70) & (abs(v) < 40)))


def test_a_small_kernel_is_no_card():
    mask = np.zeros((200, 200), bool)
    mask[40:70, 30:80] = True  # 1 500 pixels
    assert not detector.card_shaped(pixels_of(mask))


def test_a_sparse_kernel_is_no_card():
    # a frame of 12 px around a 120 x 120 square: 5 184 pixels, a third of its rectangle
    mask = np.zeros((200, 200), bool)
    mask[30:150, 30:150] = True
    mask[42:138, 42:138] = False
    assert not detector.card_shaped(pixels_of(mask))


def test_numbers_in_order_agree():
    order = list(range(12))
    assert not detector.numbers_contradict([order], {k: k + 1 for k in range(12)})
    # one number misread: not enough
    assert not detector.numbers_contradict([order], {0: 1, 1: 7, 2: 3, 3: 4})


def test_numbers_of_a_list_missing_its_first_words_contradict():
    # 21 lines read, the first five words missing: lines numbered from 6
    order = list(range(21))
    values = {k: k + 6 for k in range(3)} | {k: k + 5 for k in range(5, 12)}
    assert detector.numbers_contradict([order], values)


def test_the_best_order_is_taken():
    by_columns, by_rows = [0, 1, 2, 3], [0, 2, 1, 3]
    values = {1: 3, 2: 2}
    assert not detector.numbers_contradict([by_columns, by_rows], values)
    assert detector.numbers_contradict([by_columns], values)


def test_no_number_read_no_contradiction():
    assert not detector.numbers_contradict([[0, 1, 2]], {})
