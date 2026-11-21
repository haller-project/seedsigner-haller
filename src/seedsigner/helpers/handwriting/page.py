"""
    Words found on a photo (WordLayout) and the failure to find them
    (WordsNotFound), shared by the word detector (detector.py) and the reading
    (__init__.py).
"""
import numpy as np
from PIL import Image

SEED_LENGTHS = (12, 15, 18, 21, 24)



class WordsNotFound(Exception):
    """
        No card was found on the photo, and no list with a valid number of lines.
        `at_edge`: the ink of the list reaches the edge of the photo, so lines are
        probably cut off by the frame.
        `line_count`: number of lines found in the list, if any.
        `card`: the words are those of a card, part of which is out of the photo.
    """
    def __init__(self, message: str, at_edge: bool = False, line_count: int = 0, card: bool = False):
        super().__init__(message)
        self.at_edge = at_edge
        self.line_count = line_count
        self.card = card



class WordLayout:
    """
        Words found on a photo.

        kind: "card12", "card24" or "list"
        count: number of words
        orders: possible reading orders (lists of word indices); the first one
            reads column by column, a second one row by row is given for two
            columns of equal length
        variants: number of crop variants per word (crop(k, variant))
        alternative: None, or a function returning the layout to read if this
            one gives no valid seed; it raises WordsNotFound if there is none
        complete: False if text is left out of the crops, so that the number
            of words is uncertain, or if the card is too small in the photo:
            no word is then certain unless confirmed
        number_crop: None, or a function returning the grayscale crop of the
            number written before word k, or None if none was found
        fingerprint: None, or the grayscale crop of a line left out of the
            words because it reads as a master key fingerprint (see
            spare_lines)
    """
    def __init__(self, kind: str, count: int, orders: list[list[int]], variants: int, crop, outline,
                 alternative=None, complete: bool = True, number_crop=None):
        self.kind = kind
        self.count = count
        self.orders = orders
        self.variants = variants
        self._crop = crop
        self._outline = outline
        self.alternative = alternative
        self.complete = complete
        self.number_crop = number_crop
        self.fingerprint = None


    def crop(self, k: int, variant: int = 0) -> Image.Image | None:
        """
            Grayscale crop of word k. Variant 0 is the first reading, the
            others are wider or taller bands around the word (None when
            there is no such crop).
        """
        return self._crop(k, variant)


    def outline(self, k: int) -> list[tuple[float, float]]:
        """
            Corners of crop k (variant 0) on the photo, clockwise from the top
            left.
        """
        return self._outline(k)



def no_step(cost: float):
    """
        Default `step` function: the progress is not reported.
    """



def homography(src: list, dst: list) -> np.ndarray:
    """
        Pillow PERSPECTIVE coefficients mapping the points dst (output) to the
        points src (input); least squares fit beyond 4 points.
    """
    a, b = [], []
    for (u, v), (x, y) in zip(dst, src):
        a.append([u, v, 1, 0, 0, 0, -x * u, -x * v])
        b.append(x)
        a.append([0, 0, 0, u, v, 1, -y * u, -y * v])
        b.append(y)
    if len(src) > 4:
        return np.linalg.lstsq(np.array(a, float), np.array(b, float), rcond=None)[0]
    return np.linalg.solve(np.array(a, float), np.array(b, float))


def spare_lines(columns: list[list[tuple[int, int, int, int]]]) -> list[tuple[int, int]]:
    """
        (column, line) of the lines of a list that may be a fingerprint
        written with the words, when the list has one line more than a valid
        number of words: the first line of the list and the last line of
        each column. With two columns, only lines whose removal leaves two
        columns of the same length.
    """
    found = []
    for c, i in [(0, 0)] + [(c, len(column) - 1) for c, column in enumerate(columns)]:
        lengths = [len(column) - (j == c) for j, column in enumerate(columns)]
        if (c, i) not in found and (len(columns) == 1 or len(set(x for x in lengths if x)) == 1):
            found.append((c, i))
    return found
