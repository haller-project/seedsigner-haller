"""
    C library of the handwriting reader (handwriting.c): word network,
    dictionary decoding, word detector, and the flood fill and connected
    components used to find the card and the words. It is the only
    implementation of these steps. SeedSigner OS cross-compiles it with
    the image; elsewhere, run `make` in this directory.

    This module does not use numpy: the reference vectors are checked with it
    alone (tests/test_handwriting_vectors.py).
"""
import ctypes
import logging
import os

logger = logging.getLogger(__name__)

LIBRARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhandwriting.so")

# Output classes of the word network: 0 is the CTC blank, then the letters and the digits
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
NUM_CLASSES = 1 + len(ALPHABET)
FIRST_DIGIT_CLASS = 1 + ALPHABET.index("0")

# Layers of the word network with int16 weights, int32 biases and requantization multipliers
LAYERS = ("b1", "b2", "b3", "b4", "t1", "t2")

_library = []



class LibraryNotBuilt(RuntimeError):
    pass



class Network(ctypes.Structure):
    _fields_ = [
        ("c1", ctypes.c_int), ("c2", ctypes.c_int), ("c3", ctypes.c_int), ("hidden", ctypes.c_int),
        ("w", ctypes.c_void_p * 6), ("b", ctypes.c_void_p * 6), ("m", ctypes.c_void_p * 6),
        ("wo", ctypes.c_void_p), ("so", ctypes.c_void_p), ("bo", ctypes.c_void_p),
    ]



# Layers of the word detector, in the order of handwriting.h
DETECTOR_LAYERS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "d4", "d3", "d2", "d1")
DETECTOR_SIZE = (288, 384)  # width, height of the input
DETECTOR_OUTPUTS = 8



class Detector(ctypes.Structure):
    _fields_ = [
        ("channels", ctypes.c_int * 12),
        ("w", ctypes.c_void_p * 12), ("b", ctypes.c_void_p * 12), ("m", ctypes.c_void_p * 12),
        ("wo", ctypes.c_void_p), ("so", ctypes.c_void_p), ("bo", ctypes.c_void_p),
    ]



class Trie(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_int), ("words", ctypes.c_int),
        ("parent", ctypes.c_void_p), ("letter", ctypes.c_void_p), ("word_end", ctypes.c_void_p),
    ]



def load() -> ctypes.CDLL:
    """
        The shared library, loaded once. Raises LibraryNotBuilt if it has not
        been built.
    """
    if not _library:
        try:
            lib = ctypes.CDLL(LIBRARY_PATH)
        except OSError as e:
            message = f"Handwriting library not loaded ({e}): run make in {os.path.dirname(LIBRARY_PATH)}"
            logger.error(message)
            raise LibraryNotBuilt(message) from e
        lib.hw_network_work_size.restype = ctypes.c_size_t
        lib.hw_network_work_size.argtypes = [ctypes.POINTER(Network), ctypes.c_int]
        lib.hw_network_logits.restype = ctypes.c_int
        lib.hw_network_logits.argtypes = [ctypes.POINTER(Network), ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_void_p]
        lib.hw_log_softmax.restype = None
        lib.hw_log_softmax.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.hw_decoder_work_size.restype = ctypes.c_size_t
        lib.hw_decoder_work_size.argtypes = [ctypes.POINTER(Trie)]
        lib.hw_word_scores.restype = None
        lib.hw_word_scores.argtypes = [ctypes.POINTER(Trie), ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
                                       ctypes.c_void_p, ctypes.c_void_p]
        lib.hw_detector_work_size.restype = ctypes.c_size_t
        lib.hw_detector_work_size.argtypes = [ctypes.POINTER(Detector)]
        lib.hw_detector_maps.restype = None
        lib.hw_detector_maps.argtypes = [ctypes.POINTER(Detector), ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p]
        lib.hw_components.restype = ctypes.c_int
        lib.hw_components.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        lib.hw_border_connected.restype = None
        lib.hw_border_connected.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                            ctypes.c_void_p]
        _library.append(lib)
    return _library[0]



def address(buffer) -> int:
    """
        Address of a writable C-contiguous buffer (numpy array, bytearray, ctypes array).
    """
    view = memoryview(buffer).cast("B")
    return ctypes.addressof(ctypes.c_char.from_buffer(view)) if view.nbytes else 0



def network(arrays: dict) -> Network:
    """
        Network structure pointing to the weight arrays (name -> writable
        buffer in the layout of handwriting.h, as written by haller.quantize),
        which it keeps referenced.
    """
    count = lambda name: memoryview(arrays[name]).nbytes // 4  # int32 biases: one per output channel
    net = Network(count("b1.bias"), count("b2.bias"), count("b3.bias"), count("t1.bias"))
    for i, layer in enumerate(LAYERS):
        net.w[i], net.b[i], net.m[i] = (address(arrays[layer + k]) for k in (".weight", ".bias", ".mul"))
    net.wo, net.so, net.bo = (address(arrays[k]) for k in ("out.weight", "out.scale", "out.bias"))
    net.arrays = arrays
    return net



def detector(arrays: dict) -> Detector:
    """
        Detector structure pointing to the weight arrays (name -> writable
        buffer in the layout of handwriting.h, as written by
        haller.detector.quantize), which it keeps referenced.
    """
    net = Detector()
    for i, layer in enumerate(DETECTOR_LAYERS):
        net.channels[i] = memoryview(arrays[layer + ".bias"]).nbytes // 4
        net.w[i], net.b[i], net.m[i] = (address(arrays[layer + k]) for k in (".weight", ".bias", ".mul"))
    net.wo, net.so, net.bo = (address(arrays[k]) for k in ("head.weight", "head.scale", "head.bias"))
    net.arrays = arrays
    return net



def trie(words: list[str]) -> Trie:
    """
        Prefix tree of `words` (letters of ALPHABET), nodes stored parent before
        child, node 0 being the root.
    """
    index = {"": 0}
    parent, letter = [0], [0]
    for prefix in sorted({w[:i] for w in words for i in range(1, len(w) + 1)}, key=lambda s: (len(s), s)):
        index[prefix] = len(parent)
        parent.append(index[prefix[:-1]])
        letter.append(ALPHABET.index(prefix[-1]) + 1)
    arrays = ((ctypes.c_int32 * len(parent))(*parent), (ctypes.c_int16 * len(letter))(*letter),
              (ctypes.c_int32 * len(words))(*(index[w] for w in words)))
    t = Trie(len(parent), len(words), *(ctypes.addressof(a) for a in arrays))
    t.arrays = arrays
    return t
