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
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

LIBRARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhandwriting.so")
# SHA-256 of every weight file, by file name (haller-weights.json, next to the weight files)
WEIGHTS_MANIFEST = "haller-weights.json"

# Output classes of the word network: 0 is the CTC blank, then the letters and the digits
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
NUM_CLASSES = 1 + len(ALPHABET)
FIRST_DIGIT_CLASS = 1 + ALPHABET.index("0")

# Layers of the word network with int16 weights, int32 biases and requantization multipliers
LAYERS = ("b1", "b2", "b3", "b4", "t1", "t2")

_library = []



class LibraryNotBuilt(RuntimeError):
    pass



class WeightsError(ValueError):
    """
        A weight file that is not listed in WEIGHTS_MANIFEST, does not match
        its SHA-256 there, or whose arrays do not have the sizes the C library
        reads.
    """



class Network(ctypes.Structure):
    _fields_ = [
        ("c1", ctypes.c_int), ("c2", ctypes.c_int), ("c3", ctypes.c_int), ("hidden", ctypes.c_int),
        ("w", ctypes.c_void_p * 6), ("b", ctypes.c_void_p * 6), ("m", ctypes.c_void_p * 6),
        ("wo", ctypes.c_void_p), ("so", ctypes.c_void_p), ("bo", ctypes.c_void_p),
    ]



# Layers of the word detector, in the order of handwriting.h
DETECTOR_LAYERS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "d4", "d3", "d2", "d1")
DETECTOR_SIZE = (288, 384)  # width, height of the input
DETECTOR_OUTPUTS = 11



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



def read_weights(path: str) -> bytes:
    """
        Content of the weight file `path`, checked against its SHA-256 in
        WEIGHTS_MANIFEST (in the same directory) before anything is read from
        it. Raises WeightsError if the file is not listed there or does not
        match.
    """
    directory, name = os.path.split(os.path.abspath(path))
    manifest = os.path.join(directory, WEIGHTS_MANIFEST)
    try:
        with open(manifest) as f:
            expected = json.load(f)["weights"].get(name)
    except (OSError, ValueError, KeyError, AttributeError) as e:
        raise WeightsError(f"Handwriting weights {name}: manifest {manifest} not read ({e})") from e
    if expected is None:
        raise WeightsError(f"Handwriting weights {name}: not listed in {manifest}")
    with open(path, "rb") as f:
        content = f.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected:
        message = f"Handwriting weights {name}: SHA-256 {digest} instead of {expected} ({manifest}); file corrupted or replaced"
        logger.error(message)
        raise WeightsError(message)
    return content



def check_sizes(arrays: dict, sizes: dict, what: str):
    """
        Raises WeightsError unless every buffer of `sizes` (name -> number of
        values, size of a value in bytes) is in `arrays`, C-contiguous, with
        exactly that many bytes: the C library reads that many from its
        address.
    """
    for name, (count, itemsize) in sizes.items():
        if name not in arrays:
            raise WeightsError(f"{what}: array {name} missing")
        view = memoryview(arrays[name])
        if not view.c_contiguous or view.nbytes != count * itemsize:
            raise WeightsError(f"{what}: array {name} has {view.nbytes} bytes, {count} values of {itemsize} bytes "
                               f"({count * itemsize} bytes) expected")



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
    for name in ("b1.bias", "b2.bias", "b3.bias", "t1.bias"):
        if name not in arrays:
            raise WeightsError(f"Word network: array {name} missing")
    count = lambda name: memoryview(arrays[name]).nbytes // 4  # int32 biases: one per output channel
    c1, c2, c3, hidden = (count(name) for name in ("b1.bias", "b2.bias", "b3.bias", "t1.bias"))
    if not all(c > 0 and c % 2 == 0 for c in (c1, c2, c3, hidden)):
        raise WeightsError(f"Word network: channels {c1}, {c2}, {c3}, {hidden} (even, positive numbers expected)")
    # outputs and inputs of each layer (handwriting.h): 3x3 convolutions, then 1-d convolutions of width 3 on the
    # columns of 2 * c3 values
    shapes = {"b1": (c1, 9, 1), "b2": (c2, 9, c1), "b3": (c3, 9, c2), "b4": (c3, 9, c3),
              "t1": (hidden, 3, 2 * c3), "t2": (hidden, 3, hidden)}
    sizes = {"out.weight": (NUM_CLASSES * hidden, 2), "out.scale": (NUM_CLASSES, 4), "out.bias": (NUM_CLASSES, 4)}
    for layer, (outputs, taps, inputs) in shapes.items():
        sizes.update({layer + ".weight": (outputs * taps * inputs, 2), layer + ".bias": (outputs, 4),
                      layer + ".mul": (outputs, 4)})
    check_sizes(arrays, sizes, "Word network")
    net = Network(c1, c2, c3, hidden)
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
    channels = []
    for layer in DETECTOR_LAYERS:
        if layer + ".bias" not in arrays:
            raise WeightsError(f"Word detector: array {layer}.bias missing")
        channels.append(memoryview(arrays[layer + ".bias"]).nbytes // 4)
    if not all(c > 0 and c % 4 == 0 for c in channels):
        raise WeightsError(f"Word detector: channels {channels} (positive multiples of 4 expected)")
    # inputs of each layer (handwriting.h): the previous one, and for d4-d1 the encoder layer concatenated
    c = dict(zip(DETECTOR_LAYERS, channels))
    inputs = {"e1": 1, "e2": c["e1"], "e3": c["e2"], "e4": c["e3"], "e5": c["e4"], "e6": c["e5"], "e7": c["e6"],
              "e8": c["e7"], "d4": c["e6"] + c["e8"], "d3": c["e3"] + c["d4"], "d2": c["e2"] + c["d3"],
              "d1": c["e1"] + c["d2"]}
    sizes = {"head.weight": (DETECTOR_OUTPUTS * 9 * c["d1"], 2), "head.scale": (DETECTOR_OUTPUTS, 4),
             "head.bias": (DETECTOR_OUTPUTS, 4)}
    for layer in DETECTOR_LAYERS:
        sizes.update({layer + ".weight": (c[layer] * 9 * inputs[layer], 2), layer + ".bias": (c[layer], 4),
                      layer + ".mul": (c[layer], 4)})
    check_sizes(arrays, sizes, "Word detector")
    net = Detector()
    for i, layer in enumerate(DETECTOR_LAYERS):
        net.channels[i] = channels[i]
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
