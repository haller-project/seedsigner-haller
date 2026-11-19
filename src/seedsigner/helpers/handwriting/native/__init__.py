"""
    Optional C implementation of the word network, of the dictionary decoding
    and of the flood fill used to find the card (handwriting.c), built by
    running `make` in this directory. Without the shared library, the numpy
    implementation gives the same network output and word scores within about
    2e-3, and the same flood fill.
"""
import ctypes
import os

LIBRARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhandwriting.so")

_library = []



class Network(ctypes.Structure):
    _fields_ = [
        ("c1", ctypes.c_int), ("c2", ctypes.c_int), ("c3", ctypes.c_int), ("hidden", ctypes.c_int),
        ("w", ctypes.c_void_p * 6), ("b", ctypes.c_void_p * 6), ("m", ctypes.c_void_p * 6),
        ("wo", ctypes.c_void_p), ("so", ctypes.c_void_p), ("bo", ctypes.c_void_p),
    ]



class Trie(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_int), ("words", ctypes.c_int),
        ("parent", ctypes.c_void_p), ("letter", ctypes.c_void_p), ("word_end", ctypes.c_void_p),
    ]



def load() -> ctypes.CDLL | None:
    """
        The shared library, or None if it has not been built or cannot be loaded.
    """
    if not _library:
        try:
            lib = ctypes.CDLL(LIBRARY_PATH)
        except OSError:
            lib = None
        if lib is not None:
            lib.hw_init.restype = None
            lib.hw_init.argtypes = []
            lib.hw_init()
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
            lib.hw_border_connected.restype = None
            lib.hw_border_connected.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                                ctypes.c_void_p]
        _library.append(lib)
    return _library[0]
