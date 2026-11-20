"""
The C library of the handwriting reader reproduces, bit for bit, the
reference vectors written with the Haller-1 weights by haller.vectors
(resources/handwriting/haller-1-*-vectors.json): network logits, scores of the
2048 BIP39 words, best words; and those of the word detector written by
haller.detector_export (haller-1-detector-vectors.json): output maps. Uses neither numpy nor the rest of the engine,
so that it runs in the CI, where the library is built first
(make -C src/seedsigner/helpers/handwriting/native).
"""
import ast
import base64
import ctypes
import glob
import hashlib
import json
import os
import struct
import zipfile
import zlib

import pytest
from embit.bip39 import WORDLIST

from seedsigner.helpers.handwriting import native

RESOURCES = os.path.join(os.path.dirname(__file__), "..", "src", "seedsigner", "resources", "handwriting")
DETECTOR_VECTORS = os.path.join(RESOURCES, "haller-1-detector-vectors.json")
VECTORS = sorted(p for p in glob.glob(os.path.join(RESOURCES, "*-vectors.json")) if p != DETECTOR_VECTORS)
INPUT_HEIGHT = 32



@pytest.fixture(scope="module")
def lib():
    try:
        return native.load()
    except native.LibraryNotBuilt:
        if os.environ.get("CI"):
            raise
        pytest.skip("C library not built (make -C src/seedsigner/helpers/handwriting/native)")



def load_npz(path: str) -> dict:
    """ Arrays of an uncompressed .npz file (format 1.0, little-endian, C order) as bytearrays. """
    arrays = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            data = z.read(name)
            assert data[:8] == b"\x93NUMPY\x01\x00"
            size = struct.unpack("<H", data[8:10])[0]
            header = ast.literal_eval(data[10:10 + size].decode("latin1"))
            assert header["descr"][0] in "<|" and not header["fortran_order"]
            arrays[name[:-len(".npy")]] = bytearray(data[10 + size:])
    return arrays



def test_vectors_present():
    assert [os.path.basename(p) for p in VECTORS] == ["haller-1-capitals-vectors.json",
                                                      "haller-1-lowercase-vectors.json"]



@pytest.mark.parametrize("path", VECTORS, ids=os.path.basename)
def test_vectors(lib, path):
    with open(path) as f:
        vectors = json.load(f)
    with open(os.path.join(RESOURCES, vectors["weights"]), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == vectors["weights_sha256"]
    network = native.network(load_npz(os.path.join(RESOURCES, vectors["weights"])))
    trie = native.trie(WORDLIST)
    decoder_work = (ctypes.c_float * (lib.hw_decoder_work_size(ctypes.byref(trie)) // 4))()
    scores = (ctypes.c_float * len(WORDLIST))()
    for strip in vectors["strips"]:
        width = strip["width"]
        ink = zlib.decompress(base64.b64decode(strip["ink"]))
        assert len(ink) == INPUT_HEIGHT * width
        frames = max(1, width // 4)
        work = (ctypes.c_int32 * -(-lib.hw_network_work_size(ctypes.byref(network), width) // 4))()
        z = (ctypes.c_float * (native.NUM_CLASSES * frames))()
        lib.hw_network_logits(ctypes.byref(network), ink, width, work, z)
        assert hashlib.sha256(bytes(z)).hexdigest() == strip["logits_sha256"], strip["label"]
        lib.hw_log_softmax(z, frames)
        lib.hw_word_scores(ctypes.byref(trie), z, frames, vectors["beam"], decoder_work, scores)
        best = sorted(range(len(WORDLIST)), key=lambda k: -scores[k])[:vectors["top"]]
        assert [WORDLIST[k] for k in best] == strip["words"], strip["label"]
        assert [scores[k] for k in best] == strip["scores"], strip["label"]
        assert hashlib.sha256(bytes(scores)).hexdigest() == strip["scores_sha256"], strip["label"]



def test_detector_vectors(lib):
    with open(DETECTOR_VECTORS) as f:
        vectors = json.load(f)
    with open(os.path.join(RESOURCES, vectors["weights"]), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == vectors["weights_sha256"]
    detector = native.detector(load_npz(os.path.join(RESOURCES, vectors["weights"])))
    width, height = native.DETECTOR_SIZE
    work = (ctypes.c_int32 * -(-lib.hw_detector_work_size(ctypes.byref(detector)) // 4))()
    maps = (ctypes.c_float * (native.DETECTOR_OUTPUTS * (height // 2) * (width // 2)))()
    assert len(vectors["images"]) == 2
    for image in vectors["images"]:
        pixels = zlib.decompress(base64.b64decode(image["image"]))
        assert len(pixels) == width * height
        lib.hw_detector_maps(ctypes.byref(detector), pixels, work, maps)
        assert hashlib.sha256(bytes(maps)).hexdigest() == image["maps_sha256"], image["label"]
