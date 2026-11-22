"""
Checks made before the C library of the handwriting reader is given the
weights: every weight file matches its SHA-256 in haller-weights.json when it
is loaded, every array has the size the C library reads, and the constants of
the word detector are the same in handwriting.h and in native/__init__.py.
"""
import array
import io
import json
import os
import re
import shutil
import zipfile

import pytest

from seedsigner.helpers.handwriting import native

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESOURCES = os.path.join(ROOT, "src", "seedsigner", "resources", "handwriting")
HEADER = os.path.join(ROOT, "src", "seedsigner", "helpers", "handwriting", "native", "handwriting.h")



def test_weights_checked_when_loaded(tmp_path):
    """ The weight files are read only if they match their SHA-256: a changed byte, a file not listed or a
    missing manifest is an error. """
    name = "haller-1-fingerprint.npz"
    for f in (name, native.WEIGHTS_MANIFEST):
        shutil.copy(os.path.join(RESOURCES, f), tmp_path / f)
    content = native.read_weights(str(tmp_path / name))
    assert content == (tmp_path / name).read_bytes()

    changed = bytearray(content)
    changed[len(changed) // 2] ^= 1
    (tmp_path / name).write_bytes(bytes(changed))
    with pytest.raises(native.WeightsError, match="SHA-256"):
        native.read_weights(str(tmp_path / name))

    (tmp_path / "other.npz").write_bytes(content)
    with pytest.raises(native.WeightsError, match="not listed"):
        native.read_weights(str(tmp_path / "other.npz"))

    os.remove(tmp_path / native.WEIGHTS_MANIFEST)
    with pytest.raises(native.WeightsError, match="manifest"):
        native.read_weights(str(tmp_path / name))



def test_every_shipped_weight_file_passes():
    with open(os.path.join(RESOURCES, native.WEIGHTS_MANIFEST)) as f:
        names = json.load(f)["weights"]
    for name in names:
        assert native.read_weights(os.path.join(RESOURCES, name))



def arrays_of(name):
    """ Arrays of a weight file as Python arrays (no numpy): name -> array.array. """
    codes = {"<i2": "h", "<i4": "i", "<f4": "f"}
    out = {}
    with zipfile.ZipFile(io.BytesIO(native.read_weights(os.path.join(RESOURCES, name)))) as z:
        for entry in z.namelist():
            data = z.read(entry)
            header_len = int.from_bytes(data[8:10], "little")
            header = data[10:10 + header_len].decode("latin1")
            dtype = re.search(r"'descr': '([^']+)'", header).group(1)
            out[entry[:-len(".npy")]] = array.array(codes[dtype], data[10 + header_len:])
    return out



def test_word_network_sizes():
    """ Each array of the word network has the size its layer reads (channels x taps x inputs): one cut short
    or of another size is an error, not a read beyond its end. """
    arrays = arrays_of("haller-1-capitals.npz")
    native.network(arrays)
    for name in ("b3.weight", "t1.weight", "out.weight", "b2.mul", "out.scale"):
        short = dict(arrays)
        short[name] = arrays[name][:-1]
        with pytest.raises(native.WeightsError, match=re.escape(name)):
            native.network(short)
    wrong = dict(arrays)
    wrong["b4.weight"] = array.array("i", arrays["b4.weight"])  # int32 instead of int16: twice the bytes
    with pytest.raises(native.WeightsError, match="b4.weight"):
        native.network(wrong)
    missing = dict(arrays)
    del missing["t2.bias"]
    with pytest.raises(native.WeightsError, match="t2.bias"):
        native.network(missing)
    odd = dict(arrays)
    odd["b1.bias"] = arrays["b1.bias"][:-1]
    with pytest.raises(native.WeightsError, match="channels"):
        native.network(odd)



def test_detector_sizes():
    arrays = arrays_of("haller-1-detector.npz")
    native.detector(arrays)
    for name in ("e1.weight", "d4.weight", "d1.weight", "head.weight", "e8.mul", "head.bias"):
        short = dict(arrays)
        short[name] = arrays[name][:-1]
        with pytest.raises(native.WeightsError, match=re.escape(name)):
            native.detector(short)
    fewer = dict(arrays)
    fewer["e6.bias"] = arrays["e6.bias"][:-4]
    with pytest.raises(native.WeightsError):
        native.detector(fewer)



def header_constants():
    with open(HEADER) as f:
        return {k: int(v) for k, v in re.findall(r"^#define (HW_\w+) (\d+)", f.read(), re.M)}



def test_constants_of_the_header():
    """ The constants of handwriting.h are those of native/__init__.py. """
    c = header_constants()
    assert (c["HW_DETECTOR_WIDTH"], c["HW_DETECTOR_HEIGHT"]) == native.DETECTOR_SIZE
    assert c["HW_DETECTOR_LAYERS"] == len(native.DETECTOR_LAYERS) == len(native.Detector().channels)
    assert c["HW_DETECTOR_OUTPUTS"] == native.DETECTOR_OUTPUTS
    assert c["HW_NUM_CLASSES"] == native.NUM_CLASSES
    assert c["HW_FIRST_DIGIT"] == native.FIRST_DIGIT_CLASS
    assert len(native.LAYERS) == len(native.Network().w)
    from seedsigner.helpers.handwriting.preprocess import INPUT_HEIGHT
    assert c["HW_INPUT_HEIGHT"] == INPUT_HEIGHT
