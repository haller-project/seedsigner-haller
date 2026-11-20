"""
The weights of the handwriting reader are those of the Haller version named in
resources/handwriting/haller-weights.json: every weight file is listed there
with its SHA-256, matches it, has its reference vectors, and is the one the
engine loads; the documentation quotes the same digests. Uses neither numpy
nor the C library, so that it runs in the CI.
"""
import glob
import hashlib
import json
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESOURCES = os.path.join(ROOT, "src", "seedsigner", "resources", "handwriting")
ENGINE = os.path.join(ROOT, "src", "seedsigner", "helpers", "handwriting")
MANIFEST = os.path.join(RESOURCES, "haller-weights.json")
DOC = os.path.join(ROOT, "docs", "handwritten_seeds.md")
VERSION = re.compile(r"^haller-(\d+)(\.\d+)?$")



@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST) as f:
        return json.load(f)



def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()



def test_manifest_fields(manifest):
    assert VERSION.match(manifest["version"]), manifest["version"]
    assert manifest["release"].endswith("/releases/tag/" + manifest["version"])
    assert isinstance(manifest["published"], bool)
    assert manifest["weights"]
    for name, digest in manifest["weights"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), name



def test_digests(manifest):
    for name, digest in manifest["weights"].items():
        path = os.path.join(RESOURCES, name)
        assert os.path.isfile(path), name
        assert sha256(path) == digest, name



def test_every_weight_file_is_listed(manifest):
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RESOURCES, "*.npz")))
    assert files == sorted(manifest["weights"])



def test_names_follow_the_version(manifest):
    major = "haller-" + VERSION.match(manifest["version"]).group(1) + "-"
    for name in manifest["weights"]:
        assert name.startswith(major), name



def test_reference_vectors(manifest):
    for name in manifest["weights"]:
        assert os.path.isfile(os.path.join(RESOURCES, name[:-len(".npz")] + "-vectors.json")), name



def test_engine_loads_listed_weights(manifest):
    loaded = set()
    for path in glob.glob(os.path.join(ENGINE, "*.py")):
        with open(path) as f:
            loaded.update(re.findall(r"haller-[\w.-]+?\.npz", f.read()))
    assert loaded
    assert loaded <= set(manifest["weights"]), loaded - set(manifest["weights"])



def test_documentation_quotes_the_digests(manifest):
    with open(DOC) as f:
        doc = f.read()
    for name, digest in manifest["weights"].items():
        assert digest in doc, name
    assert manifest["version"] in doc
