"""
    Regression cases of the handwriting reader: one synthetic page per fix
    of the harness (the twin of the photo that led to it) and per safety
    trap, each with the reading expected (tests/handwriting_regression,
    README.md there). Every active case must pass for a change of the
    harness; a pending case (fix on a branch not merged yet, or a known
    failure) is reported as an expected failure until its fix lands.

    About 1 to 3 s per page with the C library built
    (make -C src/seedsigner/helpers/handwriting/native).
"""
import json
import os

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

HERE = os.path.join(os.path.dirname(__file__), "handwriting_regression")
CASES = json.load(open(os.path.join(HERE, "cases.json")))["cases"]


def read(path, rotation=0, compared=None):
    """ (status, words, verified, trust level) of the page; status "not_found" when no list or card is found.
    `compared`, a list, receives one item per seed compared with a fingerprint line during the reading. """
    import seedsigner.helpers.handwriting as hw
    from seedsigner.helpers.handwriting import fingerprint
    from seedsigner.helpers.handwriting.page import WordsNotFound
    photo = Image.open(path).convert("L")
    if rotation:
        photo = photo.rotate(rotation, expand=True)
    seed_fingerprint = fingerprint.seed_fingerprint
    if compared is not None:
        fingerprint.seed_fingerprint = lambda words, *a, **k: compared.append(words) or seed_fingerprint(words, *a, **k)
    try:
        reading = hw.read_seed(photo)
    except WordsNotFound:
        return "not_found", [], False, None
    finally:
        fingerprint.seed_fingerprint = seed_fingerprint
    analysis = reading.analyze()
    return analysis.status, list(analysis.mnemonic), bool(getattr(reading, "verified", False)), reading.trust().level


def failures(case, status, words, verified, compared=0, trust=None):
    """ Reasons why the reading does not meet what the case expects (empty: it does). """
    expected = case["expected"]
    truth = expected.get("words") or case.get("truth")
    exact = truth is not None and words == truth
    out = []
    if truth is not None and verified and not exact:
        out.append("SAFETY: verified but wrong")
    if expected.get("never_false_sure") and status == "valid" and not exact:
        out.append("SAFETY: given as sure (valid) but wrong")
    if expected.get("rejected") and (verified or status == "valid"):
        out.append(f"to be rejected, read {status}{' verified' if verified else ''}")
    if expected.get("words") and not expected.get("rejected") and not exact:
        right = sum(a == b for a, b in zip(words, truth)) if len(words) == len(truth) else 0
        out.append(f"seed not exact ({right}/{len(truth)} words right, {len(words)} read)")
    if "status" in expected and status != expected["status"]:
        out.append(f"status {status} instead of {expected['status']}")
    if "verified" in expected and verified != expected["verified"]:
        out.append("not verified" if expected["verified"] else "verified, must not be")
    if "word_count" in expected and len(words) != expected["word_count"]:
        out.append(f"{len(words)} words read instead of {expected['word_count']}")
    if "max_seeds_compared" in expected and compared > expected["max_seeds_compared"]:
        out.append(f"SAFETY: {compared} seeds compared with the fingerprint, at most {expected['max_seeds_compared']}")
    if "max_trust" in expected and not verified and trust is not None and trust > expected["max_trust"]:
        out.append(f"SAFETY: trust level {trust}, at most {expected['max_trust']}")
    if "min_words_right" in expected:
        right = sum(a == b for a, b in zip(words, truth)) if len(words) == len(truth) else 0
        if right < expected["min_words_right"]:
            out.append(f"{right} words right, at least {expected['min_words_right']} expected")
    return out


def test_cases_are_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))
    for c in CASES:
        assert os.path.exists(os.path.join(HERE, c["image"])), c["image"]
        assert c["level"] in ("regression", "security") and c["state"] in ("active", "pending")
        assert c["expected"], c["id"]
        truth = c["expected"].get("words") or c.get("truth")
        if truth:
            from embit import bip39
            assert bip39.mnemonic_is_valid(" ".join(truth)), c["id"]


@pytest.mark.parametrize("case", [
    pytest.param(c, id=c["id"], marks=[pytest.mark.xfail(reason=f"pending ({c.get('branch', '')})", strict=False)]
                 if c["state"] == "pending" else [])
    for c in CASES])
def test_regression_case(case):
    compared = []
    status, words, verified, trust = read(os.path.join(HERE, case["image"]), case.get("rotation", 0), compared)
    assert not failures(case, status, words, verified, len(compared), trust)
