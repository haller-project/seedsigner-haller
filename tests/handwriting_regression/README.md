# Regression cases of the handwriting reader

One page per fix of the harness, with the reading expected, replayed by
`tests/test_handwriting_regression.py` like any unit test.

Each fix of the reader starts from a real photo that it misread. That photo
stays private; the case here is its **twin**: a synthetic page that goes
through the same mechanism. A twin is kept only if the engine from before
the fix fails on it and the engine with the fix reads it as expected. A few
more pages are **safety traps**: the reading may fail, but must never be a
wrong seed given as sure or verified.

## Rules

- A change of the harness (code, thresholds) breaks no active case.
- New weights: every case that flips is looked at; a flip from pass to fail is
  accepted only when it is explained and made up for elsewhere. A safety case
  never fails.
- Every new fix adds its case (its twin), in the same change.
- These cases guard against regressions; they do not replace the wide
  benches, which decide whether a change is better.

## `cases.json`

| Key | Meaning |
|---|---|
| `id` | name of the case (`s-` synthetic) |
| `fix` | the fix the case stands for, with its commit |
| `image` | page, under `images/` |
| `mechanism` | what goes wrong without the fix |
| `before` | engine commit before the fix (it fails on the page) |
| `twin_of` | private real photo the page reproduces |
| `level` | `regression` or `security` (only safety is expected) |
| `state` | `active`, or `pending` (fix on a branch not merged yet, or a known failure with no fix): expected failure |
| `truth` | seed written, when `expected` does not ask for the exact seed |
| `expected` | `words` (exact seed), `status`, `verified`, `fingerprint`, `word_count`, `min_words_right`, `rejected` (neither verified nor valid), `never_false_sure` (valid only when exact), `max_seeds_compared` (seeds compared with a fingerprint line while the photo is read, both searches included), `max_trust` (highest trust level allowed when not verified: 1 low, 2 medium, 3 high) |
| `generation` | how the page was made: generator, its options, seed and index of the page |

Every case also checks that a verified reading is exact.

## Pages

Synthetic only: test seeds drawn at random, EMNIST test capitals or
handwriting fonts under the SIL Open Font License, simulated paper and
camera. Pages are stored in grayscale, at most 1728 px on the long side
(JPEG 75), except one page whose mechanism needs the full resolution.
