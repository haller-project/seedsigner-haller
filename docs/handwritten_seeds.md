# Handwritten seeds

## Status

**Proof of concept.** The reading engine is complete and measured (see
[Accuracy](#accuracy)); the user experience is not final. Known limits:

- numpy is not installed in SeedSigner's CI, so the engine tests
  (`tests/test_handwriting_core.py`) are skipped there; the reference vectors
  of the C library (`tests/test_handwriting_vectors.py`) need no numpy and run
  there (see [C library](#c-library)).
- A blank official card is read as 12 uncertain words instead of being
  rejected.
- The reading cannot be cancelled once started.


SeedSigner can load a seed from a photo of its handwritten words: the official
SeedSigner 12- or 24-word card, or a free list of 12, 15, 18, 21 or 24 words
(blank or ruled paper, one or two columns, with or without numbers). The
feature is disabled by default: once "Handwritten seeds" is enabled in the
advanced settings, "Scan written seed" is offered next to "Scan a SeedQR" in
the seed loading menus.

The words read must always be compared with the paper before the seed is
used.


## Flow

1. Camera preview, while the word models are loaded in the background; a click
   takes a full-resolution photo (2592x1944), kept as JPEG data.
2. The photo is read on the device (5 to 9 seconds on a Pi Zero for 12 words,
   see [Timings](#timings)). The progress bar moves at a steady pace from the
   click; the screen shows the photo while the words are searched, then the
   area of the words with the word being read outlined, and that word.
3. Status: a valid checksum is announced in a few words. Otherwise a full
   screen notice says what happened (words to check, or invalid checksum),
   that every word must be compared with the paper, what the marks on the
   uncertain words mean, and that unmarked words can be wrong too. The text
   is paged with the up and down keys; its "I understand" button can only be
   selected from the last page, so that a click made by mistake does not skip
   it.
4. Mandatory review: every word is listed, and "Done" is only offered once each
   uncertain word has been confirmed. Uncertain words are flagged; selecting a word
   shows its handwritten image, up to three candidate words with their
   probability, and the usual keyboard to type it. A confirmed word is fixed and
   the checksum analysis is run again.
5. "Done" passes the seed to the standard flow (fingerprint, passphrase).

Nothing is written to disk: the photo and the readings only live in memory and
are dropped with the views.


## How it works

The engine is `seedsigner.helpers.handwriting`:

| Module | Role |
|---|---|
| `page.py` | finds the card (corners, registration on its printed lines, rectification) or the lines and columns of a list (rotation, line profiles, word numbers), and cuts out one crop per word; a line box is widened while ink lies just beyond its sides (a letter set apart from the rest of the word, a faint stroke) |
| `preprocess.py` | turns a word crop into an ink map 32 pixels high |
| `word_model.py` | small convolutional network (about 160 000 parameters, CTC), run in 16-bit integers; two sets of weights, see [The Haller-1 model](#the-haller-1-model) |
| `decoder.py` | log-likelihood of each of the 2 048 BIP39 words (prefix tree, beam pruning); a line number before the word is accepted |
| `checksum.py` | combinations of plausible words, BIP39 checksum, probability of the seed and of each word given the checksum, uncertain words |
| `native/handwriting.c` | network, decoder and the flood fill of the card search in portable C (no allocation, no CPython), used through `ctypes`; required, see [C library](#c-library) |

There is no letter segmentation: the network reads the whole word, and the
dictionary decoding only ever returns BIP39 words. The checksum then rules out
most wrong combinations; a seed is reported as valid only when every word has a
probability of at least 0.95 given the photo and the checksum.

### Calibration

The word scores of the decoder are divided by a temperature (2.2, for each
network and for the blend) to serve as log-likelihoods. The analysis then
computes the probability of each valid seed given the photo and the checksum:
its reading probability divided by that of all the valid seeds, the ones
enumerated and the share of the combinations left out that passes the
checksum by chance (1 in 16 for 12 words). The probability of a word is
the sum over the valid seeds that contain it. This is the percentage shown
next to each proposed word when the user reviews an uncertain word.

The temperature and the thresholds were chosen on 640 simulated pages kept
apart from the test photos (EMNIST test capitals and two lowercase fonts
never used elsewhere): the lowest temperature with no wrong seed reported as
valid (at 1.3 and 1.8, two wrong seeds pass), then the threshold on the words.
On the 1,605 test photos, a word shown at 15 % is right 8 % of the time, at
45 % 52 %, at 75 % 82 %, at 86 % 89 % (expected calibration error 0.005; 0.020
on cursive); before this calibration, words shown at 15 to 35 % were right
16 to 23 % of the time, and seeds given 15 to 35 % were right 70 to 75 % of the
time (calibration error of the seed probability 0.070, now 0.016). One
temperature fits all writings only approximately: capitals alone would take
2.6, cards and cursive about 1.7.

The two networks of Haller-1, of the same size, read the words. The first
one reads every photo. When its seed is not valid, the photo is read again by
the second one, trained with more lowercase and cursive writing, then by a
blend of both: for each word, the weighted sum of the log-probabilities
given by the two networks (0.65 for the first one), which reuses the scores
already computed.
The most likely of these readings is kept (better status, or same status with
a more probable seed). A seed of the second network alone needs a probability
of 0.85 instead of 0.5 to be reported as valid without uncertain words: it
reads the photos the first one failed on, where a wrong combination passes
the checksum more often. The blend, which corrects most of these chance
agreements, keeps the threshold of 0.5. Since every word must reach 0.95, these
thresholds on the seed seldom decide.

When the seed is invalid, or valid but improbable (probability below 0.1), a
misread word can be read with high probability of its own while the checksum
is passed by chance by the other words: every word is then marked uncertain.
So it is when text is left out of the crops of a list (a missed line, two
columns read as one): the number of words is then uncertain.

A list in two columns of equal length is read column by column, or row by
row when only that order gives a valid seed. A permutation of a seed passes
the checksum once in 16 for 12 words (once in 256 for 24), with every word
read with high probability: when both orders give a plausible seed (valid,
probability at least 0.1), the order written is unknown. Every word is then
marked uncertain, and the status notice asks to check the order of every word
against the paper.

On a blurred photo of a card, the printed lines are spread out and lighter;
when no card candidate shows half of its lines, they are looked for again
with a lower contrast (3 gray levels instead of 6), which never turns a list
of the test photos into a card.

The card is first found by the light area inside its border, then
registered on its printed lines, so that a wrong edge does not shift the
words: the lines are looked for on a rectification wider than the card,
numbered by their regular spacing, and the first one is told by the lines
that should show where the photo covers the card, by the place the card
edges give when none of them is the frame of the photo, and by the ends of
the lines (lines 10 to 12 start further right). The card is thus read in
place when its header or an edge is out of the photo, when it is seen at an
angle, turned up to 30 degrees or upside down, small in the photo, partly in
a shadow or under a finger, or among other cards of a printed sheet; the
right column of a 24-word card is not taken for a 12-word card. When a word
line of the card is out of the photo, no word is read: the user is asked to
move the camera back so that the whole card fits. Lists are straightened up
to 30 degrees.

`read_seed(image, progress, on_word)` reports its progress after every step
of the page search and of the reading. Each step is weighted by its expected
duration on a Pi Zero (constants `*_COST` in `page.py` and `__init__.py`) and
moves the progress by its share of the work expected to remain, an estimate
revised once the card or list is found and when a second reading is needed:
the bar moves steadily when the estimates are right, and never goes back nor
stops long when they are not. `on_word(layout, k, word_reading)` is called
after the first reading of each word; `layout.outline(k)` gives the corners of
the word on the photo.

The photo is taken through `Camera`, with two optional arguments added for
this feature; their defaults keep the behavior of the other callers unchanged:

- `start_single_frame_mode(framerate=...)`: the 2592x1944 sensor mode runs at
  up to 15 frames per second, and the framerate caps the exposure time. At the
  default 24 the exposure is limited to 1/24 s, which darkens photos in dim
  light (measured on a Pi Zero: 41 ms instead of 63 ms in the same scene).
- `capture_frame(expand=True)`: rotating a landscape frame to portrait without
  expansion keeps a 2592x1944 canvas and drops a quarter of the photo (the
  top and bottom of the page).

and two methods that split `capture_frame()`, which keeps its behavior:
`capture_jpeg()` takes the photo and returns the camera's JPEG data, and
`decode_frame(jpeg, expand, grayscale)` decodes and rotates it. The scan screen
only takes the JPEG (0.6 s on a Pi Zero); the photo is decoded in the reading
thread, under the progress bar, and luminance only: 0.5 s instead of 1.9 s
for a color decoding, rotation and conversion to gray, and 5 MB of memory
instead of 15 MB.

The reading screen is redrawn at most every 0.7 second, with images prepared
by the reading thread: a redraw costs about 11 ms of processor time on a Pi
Zero (the transfer to the display takes 37 ms, during which the reading goes
on).


## C library

The network, the dictionary decoding and the flood fill of the card search
exist only in C, in `native/handwriting.c`: there is no Python fallback,
which would be 4 to 7 times slower on a Pi Zero (35 s for a card, 45 s for a
list) and would hide a library left unbuilt. `native/__init__.py` loads
`libhandwriting.so` from the same directory with `ctypes`; if it is missing,
loading the word models raises `LibraryNotBuilt` with the command to build
it, and the error is logged.

Dependencies: no new Python dependency. Pillow, numpy and embit already ship
with SeedSigner OS (numpy is listed in `requirements-raspi.txt` and imported
at startup by `BackgroundImportThread`); the small C library is
cross-compiled with the OS.

The C file has no dependency besides `libm`, allocates nothing and does not use
the CPython API. On ARMv6 it uses the `SMLAD` instruction when the compiler
defines `__ARM_FEATURE_DSP` (GCC does for `-mcpu=arm1176jzf-s`), and portable C
otherwise or with `-DHW_GENERIC`; both give the same bits.

```bash
# development machine
make -C src/seedsigner/helpers/handwriting/native
# on a Pi Zero running Raspberry Pi OS
make -C src/seedsigner/helpers/handwriting/native CFLAGS_ARCH="-mcpu=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard"
```

The built library is ignored by git. SeedSigner OS builds it for each board
when the image is built (Buildroot package `seedsigner-handwriting`, branch
`handwritten-seed` of the SeedSigner OS fork): the Buildroot toolchain wrapper
adds the board's CPU flags (pi0: `-mcpu=arm1176jzf-s -mfpu=vfp
-mfloat-abi=hard`), the other flags are those of the Makefile. The device has
no compiler.

### Reference vectors

Each set of weights comes with reference vectors written with it by
`haller.vectors` (Haller repository, also run by `haller.quantize`):
`resources/handwriting/haller-1-capitals-vectors.json` and
`haller-1-lowercase-vectors.json`. Each file holds 62 word strips, all
synthetic (20 in capitals from simulated photos, 20 mixed-case words of the
training generator, 20 cursive words, an empty strip and a strip of noise of
the maximum width), and for each one the output of the C library: SHA-256 of
the network logits, SHA-256 of the scores of the 2,048 BIP39 words, and the 5
best words with their scores. `tests/test_handwriting_vectors.py` checks that
the library gives the same bits, with ctypes alone (no numpy), so it runs in
SeedSigner's CI, which builds the library first. It also passes on ARMv6
(arm1176 emulated by QEMU, Raspberry Pi OS userland) with the library
cross-compiled by SeedSigner OS's toolchain, with SMLAD and with
`-DHW_GENERIC`.

Bit-identical results across platforms rely on integer arithmetic in the
network, on IEEE single precision without contraction (`-ffp-contract=off`)
elsewhere, and on a constant table for the log-add of the decoder:
`log1pf`/`expf` of glibc differ in the last bit between x86-64 and armhf on
12 of its 1,025 values. Only the log-softmax calls `libm` (`expf`, `logf`).

Haller also checks, when it writes the vectors, that the quantized C model
reads them as the float model it comes from (PyTorch): see the Haller
README for the tolerance.

The vectors must be written again for every new set of weights (Haller-2…),
and for any change of `handwriting.c` that alters its output (none is
expected: a change that keeps the results must keep the vectors):

```bash
# in the Haller repository, SeedSigner's sources on PYTHONPATH
python -m haller.vectors src/seedsigner/resources/handwriting/haller-N-capitals.npz \
    --strips capitals=EVAL.npz --strips mixed=SHARD.npz --strips cursive=CURSIVE_VAL.npz \
    --float MODEL_FLOAT.npz
```

### Tests in the CI (proposal)

The engine tests that need numpy (`tests/test_handwriting_core.py`: page
detection, normalization, reading of rendered words, progress) are skipped in
the CI, where `tests/base.py` replaces numpy with a mock. Adding numpy to
`tests/requirements.txt`, hash-locked like the other test dependencies, would
run them there too: numpy is already a runtime dependency on the device, so
the CI would test what ships. Not done here, to leave that choice to the
maintainers: it adds a large wheel to the CI install, and the C library, the
part most likely to differ between platforms, is already covered by the
vectors.


## Accuracy

Measured on 835 photos (800 simulated: cards and free lists, varied hands,
paper, lighting and framing; 35 real), the engine reads 86 % of the seeds
exactly and reports 83 % as valid and exact, with no wrong seed reported as
valid. On 460 other simulated photos, generated independently: 88 % exact,
85 % valid and exact, no wrong seed reported as valid. Of the real photos,
17 of 18 cards and the 6 sharp lists are read exactly and reported as valid;
the 11 blurry ones are never reported as valid.

These photos are written in capitals. Lowercase and cursive writing is read
less well. By type of writing (seeds valid and exact / exact / wrong but
reported as valid; words read correctly):

| Writing | Photos | Valid and exact | Exact | Wrong valid | Words |
|---|---|---|---|---|---|
| Capitals (photos above) | 1295 | 1083 | 1121 | 0 | 87 % |
| Neat cursive (simulated) | 301 | 95 | 147 | 0 | 70 % |
| Real photos, lowercase and cursive | 4 | 0 | 2 | 0 | 41 of 48 |

The simulated cursive pages are written in 13 handwriting fonts (10
Playwrite school scripts, League Script, Meddon, Fleur De Leah) never used
in training, laid out and photographed like the other simulated pages. Most
cursive seeds are read exactly but reported with words to check: joined
letters (`nn`/`r`, `ak`/`ar`) and cursive `j` read as `s` are the usual
confusions. A fast, sloppy cursive is read worst (8 to 9 of 12 words on real
photos). In every case the uncertain words are marked, and a seed read with
low confidence has all its words marked.

Most failures are lists cut by the frame of the photo. When the ink of a list
reaches the edge of the photo, the user is asked to move the camera back
(36 of 39 such photos in the measurement above); otherwise to retake the photo.
A card with a word line out of the photo is refused the same way ("Card cut
off").

Framing was measured on 735 other simulated photos (capitals and neat
cursive; cards and lists cut by the frame, close, far, at an angle, turned,
upside down, among other cards, under objects and shadows, under uneven
light), 35 per kind of defect and support. Cards, seeds valid and exact
before and after the registration on the printed lines, and seeds read with
words shifted to another place: cut at the top 8 to 15 (shifted 4 to 3),
cut on a side 11 to 14 (2 to 0), close 18 to 22, turned up to 30 degrees 13
to 29, upside down 0 to 21, at an angle 15 to 16, under objects 16 to 17
(shifted 7 to 4), among other cards 21 to 23; far from the camera, 0 to 16
read exactly, never reported as valid. Lists turned up to 30 degrees: 11 to
27. No wrong seed is reported as valid on cards (one before); the one wrong
valid list (cut at the bottom) is unchanged. A whole printed sheet of cards
in the photo and an upside down list are not read. A card that is found
whole takes the same path as before; the registration only runs on a card
found on part of its lines or on the frame of the photo (5 to 6 s more for
the search on a Pi Zero, where a shifted reading used to take 21 s).

## Timings

Measured on a Raspberry Pi Zero with the C library, the application stopped,
on photos taken by the device (12 words), from the click to the result screen:

| Step | card | list, capitals | list, lowercase or cursive |
|---|---|---|---|
| photo taken (JPEG) | 0.6 s | 0.6 s | 0.6 s |
| photo decoded (gray, upright) | 0.6 s | 0.6 s | 0.6 s |
| card or list found | 0.6 s | 4.3 s | 4.3 s |
| card rectified | 1.6 s | | |
| words read, checksum | 2.0 s | 2.2 s | 2.5 to 4.8 s |
| display of the reading | 0.5 s | 0.6 s | 0.6 to 0.8 s |
| result screen | 0.2 s | 0.2 s | 0.2 s |
| **total** | **6.1 s** | **8.1 to 8.4 s** | **9.2 to 11.4 s** |

The calibrated analysis and the widening of the list crops add, on the same
photos: nothing on the card, 0.4 s on the lists in capitals, 0.5 to 1.9 s on
the lowercase and cursive lists (more readings of each word to combine).

The word models (1 s; numpy is already imported at startup by
SeedSigner) are loaded while the photo is framed; the camera preview runs at
3.3 instead of 5 frames per second during that second. The progress bar never
stays still for more than 0.9 s. The display of the reading costs 3 to 5 % of
processor time.


## The Haller-1 model

The words are read by Haller-1, the first version of the Haller word
recognizer. It is made of two networks of the same architecture, combined as
described in [How it works](#how-it-works):

| Weights (`src/seedsigner/resources/handwriting/`) | Role | SHA-256 |
|---|---|---|
| `haller-1-capitals.npz` | reads every photo; trained mostly on capitals | `2e9275f1acd0d347437e856160430e96a283ec484a48ab96a2ea1ca91531fe14` |
| `haller-1-lowercase.npz` | second reading; more lowercase and cursive writing | `5d340b354ced032992cb474f03bcc0b063f24ee8e1446d2953203b778652e509` |

They are produced, from scratch and reproducibly, by the separate
[Haller](https://github.com/pizza-bitcoiner/haller) repository (tag
`haller-1`): synthetic words, real handwritten words, training, then
quantization. Photos taken by users are never used.

Haller is named after Neil M. Haller of Bellcore, author of RFC 1760, "The
S/KEY One-Time Password System" (1995). S/KEY, designed with Phil Karn and
John Walden on an idea by Leslie Lamport, already wrote a secret as words
from a 2,048-word dictionary, 11 bits per word, with spare bits as a
checksum: the same layout BIP39 would use eighteen years later.

Training data of these weights:

- Synthetic words written with handwriting fonts from Google Fonts (SIL Open
  Font License 1.1 or Apache License 2.0) and with EMNIST letters (Cohen,
  Afshar, Tapson and van Schaik, "EMNIST: an extension of MNIST to
  handwritten letters", 2017, derived from NIST Special Database 19).
- Word images from GNHK, the GoodNotes Handwriting Kollection (Lee, Chung and
  Lee, "GNHK: A Dataset for English Handwriting in the Wild", ICDAR 2021,
  https://goodnotes.com/gnhk/), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The word images
  were cropped, straightened and normalized for training; no GNHK image is
  included here.
- The BIP39 English word list, and the BIP39 lists of the other Latin-script
  languages (accents removed) as non-target words.
- Second network only: more synthetic words in lowercase, written as whole
  words in cursive fonts, and BIP39 words written in the hands of GNHK writers
  by Emuru (Pippi et al., "Zero-Shot Styled Text Image Generation, but Make It
  Autoregressive", CVPR 2025; model under the MIT license), checked by a
  vision model.


## Limits and threat model

- A reading can be wrong even with a valid checksum: the review against the
  paper is not optional.
- The photo shows the whole seed to the camera. SeedSigner keeps it in memory
  only, like the frames of a SeedQR scan, and has no network.
- Blurry photos, very small writing and unusual layouts fail; the user is asked
  to retake the photo.
