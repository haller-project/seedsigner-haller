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


## Trying it on a development machine

There is no desktop emulator in this repository. On a development machine
(Python 3.10 or later, a C compiler and `make`):

```bash
pip install -r requirements.txt && pip install -e . && pip install numpy
make -C src/seedsigner/helpers/handwriting/native
```

then read a photo (upright, full resolution) with the engine:

```python
from PIL import Image
from seedsigner.helpers.handwriting import read_seed

analysis = read_seed(Image.open("photo.jpg")).analyze()
print(analysis.status, " ".join(analysis.mnemonic), analysis.uncertain)
```

`status` is `valid`, `uncertain` (valid checksum, words to check, listed in
`uncertain`) or `invalid`; `read_seed` raises `WordsNotFound` when no card or
list of 12 to 24 words is found. Use test photos only (for example the BIP39
test vectors written by hand), never a real seed.

The screens of the feature are rendered by the screenshot generator, as in the
CI (`python setup.py compile_catalog` first, see `tests/README.md`):
`python -m pytest tests/screenshot_generator/generator.py` writes them to
`seedsigner-screenshots/<language>/handwritten_seed_views/`.


## How it works

The engine is `seedsigner.helpers.handwriting`:

| Module | Role |
|---|---|
| `page.py` | finds the card (corners, registration on its printed lines, rectification) or the lines and columns of a list (rotation, line profiles, word numbers), and cuts out one crop per word; a line box is widened while ink lies just beyond its sides (a letter set apart from the rest of the word, a faint stroke) |
| `preprocess.py` | turns a word crop into an ink map 32 pixels high |
| `word_model.py` | small convolutional network (about 160 000 parameters, CTC), run in 16-bit integers; two sets of weights, see [The Haller-1 model](#the-haller-1-model) |
| `decoder.py` | log-likelihood of each of the 2 048 BIP39 words (prefix tree, beam pruning); a line number before the word is accepted; probability of each line number |
| `detector.py` | word detector of Haller-1: finds the words, the line numbers and the card when the rules of `page.py` give no valid seed, see [Word detector](#word-detector) |
| `checksum.py` | combinations of plausible words, BIP39 checksum, probability of the seed and of each word given the checksum, uncertain words |
| `native/handwriting.c` | networks, decoder, flood fill and connected components in portable C (no allocation, no CPython), used through `ctypes`; required, see [C library](#c-library) |

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

Line numbers, when they are written, tell the order. They are read only when
the order matters (more than one order passes the checksum), by the first
network, on the number cut off the line or, when it is written against the
word ("7)WORD"), on the crop of the word itself, whose network outputs are
already computed. The probability of each number 1 to the number of words is
that of the digits read being exactly its digits, letters, punctuation and
blanks aside (`decoder.number_probabilities`): a "10" read as "1o" gives no
probable number. A number read with a probability of at least 0.9 votes for
the orders in which its word has that position (the first and last words,
at the same position in both orders, do not vote). An order with at least
three votes, the other having none, is the order written: its reading is
kept, even if it does not pass the checksum, and the words are not marked for
the order. Otherwise nothing changes. On 317 simulated two-column lists kept
apart for this setting (row by row or column by column, numbers "1.", "1)"
or "1", written apart or against the word), the order was told on 218, and
never wrongly. On the 40 test lists written row by row with numbers, which
were all uncertain, 29 are now valid and exact; the other photos measured do
not change (1 605 photos: 1 195 valid and exact instead of 1 189, no wrong
seed reported as valid).

### Word detector

When the rules of `page.py` find no card or list, or a seed that is not
valid, the photo is searched again by the word detector of Haller-1, and its
words are read the same way; its reading is kept if its status is better, or
as good with a more probable seed. The detector is a small fully
convolutional network (U-Net, 223 000 parameters, same 16-bit integer
arithmetic as the word network) on the photo reduced to 288 x 384 pixels;
it outputs maps of the word and line number kernels, of the card area and of
its four corners, and of the height of the boxes. A card is placed from its
corners (or those of the rules when both designate the same card, these
being more precise) and cut along its printed lines as before; a list is cut
around the detected word boxes, grouped into lines and columns.

A second search is a second chance to fall on a wrong seed that passes the
checksum, and a second reading of words the rules already read. Its seed is
only declared valid if:

- no reading of the rules (both networks, the blend, both orders) gives
  another plausible seed;
- the rules did not read an uncertain seed on a complete layout: their
  uncertainty is then about the words themselves, which reading them again
  does not remove (a layout the rules found incomplete, with text left out or
  a card too small in the photo, or read as an invalid seed, may be read
  anew);
- no word read by the rules with a probability of at least 0.9 differs;
- the detector reading itself is complete: no crop goes out of the photo, the
  card covers at least 8 % of it, the list does not reach its edge, and at
  most one detected word lies outside the crops (another card of a sheet, the
  cut column of a 24-word card, a word written above its line).

Otherwise every word is marked uncertain. When the rules read an uncertain
seed on a complete layout, the detector is not run at all.

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

The networks, the dictionary decoding, the flood fill of the card search and
the connected components of the detector maps exist only in C, in `native/handwriting.c`: there is no Python fallback,
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
`handwritten-seed` of the SeedSigner OS fork, not published yet): the
Buildroot toolchain wrapper adds the board's CPU flags (pi0:
`-mcpu=arm1176jzf-s -mfpu=vfp -mfloat-abi=hard`), the other flags are those
of the Makefile. The device has no compiler. Until that fork is published, the
feature runs on a Pi installed by hand
([Raspberry Pi OS build instructions](raspberry_pi_os_build_instructions.md)),
with the library built on the Pi by the second command above, and numpy
installed (`requirements-raspi.txt`).

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

The word detector has its own vectors, `haller-1-detector-vectors.json`,
written with its weights by `haller.detector_export` (Haller repository,
branch `detector`): two synthetic images (a page of strokes, a card with
printed lines) and the SHA-256 of the output maps of the C library, which
`haller.detector_export` also compares with the integer reference in numpy
(`haller.detector.integer_forward`). They pass on the Pi Zero (SMLAD) as on
x86-64.

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

The word detector was measured on the same photos (seeds valid and exact /
exact / wrong but reported as valid; words read correctly), before and after:

| Photos | Before | With the detector |
|---|---|---|
| 1 605 photos of the measurement above | 1 195 / 1 291 / 0 / 83.9 % | 1 231 / 1 370 / 0 / 90.9 % |
| of which capitals (1 301) | 1 092 / 1 135 / 0 / 87.0 % | 1 116 / 1 191 / 0 / 93.3 % |
| of which neat cursive (301) | 103 / 155 / 0 / 70.6 % | 115 / 178 / 0 / 80.8 % |
| 1 167 badly framed photos (735 above and 432 others) | 420 / 520 / 2 / 50.5 % | 460 / 615 / 2 / 62.7 % |
| 20 real photos taken by the device | 11 / 17 / 0 / 95.4 % | 11 / 17 / 0 / 95.8 % |

The two wrong valid seeds of the badly framed photos are the same before and
after (a list cut at the bottom, a card in strong perspective). On 920
photos built to provoke wrong valid seeds (see the threat model), the
detector adds none (24 before and after, the same photos). Its rules were set
on 1 331 other simulated photos, apart from all these.

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

A photo read as valid by the rules takes the same time as before: the
detector only runs when they find no seed, an invalid one, or an incomplete
layout. It then adds 1.9 s for the network and the reading of the words
again: on a Pi Zero, badly framed cards that the rules could not read took 5
to 8 s to be refused and take 13 to 17 s to be read. The line numbers add
nothing measurable: they are read from network outputs already computed,
except for numbers cut off the line.

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
| `haller-1-detector.npz` | word detector, when the rules find no valid seed | `bcda59a220e52979139f44a634a0946dd9d4fe72cd4b246ee23edc6437964aa4` |

They are produced, from scratch and reproducibly, by the separate
[Haller](https://github.com/haller-project/haller) repository (tag
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
- Word detector only: simulated photos of cards and lists (the same fonts and
  EMNIST letters, laid out on paper and photographed with framing defects:
  cut, close, far, turned, in perspective, among other cards, under shadows)
  and GNHK pages with their word boxes.
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
