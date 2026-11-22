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
3. Status: a seed is announced as **verified** only when the fingerprint
   written with the words is read and matches them (see
   [Trust in a reading](#trust-in-a-reading)). Any other reading is never
   called valid, sure or verified, however sure each word is: a full screen
   notice, headed « Not verified », shows the trust gauge (four segments, the
   last one reserved for « verified »), says what happened (no fingerprint,
   a fingerprint that does not match, two possible orders, or an invalid
   checksum), that every word must be compared with the paper, what the
   marks on the uncertain words mean, and that unmarked words can be wrong
   too. The text
   is paged with the up and down keys; its "I understand" button can only be
   selected from the last page, so that a click made by mistake does not skip
   it.
4. Mandatory review: every word is listed, and "Done" is only offered once each
   uncertain word has been confirmed. Uncertain words are flagged; selecting a word
   shows its handwritten image, up to three candidate words with their
   probability, and the usual keyboard to type it. A confirmed word is fixed and
   the checksum analysis is run again.
5. When the seed was not verified (no fingerprint, a fingerprint that does
   not match, or a word of a verified seed changed by the user), a screen
   shows the fingerprint of the words alone, to be written next to them, so
   that the next reading can be verified; when a fingerprint was read and did
   not match, it is quoted and every word is to compare with the paper.
6. The seed then goes to the standard flow (fingerprint, passphrase).

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

reading = read_seed(Image.open("photo.jpg"))
analysis, trust = reading.analyze(), reading.trust()
print(trust.level, trust.verified, " ".join(analysis.mnemonic), analysis.uncertain)
```

`trust.level` is 4 (verified by the fingerprint), 3, 2 or 1 (see
[Trust in a reading](#trust-in-a-reading)). `analysis.status`, internal to the
search, is `valid` (no uncertain word), `uncertain` (valid checksum, words to
check, listed in `uncertain`) or `invalid`; `read_seed` raises `WordsNotFound` when no card or
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
| `page.py` | the layout found on a photo (`WordLayout`: crops and outline of each word, reading orders) and the geometry shared by the detector (homography, spare lines of a list) |
| `preprocess.py` | turns a word crop into an ink map 32 pixels high |
| `word_model.py` | small convolutional network (about 160 000 parameters, CTC), run in 16-bit integers; two sets of weights, see [The Haller-1 model](#the-haller-1-model) |
| `decoder.py` | log-likelihood of each of the 2 048 BIP39 words (prefix tree, beam pruning); a line number before the word is accepted; probability of each line number |
| `detector.py` | word detector of Haller-1: finds the words, the line numbers and the cards on every photo, groups them into documents, lines and columns, and cuts out one crop per word, see [Word detector](#word-detector) |
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

The words are found by the word detector of Haller-1, with no rule about the
kind of document: a SeedSigner card, another card, a list on any paper. The
detector is a small fully convolutional network (U-Net, 223 000 parameters,
same 16-bit integer arithmetic as the word network) on the photo reduced to
288 x 384 pixels. It outputs maps of the handwritten words and of the line
numbers (printed or handwritten), of the kernels and the area of the cards,
of the links from a number to its word, and of the height of the boxes; it
puts no box on printed text. Then `detector.find_words`:

1. detects again on the region of the document when it covers less than 40 %
   of the photo (cards and lists far from the camera);
2. makes documents of the card kernels (a kernel with fewer than 3 words is
   not a document), and counts the words of a document by its numbers when
   they are read; a document cut by the frame of the photo comes after a whole
   one;
3. groups the words into lines, columns and grids; each word is placed in the
   slot of its number, and a word written off the line of its number goes to
   the nearest free slot, after the drift of the words from the lines of their
   numbers (perspective, height of the writing) is taken out; a word box that
   spans two numbered lines is shared between them;
4. tells the direction of reading by the numbers (before or after the words),
   otherwise reads the words both ways;
5. takes the order of the words from the numbers read when they tell it,
   otherwise from the columns and rows; when both orders of two columns pass
   the checksum, no word is certain;
6. tells a card cut by the frame of the photo: its area reaches a real edge of
   the photo beyond its text, and a word or number of the card is seen beyond
   the text on that side; otherwise the card is whole when the number of its
   last slot reads as the number of words.

A seed found this way is only valid (level 3 of the gauge at most, see
[Trust in a reading](#trust-in-a-reading)) if its probability is at least 0.9
(`MIN_DETECTOR_SEED_PROBABILITY`), no handwritten word detected with a score of
at least 0.8 and a length of at least one text height is left out of the
crops (`LEFT_OUT_SCORE`, `LEFT_OUT_LENGTH`), the words are not stretched more
than 1.7 times their usual spacing (`STRETCH`), and no faint ink left unread by
the word network lies beyond a word over 1.8 text heights
(`preprocess.unread_ink`, `UNREAD_INK`). Otherwise every word is marked
uncertain. These guards were set on simulated tuning pages and on the
development half of the real photos, never on the test photos.

### Fingerprint line

Many users write the fingerprint of the master key (8 hexadecimal digits,
the first 4 bytes of the HASH160 of the BIP32 master public key, as shown by
wallets and by SeedSigner) under their words, sometimes after « FP: »,
« xfp » or « fingerprint ». The list then has one line more than a valid
number of words (13 lines for 12 words). When a list found by the detector has a number of lines one above a
valid number of words, the lines that may be such a fingerprint (the first
line of the list and the last line of each column, a removal that leaves two
columns of equal length when there are two) are read by both networks
(`fingerprint.py`):

- the outputs of the network are decoded by CTC constrained to one of the
  prefixes « », « fp », « xfp », « mfp », « fingerprint » (the colon and
  spaces read as blanks) followed by exactly 8 hexadecimal digits: the sum
  over every string of 8 digits is the likelihood of the line being a
  fingerprint;
- the *word margin* compares it with the likelihood of the best BIP39 word
  of the line, the *fit* with the most probable path of the network without
  constraint (a line of two words merged by the detector, « 12. ORIENT 24.
  FEED », is better explained by any 8 digits than by any single word, but
  far worse than by its own reading).

The line is left out of the words when, on the mean of both networks, the
word margin is at least 17 and the fit at least −21
(`FINGERPRINT_MARGIN`, `FINGERPRINT_FIT`), and only if exactly one of the
candidate lines passes. The thresholds were chosen on simulated pages kept
apart from the test photos (DEV2 and lists with a fingerprint): 72 % of 660
fingerprints are left out; through the whole reading of every bench, none of
1 208 word lines comes within 3 of both thresholds (with a margin of 12, one
line of a 24-word card read as a list was left out). A fingerprint of digits only, which the second network reads poorly, is
often kept as a line, and the list is then reported with its number of lines
as before. The line left out is kept with the layout (`layout.fingerprint`).
The official SeedSigner card has no such line: its fingerprint box is on the
back.

### Labels and lines without a number

Some lists carry a label on a line of its own (« fingerprint » above the
fingerprint, « Seed words », « 12 words »...), or a fingerprint the detector
cuts in pieces. When a list is numbered on most of its lines with a valid
count but some lines carry no number, and its reading is not verified, it is
read again without one or two rows at its start or end
(`detector.extra_line_sets`: the first, the last, the first two, the last
two, the first and the last; rows of at most 3 pieces) whose removal leaves a
valid number of words. Each row left out must read as a fingerprint, as a
label (`fingerprint.LABELS`, CTC likelihood of the best label against the
best BIP39 word, `LABEL_MARGIN`) or, next to a fingerprint, as no word
(`handwriting.line_role`). Two fingerprint lines are checked together (mean
of their probabilities, which still sum to 1). At most one fingerprint check
is made per attempt and at most 3 seeds are checked: the bound of 1e-6 per
photo below is unchanged. A reading left with rows read as a fingerprint or a
label and not verified is trusted low. A list not found is not read again
this way: measured on a Pi Zero, it took 26 to 93 s on pages refused
without it in 4.5 to 33.5 s.

### Ink of the rows, empty numbered slots, rows across the columns

Three rules of the layout complete what the detector gives (`detector.py`):

- **Lines extended by the ink of their row** (`InkReach`, `ink_extend`): a
  line whose word map stops before the end of its strokes (« RESULT » boxed
  as « SULT ») is extended on either side, up to `INK_REACH` heights, over
  the strokes of a strip of the photo along its center line: pixels darker
  than half the way from the background to the ink of the line itself
  (`INK_LEVEL`), gaps between strokes of at most `INK_GAP` heights
  (`INK_DIGIT_GAP` for a line tried as the fingerprint), ruled or printed
  lines and the dark edges of the sheet left out (`INK_RULE`, `INK_DARK`).
  A side is extended only if it grows by half a height or more, if the
  strokes keep the height of the line (`INK_SHIFT`), and, in a column of
  aligned lines, if the line does not start more than half a height before
  the others (`INK_ALIGN`, `INK_MIN_LEAD`).
- **Empty numbered slots holding strokes** (`InkReach.slot`): on a numbered
  card or list whose slots are anchored on the numbers, a slot with no word
  box is taken as written when its central band holds at least one height
  of strokes (`INK_MIN_WORD`), only when the lines alone give no valid
  number of words, when the count becomes valid with these slots, and never
  in the second search (crossed-out words). Its word stays uncertain.
- **A row across the columns** (`bridged_columns`): one or two rows at the
  start or end of the list (a title, a fingerprint) that overlap the
  corridor between two columns and close it (`group_lines` would then merge
  the columns) are left out of the words, if the rest forms two columns or
  more (or a grid) of a valid number of words. This is done when the count
  is wrong, or when it is valid but the reading without these rows gives
  more columns; never in the second search. The rows left out go through the
  usual search of the fingerprint; failing a fingerprint at the strict
  thresholds, one passing the loose threshold is checked against the seeds
  read, and the list is not refused if it matches none
  (`layout.fingerprint_optional`).

Measured on 18 benches (11 383 simulated and real pages; 22/11/2026): 220
more exact seeds and 208 more verified seeds; on the tuning bench of the
fingerprint (779 pages), 28 and 34 more; no false verified seed, no loss
above 1 % of exact seeds for any writing or support. The rows
across the columns bring most of the gain (fingerprints written under or
above a list in columns). On a Pi Zero the ink strips add about 1.5 s per
photo; a few pages refused before and now read take 40 to 50 s more.

### Numbers that contradict the order

Once the order of the words is decided, the numbers the engine has already
read (no number is read for this check) are compared with the position of
their line (`NUMBER_CHECK`, `numbers_contradict`). Only sure numbers count:
a value read with a probability of at least `NUMBER_SURE` (0.9), linked to
its line by the link map (`LINK_THRESHOLD`), not by distance alone. The
list is incoherent when, in the order left that fits them best, at least
two sure numbers, and at least half of them, differ from their position:
a list that misses its first words (numbers starting at 6), a fingerprint
whose digits were taken for numbers, duplicated numbers.

An incoherent list is not taken as read. If a card kernel holds words (the
words of another document, or a part of the sheet taken for a card), the
photo is read again as one list with every word, card kernels ignored
(`find_words(merged=True)`); that reading is kept if at least
`MIN_NUMBERS_READ` sure numbers are read on it and agree with its order.
Otherwise the first reading is kept with every word uncertain and the trust
level Low (`numbers_incoherent`). A verified fingerprint always prevails.

Word crops read during one reading of a photo are kept by content
(`WORD_CACHE`): a crop read again by another search of the same photo
(crossed-out words, extra lines, merged list) reuses its ink map and the
outputs of the networks, which depend on the crop alone. No reading
changes; the second searches cost less on a Pi Zero.

A variant stays in the code, off: `CARD_SHAPE` (a card kernel holds a
document only if it has the size and the compactness of a card) frees the
words of SeedSigner cards cut by the frame on the benches. The second
reading without label or fingerprint rows (`layout.extra_lines`) is
offered only when every word box is on a line of the list; offering it
also when words are left out gained nothing on the benches and cost up to
88 s more on a Pi Zero.

Measured on the 19 frozen benches of the decision tools (22/11/2026): no
false verified seed, no loss; 1 more exact and verified seed on the photos
of the site (a list of 24 words in three numbered columns whose first five
words were taken for a card: read 21 lines, now 24 words with the
fingerprint verified), 1 to 2 more on the fingerprint and trap benches;
worst estimated time on a Pi Zero 42.8 → 37.7 s on the real photos.

### Fingerprint at the end of a word's row, optional fingerprint

- **At the end of a row** (`FINGERPRINT_TAIL`): a fingerprint written on
  the row of the first or last word of a column, after it (« 12. GUARD
  2cd9b02c »), is merged into the line of that word by `group_lines`. On
  the first and last line of each column, the last piece, more than
  `TAIL_GAP` (1.5) heights after the piece before it and at least
  `FINGERPRINT_MIN_LENGTH` heights long, is tried as the fingerprint at the
  strict thresholds; taken, it is cut off the line, the word keeping its
  part.
- **Optional fingerprint** (`FINGERPRINT_OPTIONAL_ANY`): with a valid count
  and no line at the strict thresholds, one crop of one line (that last
  piece if there is one, else the longest line outside the words) is tried
  at the loose margin; passing it, it is checked against the seeds read,
  and the list is kept as read if it matches none
  (`layout.fingerprint_optional`). One check per search, as on the other
  paths: the bound on a false verified seed per photo does not change. The
  line stays a word left out of the list until it matches. The outputs of
  the fingerprint network are kept per crop during a reading, and this
  line is checked on its crop only: the loose margin and the check cost
  one pass of the network.

Measured on the 19 frozen benches (22/11/2026): no false verified seed, no
loss; 71 more verified seeds on the fingerprint, trap and label benches, 2
more on the photos of the site (a pencil fingerprint under the list, a
fingerprint after the last word); worst estimated time on a Pi Zero 37.7 →
37.8 s on the real photos.

### Words abbreviated to 4 letters

Many backups, metal plates above all, hold only the first 4 letters of each
word: they are enough to tell the 2048 words of the BIP39 English list apart.
No abbreviated form equals another full word, so each one leads back to a
single word.

- **Decoder** (`decoder.ABBREVIATIONS`): the 1 503 abbreviated forms of the
  words longer than 4 letters are nodes of the prefix tree the decoder
  already walks; they add outputs, no state. The score of a word combines
  its two forms, `logaddexp(full, abbreviated − penalty)`. With
  `ABBREVIATIONS = False`, the scores are those of the engine without
  abbreviations, bit for bit.
- **Two levels.** Every word may be read abbreviated with a high penalty
  (`ABBREVIATION_PENALTY`, 9): words written in full read as before, a lone
  clear abbreviation still passes. A list where at least 3 words, and at
  least a quarter of the words (`ABBREVIATED_SHARE`), read better as an
  abbreviation than as any full word by `ABBREVIATION_EVIDENCE` (3) is an
  abbreviated list: all its words are read again with
  `ABBREVIATED_PAGE_PENALTY` (0), from the outputs already computed. A
  fixed low penalty word by word lost seeds on real photos: the forms of 4
  letters compete with the short words written in full (`exit` read
  `exil`, exile).
- The candidates stay full words: the screen shows the full word, and
  `WordReading.abbreviated` tells which candidates were read abbreviated.
  A word read abbreviated counts 4 letters for the stretched word guard.
  The trust in a reading keeps its rules: a form of 4 letters carries less
  evidence, its probabilities and the trust level come out lower.

Measured on the 20 frozen benches (22/11/2026, on 2c2a509): no false
verified seed, the real photos read the same photo by photo (261 photos),
no loss; 3 to 5 more exact seeds on four simulated benches of layouts. On
400 simulated pages written abbreviated (kept out of any tuning): 0 → 108
exact seeds, 1 217 → 2 385 right words, no false verified or false valid
seed; most of the rest are lists the detector does not find (short words in
spaced capitals, which the weights of Haller-1 never saw). Pi Zero: same
time on its real photos, −23 % on abbreviated pages, +4 % on the longest
simulated pages (the forms are combined in numpy for each crop).

### Check of the fingerprint

When a fingerprint line was left out, it is read again by the fingerprint
network (`haller-1-fingerprint.npz`, see [The Haller-1 model](#the-haller-1-model);
`fingerprint.FingerprintCheck`: probability of each digit at each position
by a forward-backward pass over the constrained CTC)
and compared with the fingerprints of the candidate seeds of the reading:
the 3 most probable valid seeds (`FINGERPRINT_CANDIDATES`) of every reading
of the photo (both orders of two columns, both networks and their blend), in
order of rank. The fingerprint
of a seed is computed as SeedSigner does (embit: PBKDF2-HMAC-SHA512 with 2048
rounds on the words alone, master key, HASH160 of the master public key, first 4
bytes).

- A candidate matches if the probability of its fingerprint given the line,
  among all strings of 8 hexadecimal digits, is at least 7e-4
  (`fingerprint.MIN_MATCH_PROBABILITY`). These probabilities sum to 1, so at
  most 1 428 strings reach it: the fingerprint of a wrong seed, uniform over
  2^32 values, matches by chance with a probability of at most 3.3e-7 per
  seed checked, whatever the calibration of the networks, and a photo, whose
  one fingerprint line is checked against 3 seeds at most, gets a wrong
  verified seed with a probability of at most 1e-6. The threshold is the
  lowest one with that bound for the number of candidates that verified the
  most right seeds on the tuning pages (1 to 16 candidates measured): 3. The
  right seed was among the first 3 candidates on all but 6 of 1 698 pages
  with a fingerprint read.
- A digit missed or one too many in the reading (a first digit taken for a
  line number, two strokes read as two digits) is taken into account: the
  line is also read as 7 and as 9 digits, and the probability of a
  fingerprint is a mixture of the three readings, weighted by their
  likelihood (prior 0.05 for each shift; `fingerprint.SHIFTS`). Each reading
  gives a distribution over the same strings of 8 digits, so the bound above
  holds; a digit missed gives a probability of 1/128 at most (8 positions, 16
  values).
- The first candidate that matches is the seed written: it is returned with
  every word certain (`reading.verified`), even when the reading hesitated
  (order of the columns, twin words, truncated words, a list taken for a
  card). The screen says « Verified » and shows the fingerprint computed.
- When none matches, the seed is never called wrong: a word or a digit may
  be misread. The words read are kept and every one of them is to check; the screen says
  that the fingerprint read does not match these words, and the trust level
  is low (see below).

### Trust in a reading

Decision of 21/11/2026: a seed is **verified**, the only state in which it is
announced as right, when and only when the fingerprint written with its
words is read and matches them. Without a fingerprint, or when it does not
match, the screens never say « valid », « sure » or « verified », even when
every word has a probability of 1: the checksum and the networks can agree on
a wrong seed (a truncated word read as another BIP39 word, for instance).
`SeedReading.trust(confirmed)` returns a `Trust`:

| Level | When | Announced share of fully right seeds |
|---|---|---|
| 4, verified | the fingerprint matches (`reading.verified`); a word confirmed by the user that differs from the verified seed takes it away | all |
| 3, high | no fingerprint read, and the checksum analysis leaves no word uncertain | at least 90 % |
| 2, medium | no fingerprint read, the seed is valid with words to check, and its probability is at least 0.5 (`MIN_MEDIUM_SEED_PROBABILITY`) | at least 40 % |
| 1, low | anything else: improbable or invalid seed, or a fingerprint read that matches none of the seeds read (evidence against them: 10 to 25 % of such seeds are right on the tuning pages) | none |

The announced shares (`TRUST_RATES`) were chosen on simulated tuning pages
only (DEV2, SERRE_DEV and lists with a fingerprint, 2 757 pages). With the
detector alone, the 95 % Wilson lower bound of the measured share is 97.6 %
for level 3 (845 of 858 seeds right) and 66.7 % for level 2 (302 of 426); the
announced shares stay 90 % and 40 % (a share of 60 % for level 2 would be
too optimistic on the red team: 79 of 194). On every test bench and on the
real photos, the measured share is at or above the announced one. The gauge is computed once the reading is done and does
not change the reading itself.

After a seed is loaded without being verified, SeedSigner shows its
fingerprint, that of the words alone, to be written next to them:
the next reading of that paper can then be verified.

The fingerprint is not secret; it stays in the reading, as the words do.

`read_seed(image, progress, on_word)` reports its progress after every step
of the page search and of the reading. Each step is weighted by its expected
duration on a Pi Zero (constants `*_COST` in `__init__.py`) and
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
written with its weights by `haller.detector_export` (Haller repository):
two synthetic images (a page of strokes, a card with
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

For Haller-1, the float models, the evaluation strips and the commands that
write these vectors again byte for byte are in `recipes/haller-1` of the
Haller repository.

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

Engine as shipped (word detector alone, fingerprint check, trust gauge),
measured on 21/11/2026 against the previous engine (rules of `page.py`, the
detector when they found no valid seed). Seeds verified by their fingerprint
(wrong ones) / seeds exact (every word right, whatever the trust level) /
words read correctly:

| Photos | Previous engine | Detector alone |
|---|---|---|
| Real photos, development half (38) | 0 (0) / 20 / 361 | 0 (0) / 23 / 443 |
| Real photos, test half (38) | 0 (0) / 24 / 389 | 0 (0) / 24 / 380 |
| Seed cards of an evaluation set, photographed (70) | 0 (0) / 0 / 7 | 0 (0) / 10 / 238 |
| Usual simulated photos (1 560) | 0 (0) / 1 340 / 20 519 | 0 (0) / 1 374 / 20 494 |
| Badly framed (1 167) | 0 (0) / 616 / 11 534 | 0 (0) / 710 / 12 906 |
| Red team (845) | 0 (0) / 509 / 8 542 | 0 (0) / 551 / 8 579 |
| Supports kept apart (900) | 0 (0) / 148 / 4 104 | 0 (0) / 267 / 6 363 |
| Several columns (240) | 0 (0) / 1 / 63 | 0 (0) / 31 / 769 |
| Tight columns (200) | 0 (0) / 28 / 578 | 0 (0) / 60 / 1 499 |
| Lists with a fingerprint line (1 200) | 207 (0) / 700 / 11 545 | 228 (0) / 802 / 12 741 |
| Red team with a fingerprint line (838) | 307 (0) / 384 / 6 106 | 320 (0) / 434 / 6 349 |

**0 wrong seeds verified in 548 fingerprint checks** (test benches; 173 more
on the tuning pages, none wrong). Losses of more than 1 % of exact seeds in a
category: SeedSigner cards of the red team with a fingerprint line (55 to 50)
and of the red team (48 to 47), real lowercase photos of the development half
(5 to 4), one real card of another model and one mixed writing of the test
half (1 to 0 each). The rest of this section records earlier stages.

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
detector adds none (24 before and after, the same photos). Its guards were set
on 1 331 other simulated photos, apart from all these.

The fingerprint line and its check were measured on simulated photos that
carry the fingerprint of their seed (written by the generator of the red team
under or above the list, in capitals or in the font of the page, sometimes
after « FP », « xfp » or « fingerprint »): the 920 photos of the red team with
the same pages plus the fingerprint line (cards have none), and 600 ordinary
lists (12 or 24 words, one or two columns), 1 357 of them with a fingerprint.
Seeds valid and exact / exact / wrong but reported as valid:

| Photos with a fingerprint | Before | Line left out | Checked, word networks | Checked, fingerprint network |
|---|---|---|---|---|
| All (1 357) | 13 / 67 / 7 | 455 / 582 / 18 | 425 / 596 / 7 | 559 / 603 / 7 |
| Ordinary lists, capitals (364) | 5 / 24 / 4 | 149 / 180 / 4 | 122 / 180 / 4 | 160 / 181 / 4 |
| Ordinary lists, lowercase (236) | 3 / 10 / 0 | 38 / 57 / 1 | 37 / 58 / 0 | 55 / 59 / 0 |
| Red team, capitals (441) | 4 / 22 / 3 | 217 / 264 / 7 | 208 / 270 / 3 | 262 / 273 / 3 |
| Red team, lowercase (316) | 1 / 11 / 0 | 51 / 81 / 6 | 58 / 88 / 0 | 82 / 90 / 0 |

Before, the extra line made most lists unreadable (« 13 lines found »).
With the check by the fingerprint network (last column, the engine as
shipped), 546 seeds are verified by their fingerprint, all of them right;
the wrong valid seeds left are all on photos where the fingerprint line was
not told apart (kept as a line, merged with a word, or the list then had a
valid number of lines): wherever a fingerprint was left out and read, no
wrong seed is reported as valid. A right seed whose fingerprint is misread
loses its certain words: 17 seeds with the fingerprint network, 144 when the
line was read by the two word networks (fourth column). The matching seed
was the chosen one in 91 % of the cases; 16 candidates find all but 3 of the
matches found among 64 (ranks 34 and 39 on the tuning photos). These figures
are those of the first threshold (1e-4, 16 candidates); see below for the
present one. On the 1 605 photos of the measurement above, the 920 of the red
team, the 1 167 badly framed ones and the real photos, which carry no
fingerprint, nothing changes: no line is left out (the thresholds keep a
margin) and the readings are the same, photo for photo.

### Photos of the Pi Zero camera that failed (21/11/2026)

Photos taken with the Pi Zero camera, of lists with a fingerprint written
above them, failed for reasons that no simulated bench had shown. Each one
is now handled by a general rule, measured alone on every bench:

- phantom words on the dark table below the sheet, and an object at the edge
  of the frame read as a word twice as tall as the others: when the count of
  lines is wrong, word boxes whose background is darker than 0.4 times the
  median background of the words, or at least twice the median height, are
  left out (`detector.InkStats.off_sheet`, `MAX_HEIGHT_SHARE`);
- specks and writing seen through the sheet: word kernels of less than 0.15
  times the median size are left out when the count is wrong (`SPECK`);
- a line crossed out and written again: a reading neither verified nor valid
  is tried again without the boxes whose share of dark pixels is at least
  1.8 times the median, and the better one is kept (`CROSSED_OUT`); applied
  directly, the rule would remove short or bold words;
- the first digit of a fingerprint taken for a line number and cut out of the
  line: a number box centered before the start of a line tried as the
  fingerprint is taken in with it (`detector.with_number`), and a reading
  one digit short or long is accepted (see above);
- a fingerprint line close to the threshold of the two word networks is
  judged by the fingerprint network, with the same thresholds
  (`FINGERPRINT_RESCUE_MARGIN`).

Against the previous engine, with the threshold of 7e-4 and 3 candidates:
no wrong verified seed on any bench; real photos (development half) 27 to
44 seeds right, 2 to 20 verified; real photos (test half) 24 to 25 right;
seeds right up on every simulated bench, with no loss above 1 % for any
writing or layout; verified seeds on the test pages with a fingerprint 980
to 1 048. One trust level falls short of its rate: « Medium » on the red team
pages with a fingerprint, 16 of 64 right (25 %, 33 % before): its right
pages became verified, the trap pages stayed.

Three more rules come from photos sent by participants: numbered lines written
in two bands (1-3 beside 4-6, then 7-9 beside 10-12) are read band by band when
the numbers tell it; words written in rows without columns are read row by
row; a word box running over the start of the next column is cut there. They
change no page of the simulated benches; on the real photos, one more seed is
right in the test half and four more among the participants' photos, with no
wrong verified seed.

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

With the detector alone (Pi Zero, 21/11/2026, application stopped, 9 photos
taken by the device or found online, two runs within 0.7 s of each other):
1.7 to 11.0 s, median 5.9 s, against 2.1 to 23.2 s, median 4.6 s, for the
engine made of rules only. A card takes 5.9 to 8.6 s (4.4 to 23.2 s
before: the detection on the region of the card adds a pass of the
detector), a list 4.3 to 11.0 s; a list the rules refused is read in 4.3 s.
The line numbers add
nothing measurable: they are read from network outputs already computed,
except for numbers cut off the line.

The word models (1 s; numpy is already imported at startup by
SeedSigner) are loaded while the photo is framed; the camera preview runs at
3.3 instead of 5 frames per second during that second. The progress bar never
stays still for more than 0.9 s. The display of the reading costs 3 to 5 % of
processor time.


A list with a fingerprint line costs more (measured on a Pi Zero, numpy and
embit of the device, 21/11/2026): each line that may be a fingerprint is read
by both networks and decoded (0.2 to 0.35 s per network for the decoding),
the line left out is read again by the fingerprint network, loaded the first
time (1 to 2 s), with the probability of each digit (0.3 to 0.5 s), the
fingerprint of the chosen seed is computed (0.071 s:
PBKDF2 0.062 s, master key 0.005 s, libsecp256k1 of embit) and, when it does
not match, those of the other candidates (2 at most since 21/11/2026, 16
before: 1.1 s, and 0.6 to 1.1 s to compare them with the line). About 2 to
5 s in all, plus the readings of the line as 7 and 9 digits (2 passes); a
photo without a fingerprint line takes the same time as before.

## The Haller-1 model

The words are read by Haller-1, the first version of the Haller word
recognizer. It is made of two networks of the same architecture, combined as
described in [How it works](#how-it-works):

| Weights (`src/seedsigner/resources/handwriting/`) | Role | SHA-256 |
|---|---|---|
| `haller-1-capitals.npz` | reads every photo; trained mostly on capitals | `2e9275f1acd0d347437e856160430e96a283ec484a48ab96a2ea1ca91531fe14` |
| `haller-1-lowercase.npz` | second reading; more lowercase and cursive writing | `5d340b354ced032992cb474f03bcc0b063f24ee8e1446d2953203b778652e509` |
| `haller-1-detector.npz` | word, line number and card detector: finds the words on every photo | `b33f6d02f1af87a4aff1cecc73368cc2230c2fa68d0eda977a112df437d501cc` |
| `haller-1-fingerprint.npz` | reads the fingerprint line that is checked; same architecture as the word networks | `b92cca552dc0a718d309a98b3e5f7ed6f20ee0be32f92aec8df1804d809de454` |

The fingerprint network was fine-tuned from the float capitals network
(`train --init`) in two steps, each the first epoch of a longer learning-rate
schedule (4 epochs at 5e-4, then 3 epochs at 3e-4), on 8 synthetic shards of
50 000 lines, half of them fingerprints (`generate --hex-share 0.5`: 8
hexadecimal digits, sometimes in two groups, after an optional « fp »,
« xfp », « fingerprint »…, in capitals or lowercase; two shards in cursive
fonts), and the filtered GNHK words. The first epoch of the second step was
kept, chosen on 4 000 separate fingerprint lines (two more epochs read the
words worse and the fingerprints no better). The data, both float models, the
commands and the content digests are in `recipes/haller-1-fingerprint` of the
Haller repository; its quantization gives this file byte for byte. On
those lines, the probability of the right fingerprint reaches the match
threshold of 1e-4 then in use for 80 % of them (43 % with the two word networks), 69 %
in cursive fonts (29 %); the 8 digits are read exactly in 35 % of them (9 %).
Its reference vectors were written without the comparison with the float
model, which disagreed on the best word of one strip close to a tie.

The detector was trained for 10 epochs on 136 shards of 1,000 photos and on
the GNHK windows: 120 shards of simulated pages (lists, sheets, printed seed
cards drawn from 25 card templates, handwriting from fonts and EMNIST
letters) and 16 shards of simulated handwriting composed on photos of real
blank papers and of printed official cards; one more simulated shard was
used for validation and for the calibration of the quantization. These
shards were made before the variety audit that the later synthetic data of
the project pass. The frozen code that made them, the training command, the
float model, the activation ranges of the quantization and the content
digests of the shards are in `recipes/haller-1-detector` of the Haller
repository; the quantization from these ranges and the export give this file
and its reference vectors byte for byte. The photos of blank supports and of
cards are listed there by digest, to be published.

They are produced, from scratch and reproducibly, by the separate
[Haller](https://github.com/haller-project/haller) repository (tag
`haller-1`): synthetic words, real handwritten words, training, then
quantization. Photos taken by users are never used.

`resources/handwriting/haller-weights.json` names the Haller version of the
weights (`haller-1`), the SHA-256 of each weight file and its release page;
`tests/test_handwriting_weights.py` checks that every weight file is listed
there, matches its digest, has its reference vectors and is the one the
engine loads, and that this page quotes the same digests. The weights stay in
this repository, since the device works offline; the
[Haller-1 release](https://github.com/haller-project/haller/releases/tag/haller-1)
(not published yet) carries the same files with the model card, the recipe
and the evaluation report. New weights (Haller-2, or Haller-1.1 without
retraining) replace the files, the manifest, the reference vectors and the
table above together.

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
  cut, close, far, turned, in perspective, among other cards, under shadows),
  printed seed card templates of SeedSigner and of other vendors (sources and
  licenses listed in `recipes/haller-1-detector/data/card_templates.json` of
  the Haller repository), photos of blank papers and of printed cards taken
  for the project with simulated handwriting composed on them, and GNHK pages
  with their word boxes.
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
