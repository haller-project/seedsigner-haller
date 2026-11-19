/*
 * Handwritten word recognition: integer word network, dictionary decoding and
 * flood fill used to find the seed card on the photo.
 *
 * Pure C99, no allocation: the caller provides the work buffers, whose sizes
 * are given by hw_network_work_size() and hw_decoder_work_size().
 */
#ifndef HANDWRITING_H
#define HANDWRITING_H

#include <stddef.h>
#include <stdint.h>

#define HW_INPUT_HEIGHT 32
#define HW_NUM_CLASSES 37   /* 0: CTC blank, 1-26: a-z, 27-36: 0-9 */
#define HW_FIRST_DIGIT 27

/*
 * Quantized word network (layout written by haller.quantize).
 *   conv 3x3 1->c1, max 2x2 ; conv 3x3 c1->c2, max 2x2 ;
 *   conv 3x3 c2->c3, max (2,1) ; conv 3x3 c3->c3, max (2,1) ;
 *   columns -> vectors of 2*c3 ; conv1d k3 -> hidden ; conv1d k3 dilation 2
 *   -> hidden ; conv1d k1 -> classes.
 * Activations are 12-bit integers in int16, channels last. Layer i has int16
 * weights w[i], int32 biases b[i] and int32 requantization multipliers m[i]
 * (fixed point, 16 fractional bits). The output layer has int16 weights wo
 * and float scales so and biases bo: logit = accumulator * so + bo.
 * c1, c2, c3 and hidden must be even (pairs of int16 per 32-bit load).
 */
typedef struct {
    int c1, c2, c3, hidden;
    const int16_t *w[6];
    const int32_t *b[6];
    const int32_t *m[6];
    const int16_t *wo;
    const float *so, *bo;
} hw_network;

/* Prefix tree of the dictionary, nodes stored parent before child (node 0:
 * root). Letters are class indices 1-36. */
typedef struct {
    int nodes, words;
    const int32_t *parent;   /* nodes */
    const int16_t *letter;   /* nodes */
    const int32_t *word_end; /* words: node of the last letter of each word */
} hw_trie;

/* Number of output frames for an ink map of the given width. */
int hw_frames(int width);

size_t hw_network_work_size(const hw_network *net, int width);

/*
 * Logits (HW_NUM_CLASSES x hw_frames(width), class-major) of the ink map
 * img (HW_INPUT_HEIGHT x width, row-major, 0 paper, 255 ink).
 * work: at least hw_network_work_size() bytes, 4-byte aligned.
 * Returns the number of frames.
 */
int hw_network_logits(const hw_network *net, const uint8_t *img, int width, void *work, float *out);

/* In place log-softmax over the classes of each frame. */
void hw_log_softmax(float *z, int frames);

size_t hw_decoder_work_size(const hw_trie *trie);

/*
 * CTC log-likelihood of every dictionary word (out: trie->words values) given
 * log-probabilities lp (HW_NUM_CLASSES x frames). A prefix of blanks and
 * digits is allowed before the word (line numbers). States more than beam
 * below the best state of the previous frame are dropped (beam 0: exact);
 * words whose states were all dropped get about -1e30.
 * work: at least hw_decoder_work_size() bytes, 4-byte aligned.
 */
void hw_word_scores(const hw_trie *trie, const float *lp, int frames, float beam, void *work, float *out);

/*
 * Sets reached (h x w, row-major) to 1 on the pixels of mask (h x w, nonzero:
 * in the mask) connected to the border of the image through pixels of the
 * mask, 4-connectivity, and to 0 elsewhere.
 * work: at least h * w int32 values.
 */
void hw_border_connected(const uint8_t *mask, int h, int w, uint8_t *reached, int32_t *work);

#endif
