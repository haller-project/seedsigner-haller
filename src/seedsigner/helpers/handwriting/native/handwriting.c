/*
 * Handwritten word recognition: integer word network, dictionary decoding and
 * flood fill. See handwriting.h. This is the only implementation: any change
 * that alters its output requires new reference vectors (resources/handwriting/
 * *-vectors.json, written by haller.vectors).
 */
#include "handwriting.h"

#include <math.h>
#include <string.h>

#define AMAX 2047       /* activations: 12-bit unsigned range */
#define SHIFT 16        /* fractional bits of the requantization multipliers */
#define H HW_INPUT_HEIGHT
#define NEG (-1e30f)


/* acc + a.lo * b.lo + a.hi * b.hi on pairs of int16, wrapping on overflow;
 * HW_GENERIC forces the portable C version (to compare both on ARM) */
#if !defined(HW_GENERIC) && (defined(__ARM_FEATURE_DSP) || (defined(__arm__) && defined(__ARM_ARCH) && __ARM_ARCH >= 6))
static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    int32_t r;
    __asm__("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}
#else
static inline int32_t smlad(uint32_t a, uint32_t b, int32_t acc)
{
    int32_t lo = (int32_t)(int16_t)(a & 0xffff) * (int16_t)(b & 0xffff);
    int32_t hi = (int32_t)(int16_t)(a >> 16) * (int16_t)(b >> 16);
    return (int32_t)((uint32_t)acc + (uint32_t)lo + (uint32_t)hi);
}
#endif


/* two consecutive int16 as one 32-bit word */
static inline uint32_t pair(const int16_t *p)
{
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}


/* ReLU and requantization of an accumulator to the 12-bit activation range */
static inline int16_t requant(int32_t acc, int32_t mul)
{
    if (acc <= 0)
        return 0;
    int64_t v = ((int64_t)acc * mul + (1 << (SHIFT - 1))) >> SHIFT;
    return (int16_t)(v > AMAX ? AMAX : v);
}


int hw_frames(int width)
{
    return width / 4 > 0 ? width / 4 : 1;
}


/* The ink map is padded on the right with 16 empty columns, the width being
 * rounded up to a multiple of 4, as in training. */
static int padded_width(int width)
{
    return (width + 16 + 3) / 4 * 4;
}


static size_t max_size(size_t a, size_t b)
{
    return a > b ? a : b;
}


/* sizes in elements of the three work arrays: int32 accumulators, two int16 */
static void work_layout(const hw_network *net, int width, size_t *acc, size_t *act)
{
    const size_t W = padded_width(width), W2 = W / 2, W4 = W / 4;
    *acc = max_size(max_size(H * W * net->c1, 16 * W2 * net->c2), 8 * W4 * net->c3);
    size_t a = (size_t)(H + 2) * (W + 2);
    a = max_size(a, 16 * W2 * net->c1);
    a = max_size(a, 8 * W4 * net->c2);
    a = max_size(a, 4 * W4 * net->c3);
    a = max_size(a, W4 * 2 * net->c3);
    a = max_size(a, W4 * net->hidden);
    *act = (a + 1) / 2 * 2;
}


size_t hw_network_work_size(const hw_network *net, int width)
{
    size_t acc, act;
    work_layout(net, width, &acc, &act);
    return acc * sizeof(int32_t) + 2 * act * sizeof(int16_t);
}


/* First convolution (1 input channel): input zero-padded ((H+2) x (W+2)),
 * accumulators H x W x c1 */
static void conv_first(const int16_t *in, int W, const hw_network *net, int32_t *acc)
{
    const int C1 = net->c1, WB = W + 2;
    for (int o = 0; o < C1; o++) {
        const int16_t *k = net->w[0] + o * 9;
        const int32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4],
                      k5 = k[5], k6 = k[6], k7 = k[7], k8 = k[8], b = net->b[0][o];
        for (int y = 0; y < H; y++) {
            const int16_t *r0 = in + y * WB, *r1 = r0 + WB, *r2 = r1 + WB;
            int32_t *t = acc + (size_t)y * W * C1 + o;
            for (int x = 0; x < W; x++)
                t[(size_t)x * C1] = b
                    + k0 * r0[x] + k1 * r0[x + 1] + k2 * r0[x + 2]
                    + k3 * r1[x] + k4 * r1[x + 1] + k5 * r1[x + 2]
                    + k6 * r2[x] + k7 * r2[x + 1] + k8 * r2[x + 2];
        }
    }
}


/* 3x3 convolution with zero padding, input Hc x W x CC int16, accumulators
 * Hc x W x O int32, weights [O][3][3][CC]; four outputs at a time. One copy
 * per channel count, so that the inner loop has a constant trip count. */
#define CONV3X3(NAME, CC)                                                                    \
static void NAME(const int16_t *in, int Hc, int W, const int16_t *w, const int32_t *b,      \
                 int O, int32_t *acc)                                                        \
{                                                                                            \
    const int K = 9 * (CC);                                                                  \
    for (int o = 0; o + 3 < O; o += 4) {                                                     \
        const int16_t *w0 = w + (size_t)o * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;     \
        for (int y = 0; y < Hc; y++) {                                                       \
            const int ya = y > 0 ? -1 : 0, yb = y < Hc - 1 ? 1 : 0;                          \
            for (int x = 0; x < W; x++) {                                                    \
                const int xa = x > 0 ? -1 : 0, xb = x < W - 1 ? 1 : 0;                       \
                int32_t s0 = b[o], s1 = b[o + 1], s2 = b[o + 2], s3 = b[o + 3];              \
                for (int dy = ya; dy <= yb; dy++)                                            \
                    for (int dx = xa; dx <= xb; dx++) {                                      \
                        const int16_t *p = in + ((size_t)(y + dy) * W + (x + dx)) * (CC);    \
                        const int kk = ((dy + 1) * 3 + dx + 1) * (CC);                       \
                        for (int c = 0; c < (CC); c += 2) {                                  \
                            const uint32_t v = pair(p + c);                                  \
                            s0 = smlad(v, pair(w0 + kk + c), s0);                            \
                            s1 = smlad(v, pair(w1 + kk + c), s1);                            \
                            s2 = smlad(v, pair(w2 + kk + c), s2);                            \
                            s3 = smlad(v, pair(w3 + kk + c), s3);                            \
                        }                                                                    \
                    }                                                                        \
                int32_t *t = acc + ((size_t)y * W + x) * O + o;                              \
                t[0] = s0; t[1] = s1; t[2] = s2; t[3] = s3;                                  \
            }                                                                                \
        }                                                                                    \
    }                                                                                        \
}
CONV3X3(conv_16, 16)
CONV3X3(conv_24, 24)
CONV3X3(conv_32, 32)
CONV3X3(conv_48, 48)
CONV3X3(conv_64, 64)
CONV3X3(conv_96, 96)


static void conv_any(const int16_t *in, int Hc, int W, int C, const int16_t *w, const int32_t *b, int O,
                     int32_t *acc)
{
    for (int o = 0; o < O; o++) {
        const int16_t *wo = w + (size_t)o * 9 * C;
        for (int y = 0; y < Hc; y++)
            for (int x = 0; x < W; x++) {
                int32_t s = b[o];
                for (int dy = -1; dy <= 1; dy++) {
                    if (y + dy < 0 || y + dy >= Hc)
                        continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        if (x + dx < 0 || x + dx >= W)
                            continue;
                        const int16_t *p = in + ((size_t)(y + dy) * W + (x + dx)) * C;
                        const int16_t *k = wo + ((dy + 1) * 3 + dx + 1) * C;
                        for (int c = 0; c < C; c += 2)
                            s = smlad(pair(p + c), pair(k + c), s);
                    }
                }
                acc[((size_t)y * W + x) * O + o] = s;
            }
    }
}


static void conv(const int16_t *in, int Hc, int W, int C, const int16_t *w, const int32_t *b, int O,
                 int32_t *acc)
{
    if (O % 4)
        conv_any(in, Hc, W, C, w, b, O, acc);
    else if (C == 16)
        conv_16(in, Hc, W, w, b, O, acc);
    else if (C == 24)
        conv_24(in, Hc, W, w, b, O, acc);
    else if (C == 32)
        conv_32(in, Hc, W, w, b, O, acc);
    else if (C == 48)
        conv_48(in, Hc, W, w, b, O, acc);
    else if (C == 64)
        conv_64(in, Hc, W, w, b, O, acc);
    else if (C == 96)
        conv_96(in, Hc, W, w, b, O, acc);
    else
        conv_any(in, Hc, W, C, w, b, O, acc);
}


/* max pooling (ph x pw) of the accumulators (Hc x W x O), then requantization */
static void pool(const int32_t *acc, int Hc, int W, int O, int ph, int pw, const int32_t *mul, int16_t *out)
{
    const int H2 = Hc / ph, W2 = W / pw;
    for (int y = 0; y < H2; y++)
        for (int x = 0; x < W2; x++)
            for (int o = 0; o < O; o++) {
                int32_t m = INT32_MIN;
                for (int dy = 0; dy < ph; dy++)
                    for (int dx = 0; dx < pw; dx++) {
                        int32_t v = acc[(((size_t)y * ph + dy) * W + x * pw + dx) * O + o];
                        m = v > m ? v : m;
                    }
                out[((size_t)y * W2 + x) * O + o] = requant(m, mul[o]);
            }
}


/* 1D convolution, kernel 3, dilation dil, zero padding ("same"): input T x C,
 * output T x O requantized, weights [O][3][C] */
static void conv1d(const int16_t *in, int T, int C, const int16_t *w, const int32_t *b, const int32_t *mul,
                   int O, int dil, int16_t *out)
{
    const int K = 3 * C;
    int o = 0;
    for (; o + 3 < O; o += 4) {
        const int16_t *w0 = w + (size_t)o * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
        for (int t = 0; t < T; t++) {
            int32_t s0 = b[o], s1 = b[o + 1], s2 = b[o + 2], s3 = b[o + 3];
            for (int j = 0; j < 3; j++) {
                const int tt = t + (j - 1) * dil;
                if (tt < 0 || tt >= T)
                    continue;
                const int16_t *p = in + (size_t)tt * C;
                const int kk = j * C;
                for (int c = 0; c < C; c += 2) {
                    const uint32_t v = pair(p + c);
                    s0 = smlad(v, pair(w0 + kk + c), s0);
                    s1 = smlad(v, pair(w1 + kk + c), s1);
                    s2 = smlad(v, pair(w2 + kk + c), s2);
                    s3 = smlad(v, pair(w3 + kk + c), s3);
                }
            }
            int16_t *q = out + (size_t)t * O + o;
            q[0] = requant(s0, mul[o]);
            q[1] = requant(s1, mul[o + 1]);
            q[2] = requant(s2, mul[o + 2]);
            q[3] = requant(s3, mul[o + 3]);
        }
    }
    for (; o < O; o++)
        for (int t = 0; t < T; t++) {
            int32_t s = b[o];
            for (int j = 0; j < 3; j++) {
                const int tt = t + (j - 1) * dil;
                if (tt < 0 || tt >= T)
                    continue;
                for (int c = 0; c < C; c += 2)
                    s = smlad(pair(in + (size_t)tt * C + c), pair(w + ((size_t)o * 3 + j) * C + c), s);
            }
            out[(size_t)t * O + o] = requant(s, mul[o]);
        }
}


int hw_network_logits(const hw_network *net, const uint8_t *img, int width, void *work, float *out)
{
    const int W = padded_width(width), W2 = W / 2, W4 = W / 4, T = hw_frames(width);
    const int c1 = net->c1, c2 = net->c2, c3 = net->c3, d = net->hidden;
    size_t n_acc, n_act;
    work_layout(net, width, &n_acc, &n_act);
    int32_t *acc = work;
    int16_t *a = (int16_t *)(acc + n_acc), *b = a + n_act;

    /* input in b, zero-padded, 0-255 -> 0-2040 */
    memset(b, 0, sizeof(int16_t) * (H + 2) * (W + 2));
    for (int y = 0; y < H; y++)
        for (int x = 0; x < width; x++)
            b[(y + 1) * (W + 2) + x + 1] = (int16_t)(img[y * width + x] << 3);

    conv_first(b, W, net, acc);
    pool(acc, H, W, c1, 2, 2, net->m[0], a);
    conv(a, 16, W2, c1, net->w[1], net->b[1], c2, acc);
    pool(acc, 16, W2, c2, 2, 2, net->m[1], b);
    conv(b, 8, W4, c2, net->w[2], net->b[2], c3, acc);
    pool(acc, 8, W4, c3, 2, 1, net->m[2], a);
    conv(a, 4, W4, c3, net->w[3], net->b[3], c3, acc);
    pool(acc, 4, W4, c3, 2, 1, net->m[3], b);

    /* columns: a[t][h * c3 + c] = b[h][t][c] */
    for (int t = 0; t < W4; t++)
        for (int h = 0; h < 2; h++)
            memcpy(a + (size_t)t * 2 * c3 + h * c3, b + ((size_t)h * W4 + t) * c3, sizeof(int16_t) * c3);
    conv1d(a, W4, 2 * c3, net->w[4], net->b[4], net->m[4], d, 1, b);
    conv1d(b, W4, d, net->w[5], net->b[5], net->m[5], d, 2, a);

    for (int t = 0; t < T; t++) {
        const int16_t *p = a + (size_t)t * d;
        for (int k = 0; k < HW_NUM_CLASSES; k++) {
            const int16_t *wk = net->wo + (size_t)k * d;
            int32_t s = 0;
            for (int j = 0; j < d; j += 2)
                s = smlad(pair(p + j), pair(wk + j), s);
            out[(size_t)k * T + t] = (float)s * net->so[k] + net->bo[k];
        }
    }
    return T;
}


void hw_log_softmax(float *z, int frames)
{
    for (int t = 0; t < frames; t++) {
        float m = z[t], s = 0.0f;
        for (int k = 1; k < HW_NUM_CLASSES; k++)
            m = z[(size_t)k * frames + t] > m ? z[(size_t)k * frames + t] : m;
        for (int k = 0; k < HW_NUM_CLASSES; k++)
            s += expf(z[(size_t)k * frames + t] - m);
        const float l = m + logf(s);
        for (int k = 0; k < HW_NUM_CLASSES; k++)
            z[(size_t)k * frames + t] -= l;
    }
}


/* ---------- dictionary decoding ---------- */

/* log(1 + exp(-d)) for d = i / 64, i = 0..LUT_SIZE: log-add by table */
#define LUT_SIZE 1024
static float lut[LUT_SIZE + 1];


void hw_init(void)
{
    for (int i = 0; i <= LUT_SIZE; i++)
        lut[i] = log1pf(expf(-i / 64.0f));
}


static inline float logadd(float a, float b)
{
    float m = a > b ? a : b, d = a > b ? a - b : b - a;
    if (d >= 16.0f)
        return m;
    float f = d * 64.0f;
    int i = (int)f;
    return m + lut[i] + (f - i) * (lut[i + 1] - lut[i]);
}


/* blank or digit at frame t */
static float prefix_logprob(const float *lp, int frames, int t)
{
    float p = lp[t];
    for (int c = HW_FIRST_DIGIT; c < HW_NUM_CLASSES; c++)
        p = logadd(p, lp[(size_t)c * frames + t]);
    return p;
}


size_t hw_decoder_work_size(const hw_trie *trie)
{
    return 4 * (size_t)trie->nodes * sizeof(float);
}


/*
 * Every node n carries two CTC states: L (its letter) and B (a blank after its
 * letter); the root carries the prefix state P (blank or digit, looping).
 * L_n is entered from L_n, B_parent, L_parent if its letter differs, and P if
 * the parent is the root. The score of a word ending at node n is
 * logadd(L_n, B_n) at the last frame.
 */
void hw_word_scores(const hw_trie *trie, const float *lp, int frames, float beam, void *work, float *out)
{
    const int N = trie->nodes;
    const int32_t *parent = trie->parent;
    const int16_t *letter = trie->letter;
    float *L = work, *B = L + N, *L2 = B + N, *B2 = L2 + N;

    float P = prefix_logprob(lp, frames, 0);
    L[0] = B[0] = NEG;
    float best = P;
    for (int n = 1; n < N; n++) {
        L[n] = parent[n] == 0 ? lp[(size_t)letter[n] * frames] : NEG;
        B[n] = NEG;
        best = L[n] > best ? L[n] : best;
    }
    for (int t = 1; t < frames; t++) {
        const float blank = lp[t], pref = prefix_logprob(lp, frames, t);
        const float floor = beam > 0.0f ? best - beam : NEG;
        float next_best = P + pref;
        L2[0] = B2[0] = NEG;
        for (int n = 1; n < N; n++) {
            const int p = parent[n], l = letter[n];
            if (L[n] < floor && B[n] < floor && (p == 0 ? P < floor : (L[p] < floor && B[p] < floor))) {
                L2[n] = B2[n] = NEG;
                continue;
            }
            float v = L[n];
            if (p == 0)
                v = logadd(v, P);
            else {
                v = logadd(v, B[p]);
                if (letter[p] != l)
                    v = logadd(v, L[p]);
            }
            L2[n] = v + lp[(size_t)l * frames + t];
            B2[n] = logadd(B[n], L[n]) + blank;
            next_best = L2[n] > next_best ? L2[n] : next_best;
            next_best = B2[n] > next_best ? B2[n] : next_best;
        }
        best = next_best;
        P += pref;
        float *x = L; L = L2; L2 = x;
        x = B; B = B2; B2 = x;
    }
    for (int m = 0; m < trie->words; m++) {
        const int n = trie->word_end[m];
        out[m] = logadd(L[n], B[n]);
    }
}


void hw_border_connected(const uint8_t *mask, int h, int w, uint8_t *reached, int32_t *work)
{
    int top = 0;
    memset(reached, 0, (size_t)h * w);
    /* Depth-first fill from the border pixels of the mask; a pixel is marked
     * when it is pushed, so the stack holds at most h * w pixels. */
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const int i = y * w + x;
            if ((y == 0 || y == h - 1 || x == 0 || x == w - 1) && mask[i] && !reached[i]) {
                reached[i] = 1;
                work[top++] = i;
            }
        }
    while (top > 0) {
        const int i = work[--top], y = i / w, x = i % w;
        const int next[4] = {x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1};
        for (int k = 0; k < 4; k++)
            if (next[k] >= 0 && mask[next[k]] && !reached[next[k]]) {
                reached[next[k]] = 1;
                work[top++] = next[k];
            }
    }
}
