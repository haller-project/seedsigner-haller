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


/* First convolution (1 input channel): input zero-padded ((Hc+2) x (W+2)),
 * accumulators Hc x W x O, weights [O][9] */
static void conv_first(const int16_t *in, int Hc, int W, const int16_t *w, const int32_t *bias, int O,
                       int32_t *acc)
{
    const int WB = W + 2;
    for (int o = 0; o < O; o++) {
        const int16_t *k = w + o * 9;
        const int32_t k0 = k[0], k1 = k[1], k2 = k[2], k3 = k[3], k4 = k[4],
                      k5 = k[5], k6 = k[6], k7 = k[7], k8 = k[8], b = bias[o];
        for (int y = 0; y < Hc; y++) {
            const int16_t *r0 = in + y * WB, *r1 = r0 + WB, *r2 = r1 + WB;
            int32_t *t = acc + (size_t)y * W * O + o;
            for (int x = 0; x < W; x++)
                t[(size_t)x * O] = b
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
CONV3X3(conv_8, 8)
CONV3X3(conv_16, 16)
CONV3X3(conv_24, 24)
CONV3X3(conv_32, 32)
CONV3X3(conv_48, 48)
CONV3X3(conv_64, 64)
CONV3X3(conv_96, 96)
CONV3X3(conv_128, 128)


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
    else if (C == 8)
        conv_8(in, Hc, W, w, b, O, acc);
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
    else if (C == 128)
        conv_128(in, Hc, W, w, b, O, acc);
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

    conv_first(b, H, W, net->w[0], net->b[0], net->c1, acc);
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

/* log(1 + exp(-d)) for d = i / 64, i = 0..LUT_SIZE: log-add by table. The
 * values are those of log1pf(expf(-i / 64.0f)) in glibc 2.41 on x86-64, written
 * out because libm implementations differ in the last bit on a few of them
 * (12 with glibc 2.36 on armhf): with a constant table the decoding gives the
 * same bits on every platform. */
#define LUT_SIZE 1024
static const float lut[LUT_SIZE + 1] = {
    0x1.62e43p-1f, 0x1.5ee83p-1f, 0x1.5af43p-1f, 0x1.57082ep-1f, 0x1.53242ep-1f, 0x1.4f482ap-1f,
    0x1.4b7422p-1f, 0x1.47a816p-1f, 0x1.43e406p-1f, 0x1.4027ecp-1f, 0x1.3c73c8p-1f, 0x1.38c798p-1f,
    0x1.352358p-1f, 0x1.318708p-1f, 0x1.2df2a2p-1f, 0x1.2a6622p-1f, 0x1.26e188p-1f, 0x1.2364cep-1f,
    0x1.1feffp-1f, 0x1.1c82eap-1f, 0x1.191db8p-1f, 0x1.15c054p-1f, 0x1.126abcp-1f, 0x1.0f1ce6p-1f,
    0x1.0bd6dp-1f, 0x1.089874p-1f, 0x1.0561ccp-1f, 0x1.0232d2p-1f, 0x1.fe16fcp-2f, 0x1.f7d796p-2f,
    0x1.f1a766p-2f, 0x1.eb866p-2f, 0x1.e5747p-2f, 0x1.df718ap-2f, 0x1.d97dap-2f, 0x1.d398a4p-2f,
    0x1.cdc282p-2f, 0x1.c7fb2ap-2f, 0x1.c24292p-2f, 0x1.bc98a4p-2f, 0x1.b6fd5p-2f, 0x1.b17086p-2f,
    0x1.abf23p-2f, 0x1.a6823ep-2f, 0x1.a120a2p-2f, 0x1.9bcd44p-2f, 0x1.968814p-2f, 0x1.9150fcp-2f,
    0x1.8c27eap-2f, 0x1.870ccap-2f, 0x1.81ff88p-2f, 0x1.7d000ep-2f, 0x1.780e4cp-2f, 0x1.732a28p-2f,
    0x1.6e538ep-2f, 0x1.698a6ap-2f, 0x1.64cea8p-2f, 0x1.60202ep-2f, 0x1.5b7eeap-2f, 0x1.56eac2p-2f,
    0x1.5263a2p-2f, 0x1.4de974p-2f, 0x1.497c2p-2f, 0x1.451b9p-2f, 0x1.40c7acp-2f, 0x1.3c805ep-2f,
    0x1.38458cp-2f, 0x1.341722p-2f, 0x1.2ff506p-2f, 0x1.2bdf24p-2f, 0x1.27d55ep-2f, 0x1.23d7ap-2f,
    0x1.1fe5d2p-2f, 0x1.1bffdcp-2f, 0x1.1825a4p-2f, 0x1.145714p-2f, 0x1.109412p-2f, 0x1.0cdc84p-2f,
    0x1.093058p-2f, 0x1.058f6ep-2f, 0x1.01f9b2p-2f, 0x1.fcde14p-3f, 0x1.f5debcp-3f, 0x1.eef528p-3f,
    0x1.e8212ap-3f, 0x1.e1628ep-3f, 0x1.dab928p-3f, 0x1.d424bep-3f, 0x1.cda526p-3f, 0x1.c73a2cp-3f,
    0x1.c0e39ep-3f, 0x1.baa14ep-3f, 0x1.b47308p-3f, 0x1.ae589ep-3f, 0x1.a851dcp-3f, 0x1.a25e92p-3f,
    0x1.9c7e92p-3f, 0x1.96b1a6p-3f, 0x1.90f7a2p-3f, 0x1.8b5056p-3f, 0x1.85bb8ep-3f, 0x1.80391ep-3f,
    0x1.7ac8d4p-3f, 0x1.756a8p-3f, 0x1.701df4p-3f, 0x1.6ae302p-3f, 0x1.65b976p-3f, 0x1.60a124p-3f,
    0x1.5b99ep-3f, 0x1.56a376p-3f, 0x1.51bdbap-3f, 0x1.4ce88p-3f, 0x1.48239ap-3f, 0x1.436ed8p-3f,
    0x1.3eca0cp-3f, 0x1.3a350cp-3f, 0x1.35afaap-3f, 0x1.3139b8p-3f, 0x1.2cd30cp-3f, 0x1.287b78p-3f,
    0x1.2432d2p-3f, 0x1.1ff8eep-3f, 0x1.1bcdap-3f, 0x1.17b0bep-3f, 0x1.13a21cp-3f, 0x1.0fa19p-3f,
    0x1.0baef2p-3f, 0x1.07ca18p-3f, 0x1.03f2d6p-3f, 0x1.002904p-3f, 0x1.f8d8f4p-4f, 0x1.f17a2p-4f,
    0x1.ea353ap-4f, 0x1.e309fp-4f, 0x1.dbf7f8p-4f, 0x1.d4ffp-4f, 0x1.ce1ebcp-4f, 0x1.c756dcp-4f,
    0x1.c0a71ap-4f, 0x1.ba0f22p-4f, 0x1.b38eaep-4f, 0x1.ad257p-4f, 0x1.a6d324p-4f, 0x1.a0977ep-4f,
    0x1.9a723p-4f, 0x1.9462fcp-4f, 0x1.8e6992p-4f, 0x1.8885b4p-4f, 0x1.82b714p-4f, 0x1.7cfd7p-4f,
    0x1.775886p-4f, 0x1.71c81p-4f, 0x1.6c4bcap-4f, 0x1.66e374p-4f, 0x1.618ec8p-4f, 0x1.5c4d8ap-4f,
    0x1.571f76p-4f, 0x1.52044cp-4f, 0x1.4cfbdp-4f, 0x1.4805c2p-4f, 0x1.4321e2p-4f, 0x1.3e4ff4p-4f,
    0x1.398fbep-4f, 0x1.34e1p-4f, 0x1.30438p-4f, 0x1.2bb706p-4f, 0x1.273b54p-4f, 0x1.22d034p-4f,
    0x1.1e756cp-4f, 0x1.1a2ac2p-4f, 0x1.15f004p-4f, 0x1.11c4f4p-4f, 0x1.0da96p-4f, 0x1.099d12p-4f,
    0x1.059fd4p-4f, 0x1.01b174p-4f, 0x1.fba372p-5f, 0x1.f400eep-5f, 0x1.ec7aeap-5f, 0x1.e5110cp-5f,
    0x1.ddc2eap-5f, 0x1.d69024p-5f, 0x1.cf785cp-5f, 0x1.c87b32p-5f, 0x1.c19846p-5f, 0x1.bacf3cp-5f,
    0x1.b41fb6p-5f, 0x1.ad896p-5f, 0x1.a70bd8p-5f, 0x1.a0a6cap-5f, 0x1.9a59dep-5f, 0x1.9424bap-5f,
    0x1.8e070ep-5f, 0x1.880084p-5f, 0x1.8210c8p-5f, 0x1.7c378ap-5f, 0x1.767474p-5f, 0x1.70c73ap-5f,
    0x1.6b2f8cp-5f, 0x1.65ad1cp-5f, 0x1.603f9cp-5f, 0x1.5ae6bep-5f, 0x1.55a23cp-5f, 0x1.5071cap-5f,
    0x1.4b551ap-5f, 0x1.464becp-5f, 0x1.4155f2p-5f, 0x1.3c72e8p-5f, 0x1.37a28ap-5f, 0x1.32e49p-5f,
    0x1.2e38bap-5f, 0x1.299ec2p-5f, 0x1.25166ap-5f, 0x1.209f6ep-5f, 0x1.1c398ep-5f, 0x1.17e48ap-5f,
    0x1.13a026p-5f, 0x1.0f6c22p-5f, 0x1.0b4844p-5f, 0x1.07344ep-5f, 0x1.033006p-5f, 0x1.fe7662p-6f,
    0x1.f6ab28p-6f, 0x1.eefdf2p-6f, 0x1.e76e4cp-6f, 0x1.dffbcap-6f, 0x1.d8a5fep-6f, 0x1.d16c7cp-6f,
    0x1.ca4ed8p-6f, 0x1.c34cbp-6f, 0x1.bc6598p-6f, 0x1.b5992ap-6f, 0x1.aee704p-6f, 0x1.a84ec2p-6f,
    0x1.a1dp-6f, 0x1.9b6a64p-6f, 0x1.951d88p-6f, 0x1.8ee916p-6f, 0x1.88ccacp-6f, 0x1.82c7fp-6f,
    0x1.7cda8cp-6f, 0x1.770424p-6f, 0x1.714462p-6f, 0x1.6b9aeep-6f, 0x1.660776p-6f, 0x1.6089a8p-6f,
    0x1.5b212ep-6f, 0x1.55cdbcp-6f, 0x1.508efap-6f, 0x1.4b649ep-6f, 0x1.464e5ep-6f, 0x1.414be6p-6f,
    0x1.3c5ceep-6f, 0x1.37812ep-6f, 0x1.32b858p-6f, 0x1.2e0226p-6f, 0x1.295e52p-6f, 0x1.24cc92p-6f,
    0x1.204ca4p-6f, 0x1.1bde42p-6f, 0x1.17812ap-6f, 0x1.13351ap-6f, 0x1.0ef9dp-6f, 0x1.0acf0ap-6f,
    0x1.06b48cp-6f, 0x1.02aa14p-6f, 0x1.fd5eccp-7f, 0x1.f5888cp-7f, 0x1.edd0ecp-7f, 0x1.e6377cp-7f,
    0x1.debbc2p-7f, 0x1.d75d4ep-7f, 0x1.d01bb2p-7f, 0x1.c8f676p-7f, 0x1.c1ed38p-7f, 0x1.baff84p-7f,
    0x1.b42cf2p-7f, 0x1.ad7518p-7f, 0x1.a6d792p-7f, 0x1.a053f4p-7f, 0x1.99e9e2p-7f, 0x1.9398f2p-7f,
    0x1.8d60c6p-7f, 0x1.8741p-7f, 0x1.81393ep-7f, 0x1.7b4922p-7f, 0x1.757054p-7f, 0x1.6fae78p-7f,
    0x1.6a0334p-7f, 0x1.646e32p-7f, 0x1.5eef18p-7f, 0x1.598596p-7f, 0x1.543156p-7f, 0x1.4ef204p-7f,
    0x1.49c75p-7f, 0x1.44b0eap-7f, 0x1.3fae82p-7f, 0x1.3abfccp-7f, 0x1.35e47ap-7f, 0x1.311c4p-7f,
    0x1.2c66d2p-7f, 0x1.27c3ecp-7f, 0x1.23334p-7f, 0x1.1eb48cp-7f, 0x1.1a4788p-7f, 0x1.15ebecp-7f,
    0x1.11a178p-7f, 0x1.0d67e8p-7f, 0x1.093ef8p-7f, 0x1.05266cp-7f, 0x1.011dfep-7f, 0x1.fa4ae6p-8f,
    0x1.f27916p-8f, 0x1.eac616p-8f, 0x1.e33168p-8f, 0x1.dbba98p-8f, 0x1.d46134p-8f, 0x1.cd24c6p-8f,
    0x1.c604dap-8f, 0x1.bf0108p-8f, 0x1.b818dap-8f, 0x1.b14be8p-8f, 0x1.aa99c6p-8f, 0x1.a4020cp-8f,
    0x1.9d8452p-8f, 0x1.972032p-8f, 0x1.90d548p-8f, 0x1.8aa33p-8f, 0x1.84898cp-8f, 0x1.7e87f6p-8f,
    0x1.789e16p-8f, 0x1.72cb8cp-8f, 0x1.6d0ffap-8f, 0x1.676b0ep-8f, 0x1.61dc64p-8f, 0x1.5c63acp-8f,
    0x1.57009p-8f, 0x1.51b2b8p-8f, 0x1.4c79d2p-8f, 0x1.47558ap-8f, 0x1.424592p-8f, 0x1.3d4998p-8f,
    0x1.38614ep-8f, 0x1.338c64p-8f, 0x1.2eca94p-8f, 0x1.2a1b8ep-8f, 0x1.257f0ap-8f, 0x1.20f4bep-8f,
    0x1.1c7c64p-8f, 0x1.1815b2p-8f, 0x1.13c068p-8f, 0x1.0f7c3ap-8f, 0x1.0b48ecp-8f, 0x1.072638p-8f,
    0x1.0313dep-8f, 0x1.fe2338p-9f, 0x1.f63e66p-9f, 0x1.ee78ccp-9f, 0x1.e6d1eap-9f, 0x1.df494ep-9f,
    0x1.d7de7ap-9f, 0x1.d090fap-9f, 0x1.c9605ep-9f, 0x1.c24c34p-9f, 0x1.bb5408p-9f, 0x1.b4776ep-9f,
    0x1.adb5f6p-9f, 0x1.a70f3ap-9f, 0x1.a082cep-9f, 0x1.9a104cp-9f, 0x1.93b74cp-9f, 0x1.8d7768p-9f,
    0x1.87504p-9f, 0x1.81417p-9f, 0x1.7b4a9ap-9f, 0x1.756b6p-9f, 0x1.6fa36p-9f, 0x1.69f244p-9f,
    0x1.6457bp-9f, 0x1.5ed348p-9f, 0x1.5964bap-9f, 0x1.540baap-9f, 0x1.4ec7c6p-9f, 0x1.4998bap-9f,
    0x1.447e36p-9f, 0x1.3f77e4p-9f, 0x1.3a857ap-9f, 0x1.35a6a2p-9f, 0x1.30db16p-9f, 0x1.2c2286p-9f,
    0x1.277caap-9f, 0x1.22e934p-9f, 0x1.1e67dcp-9f, 0x1.19f85cp-9f, 0x1.159a6ep-9f, 0x1.114dccp-9f,
    0x1.0d123p-9f, 0x1.08e758p-9f, 0x1.04cd04p-9f, 0x1.00c2eep-9f, 0x1.f991b4p-10f, 0x1.f1bd0ap-10f,
    0x1.ea076ap-10f, 0x1.e2705p-10f, 0x1.daf74ap-10f, 0x1.d39bep-10f, 0x1.cc5d9ap-10f, 0x1.c53c08p-10f,
    0x1.be36b8p-10f, 0x1.b74d38p-10f, 0x1.b07f1ap-10f, 0x1.a9cbf4p-10f, 0x1.a3335ap-10f, 0x1.9cb4e4p-10f,
    0x1.96502cp-10f, 0x1.9004c6p-10f, 0x1.89d254p-10f, 0x1.83b87p-10f, 0x1.7db6bcp-10f, 0x1.77ccd4p-10f,
    0x1.71fa5ep-10f, 0x1.6c3efap-10f, 0x1.669a4cp-10f, 0x1.610bfep-10f, 0x1.5b93b6p-10f, 0x1.56311cp-10f,
    0x1.50e3dcp-10f, 0x1.4bab9cp-10f, 0x1.46880ep-10f, 0x1.4178ep-10f, 0x1.3c7dc2p-10f, 0x1.37966p-10f,
    0x1.32c26ep-10f, 0x1.2e01a2p-10f, 0x1.2953aep-10f, 0x1.24b846p-10f, 0x1.202f2p-10f, 0x1.1bb7f8p-10f,
    0x1.175284p-10f, 0x1.12fe7ep-10f, 0x1.0ebbap-10f, 0x1.0a89a8p-10f, 0x1.066854p-10f, 0x1.02575cp-10f,
    0x1.fcad0ep-11f, 0x1.f4cb22p-11f, 0x1.ed0876p-11f, 0x1.e5649p-11f, 0x1.dddef6p-11f, 0x1.d6772ep-11f,
    0x1.cf2cc4p-11f, 0x1.c7ff44p-11f, 0x1.c0ee3ap-11f, 0x1.b9f936p-11f, 0x1.b31fcap-11f, 0x1.ac6186p-11f,
    0x1.a5bep-11f, 0x1.9f34cep-11f, 0x1.98c588p-11f, 0x1.926fc6p-11f, 0x1.8c3324p-11f, 0x1.860f3ep-11f,
    0x1.8003b4p-11f, 0x1.7a102p-11f, 0x1.74342ap-11f, 0x1.6e6f7p-11f, 0x1.68c196p-11f, 0x1.632a42p-11f,
    0x1.5da91cp-11f, 0x1.583dccp-11f, 0x1.52e7fap-11f, 0x1.4da75p-11f, 0x1.487b7cp-11f, 0x1.43642cp-11f,
    0x1.3e610cp-11f, 0x1.3971cep-11f, 0x1.349624p-11f, 0x1.2fcdbep-11f, 0x1.2b185p-11f, 0x1.267592p-11f,
    0x1.21e534p-11f, 0x1.1d66f4p-11f, 0x1.18fa84p-11f, 0x1.149fa2p-11f, 0x1.105606p-11f, 0x1.0c1d6cp-11f,
    0x1.07f59p-11f, 0x1.03de3p-11f, 0x1.ffae1ap-12f, 0x1.f7bfc6p-12f, 0x1.eff0eap-12f, 0x1.e84108p-12f,
    0x1.e0afa2p-12f, 0x1.d93c42p-12f, 0x1.d1e674p-12f, 0x1.caadbcp-12f, 0x1.c391acp-12f, 0x1.bc91dp-12f,
    0x1.b5adbap-12f, 0x1.aee4fap-12f, 0x1.a83724p-12f, 0x1.a1a3ccp-12f, 0x1.9b2a8cp-12f, 0x1.94cafap-12f,
    0x1.8e84b4p-12f, 0x1.88575p-12f, 0x1.82427p-12f, 0x1.7c45aep-12f, 0x1.7660bp-12f, 0x1.709312p-12f,
    0x1.6adc7cp-12f, 0x1.653c92p-12f, 0x1.5fb2f6p-12f, 0x1.5a3f56p-12f, 0x1.54e154p-12f, 0x1.4f989ep-12f,
    0x1.4a64ep-12f, 0x1.4545c8p-12f, 0x1.403bp-12f, 0x1.3b4438p-12f, 0x1.366124p-12f, 0x1.319174p-12f,
    0x1.2cd4dap-12f, 0x1.282b0cp-12f, 0x1.2393bep-12f, 0x1.1f0ea8p-12f, 0x1.1a9b82p-12f, 0x1.163a02p-12f,
    0x1.11e9e8p-12f, 0x1.0daae8p-12f, 0x1.097ccp-12f, 0x1.055f2ep-12f, 0x1.0151f2p-12f, 0x1.faa996p-13f,
    0x1.f2ceeep-13f, 0x1.eb136ep-13f, 0x1.e3769ep-13f, 0x1.dbf804p-13f, 0x1.d49724p-13f, 0x1.cd538ep-13f,
    0x1.c62ccap-13f, 0x1.bf2268p-13f, 0x1.b833f4p-13f, 0x1.b16102p-13f, 0x1.aaa924p-13f, 0x1.a40becp-13f,
    0x1.9d88f8p-13f, 0x1.971fd6p-13f, 0x1.90d028p-13f, 0x1.8a9982p-13f, 0x1.847b86p-13f, 0x1.7e75dp-13f,
    0x1.7887fep-13f, 0x1.72b1b6p-13f, 0x1.6cf296p-13f, 0x1.674a42p-13f, 0x1.61b864p-13f, 0x1.5c3ca2p-13f,
    0x1.56d69ep-13f, 0x1.518608p-13f, 0x1.4c4a88p-13f, 0x1.4723cep-13f, 0x1.421184p-13f, 0x1.3d135ap-13f,
    0x1.3829p-13f, 0x1.335228p-13f, 0x1.2e8e84p-13f, 0x1.29ddc8p-13f, 0x1.253fa8p-13f, 0x1.20b3dcp-13f,
    0x1.1c3a1ap-13f, 0x1.17d21ap-13f, 0x1.137b98p-13f, 0x1.0f364ap-13f, 0x1.0b01f2p-13f, 0x1.06de48p-13f,
    0x1.02cb0cp-13f, 0x1.fd8ff8p-14f, 0x1.f5a9aep-14f, 0x1.ede2bcp-14f, 0x1.e63aa8p-14f, 0x1.deb0f8p-14f,
    0x1.d7453p-14f, 0x1.cff6dep-14f, 0x1.c8c588p-14f, 0x1.c1b0bep-14f, 0x1.bab80ep-14f, 0x1.b3db0ap-14f,
    0x1.ad194p-14f, 0x1.a67248p-14f, 0x1.9fe5b6p-14f, 0x1.997322p-14f, 0x1.931a24p-14f, 0x1.8cda58p-14f,
    0x1.86b358p-14f, 0x1.80a4c2p-14f, 0x1.7aae36p-14f, 0x1.74cf52p-14f, 0x1.6f07bep-14f, 0x1.695718p-14f,
    0x1.63bd06p-14f, 0x1.5e393p-14f, 0x1.58cb3ep-14f, 0x1.5372d8p-14f, 0x1.4e2fa8p-14f, 0x1.490158p-14f,
    0x1.43e79ap-14f, 0x1.3ee21cp-14f, 0x1.39f088p-14f, 0x1.351294p-14f, 0x1.3047fp-14f, 0x1.2b9052p-14f,
    0x1.26eb6cp-14f, 0x1.2258f4p-14f, 0x1.1dd8ap-14f, 0x1.196a2cp-14f, 0x1.150d4ap-14f, 0x1.10c1bcp-14f,
    0x1.0c8738p-14f, 0x1.085d7ep-14f, 0x1.044446p-14f, 0x1.003b56p-14f, 0x1.f884dp-15f, 0x1.f0b27ap-15f,
    0x1.e8ff3p-15f, 0x1.e16a74p-15f, 0x1.d9f3dp-15f, 0x1.d29accp-15f, 0x1.cb5eeep-15f, 0x1.c43fc8p-15f,
    0x1.bd3ce4p-15f, 0x1.b655d4p-15f, 0x1.af8a26p-15f, 0x1.a8d974p-15f, 0x1.a2434ep-15f, 0x1.9bc74cp-15f,
    0x1.956506p-15f, 0x1.8f1c16p-15f, 0x1.88ec18p-15f, 0x1.82d4a8p-15f, 0x1.7cd566p-15f, 0x1.76edeep-15f,
    0x1.711de8p-15f, 0x1.6b64f2p-15f, 0x1.65c2b2p-15f, 0x1.6036cep-15f, 0x1.5ac0ecp-15f, 0x1.5560b8p-15f,
    0x1.5015d8p-15f, 0x1.4adffap-15f, 0x1.45beccp-15f, 0x1.40b1f6p-15f, 0x1.3bb92cp-15f, 0x1.36d41ep-15f,
    0x1.32027ep-15f, 0x1.2d43fep-15f, 0x1.28985p-15f, 0x1.23ff2cp-15f, 0x1.1f784ap-15f, 0x1.1b035ep-15f,
    0x1.16a02p-15f, 0x1.124e4ep-15f, 0x1.0e0dap-15f, 0x1.09ddd4p-15f, 0x1.05bea4p-15f, 0x1.01afdp-15f,
    0x1.fb623p-16f, 0x1.f38474p-16f, 0x1.ebc5f2p-16f, 0x1.e4262ap-16f, 0x1.dca4a6p-16f, 0x1.d540ecp-16f,
    0x1.cdfa86p-16f, 0x1.c6d0fep-16f, 0x1.bfc3e4p-16f, 0x1.b8d2c6p-16f, 0x1.b1fd34p-16f, 0x1.ab42c4p-16f,
    0x1.a4a306p-16f, 0x1.9e1d94p-16f, 0x1.97b202p-16f, 0x1.915fecp-16f, 0x1.8b26ecp-16f, 0x1.85069cp-16f,
    0x1.7efeap-16f, 0x1.790e9p-16f, 0x1.733612p-16f, 0x1.6d74cap-16f, 0x1.67ca58p-16f, 0x1.62366p-16f,
    0x1.5cb88ep-16f, 0x1.575086p-16f, 0x1.51fdf4p-16f, 0x1.4cc082p-16f, 0x1.4797dcp-16f, 0x1.4283aep-16f,
    0x1.3d83aap-16f, 0x1.38977ep-16f, 0x1.33bedap-16f, 0x1.2ef972p-16f, 0x1.2a46fap-16f, 0x1.25a728p-16f,
    0x1.2119aep-16f, 0x1.1c9e46p-16f, 0x1.1834a8p-16f, 0x1.13dc9p-16f, 0x1.0f95b2p-16f, 0x1.0b5fcep-16f,
    0x1.073aa2p-16f, 0x1.0325e8p-16f, 0x1.fe42c2p-17f, 0x1.f65998p-17f, 0x1.ee8fd4p-17f, 0x1.e6e4f6p-17f,
    0x1.df588ap-17f, 0x1.d7ea12p-17f, 0x1.d0991ap-17f, 0x1.c9652ap-17f, 0x1.c24ddp-17f, 0x1.bb529cp-17f,
    0x1.b4731cp-17f, 0x1.adaee4p-17f, 0x1.a70588p-17f, 0x1.a0769ap-17f, 0x1.9a01b4p-17f, 0x1.93a66ep-17f,
    0x1.8d6464p-17f, 0x1.873b3p-17f, 0x1.812a7p-17f, 0x1.7b31c2p-17f, 0x1.7550c6p-17f, 0x1.6f872p-17f,
    0x1.69d472p-17f, 0x1.643862p-17f, 0x1.5eb296p-17f, 0x1.5942b4p-17f, 0x1.53e868p-17f, 0x1.4ea358p-17f,
    0x1.497334p-17f, 0x1.4457a6p-17f, 0x1.3f505ep-17f, 0x1.3a5d0cp-17f, 0x1.357d5ep-17f, 0x1.30b10ap-17f,
    0x1.2bf7cp-17f, 0x1.275134p-17f, 0x1.22bd2p-17f, 0x1.1e3b36p-17f, 0x1.19cb3p-17f, 0x1.156cc8p-17f,
    0x1.111fb8p-17f, 0x1.0ce3b6p-17f, 0x1.08b884p-17f, 0x1.049ddep-17f, 0x1.009382p-17f, 0x1.f9325ep-18f,
    0x1.f15d4cp-18f, 0x1.e9a74cp-18f, 0x1.e20feap-18f, 0x1.da96aap-18f, 0x1.d33b12p-18f, 0x1.cbfcacp-18f,
    0x1.c4db0ap-18f, 0x1.bdd5b2p-18f, 0x1.b6ec3ap-18f, 0x1.b01e2ep-18f, 0x1.a96b26p-18f, 0x1.a2d2b4p-18f,
    0x1.9c547p-18f, 0x1.95eff2p-18f, 0x1.8fa4d2p-18f, 0x1.8972acp-18f, 0x1.83591ep-18f, 0x1.7d57c4p-18f,
    0x1.776e44p-18f, 0x1.719c36p-18f, 0x1.6be144p-18f, 0x1.663d1p-18f, 0x1.60af4p-18f, 0x1.5b377cp-18f,
    0x1.55d568p-18f, 0x1.5088b4p-18f, 0x1.4b5108p-18f, 0x1.462e12p-18f, 0x1.411f7ep-18f, 0x1.3c24fep-18f,
    0x1.373e3ep-18f, 0x1.326af2p-18f, 0x1.2daacep-18f, 0x1.28fd84p-18f, 0x1.2462cap-18f, 0x1.1fda58p-18f,
    0x1.1b63ep-18f, 0x1.16ff22p-18f, 0x1.12abd2p-18f, 0x1.0e69acp-18f, 0x1.0a387p-18f, 0x1.0617d4p-18f,
    0x1.02079cp-18f, 0x1.fc0f06p-19f, 0x1.f42e96p-19f, 0x1.ec6d6ap-19f, 0x1.e4cb04p-19f, 0x1.dd46ecp-19f,
    0x1.d5e0a6p-19f, 0x1.ce97cp-19f, 0x1.c76bc4p-19f, 0x1.c05c3ep-19f, 0x1.b968bep-19f, 0x1.b290d6p-19f,
    0x1.abd416p-19f, 0x1.a53212p-19f, 0x1.9eaa62p-19f, 0x1.983c9ep-19f, 0x1.91e85cp-19f, 0x1.8bad3ap-19f,
    0x1.858ad2p-19f, 0x1.7f80c4p-19f, 0x1.798ebp-19f, 0x1.73b43p-19f, 0x1.6df0fp-19f, 0x1.68448cp-19f,
    0x1.62aeaep-19f, 0x1.5d2efap-19f, 0x1.57c51ap-19f, 0x1.5270b6p-19f, 0x1.4d3178p-19f, 0x1.48070ep-19f,
    0x1.42f124p-19f, 0x1.3def6cp-19f, 0x1.39019p-19f, 0x1.342744p-19f, 0x1.2f603ep-19f, 0x1.2aac2ap-19f,
    0x1.260ac2p-19f, 0x1.217bbep-19f, 0x1.1cfecep-19f, 0x1.1893aep-19f, 0x1.143a1ap-19f, 0x1.0ff1c8p-19f,
    0x1.0bba76p-19f, 0x1.0793ep-19f, 0x1.037dc2p-19f, 0x1.feefb8p-20f, 0x1.f703dcp-20f, 0x1.ef377p-20f,
    0x1.e789f6p-20f, 0x1.dffaf8p-20f, 0x1.d889f8p-20f, 0x1.d13682p-20f, 0x1.ca001ep-20f, 0x1.c2e65cp-20f,
    0x1.bbe8c6p-20f, 0x1.b506fp-20f, 0x1.ae406cp-20f, 0x1.a794cap-20f, 0x1.a103a2p-20f, 0x1.9a8c8cp-20f,
    0x1.942f1cp-20f, 0x1.8deafp-20f, 0x1.87bfa6p-20f, 0x1.81acd4p-20f, 0x1.7bb21ep-20f, 0x1.75cf24p-20f,
    0x1.700388p-20f, 0x1.6a4eeap-20f, 0x1.64b0f2p-20f, 0x1.5f2944p-20f, 0x1.59b78cp-20f, 0x1.545b6cp-20f,
    0x1.4f1494p-20f, 0x1.49e2acp-20f, 0x1.44c564p-20f, 0x1.3fbc66p-20f, 0x1.3ac766p-20f, 0x1.35e612p-20f,
    0x1.31181ap-20f, 0x1.2c5d36p-20f, 0x1.27b51ap-20f, 0x1.231f76p-20f, 0x1.1e9c06p-20f, 0x1.1a2a7ep-20f,
    0x1.15ca9ap-20f, 0x1.117c12p-20f, 0x1.0d3ea4p-20f, 0x1.091208p-20f, 0x1.04f5fep-20f, 0x1.00ea42p-20f,
    0x1.f9dd2cp-21f, 0x1.f2057p-21f, 0x1.ea4cd8p-21f, 0x1.e2b2e2p-21f, 0x1.db3718p-21f, 0x1.d3d902p-21f,
    0x1.cc982ap-21f, 0x1.c5741cp-21f, 0x1.be6c64p-21f, 0x1.b78092p-21f, 0x1.b0b03ap-21f, 0x1.a9faecp-21f,
    0x1.a3604p-21f, 0x1.9cdfcap-21f, 0x1.96792p-21f, 0x1.902bdep-21f, 0x1.89f7ap-21f, 0x1.83dcp-21f,
    0x1.7dd8ap-21f, 0x1.77ed1ep-21f, 0x1.721918p-21f, 0x1.6c5c36p-21f, 0x1.66b618p-21f, 0x1.612666p-21f,
    0x1.5bacc8p-21f, 0x1.5648e2p-21f, 0x1.50fa66p-21f, 0x1.4bc0f6p-21f, 0x1.469c42p-21f, 0x1.418bf8p-21f,
    0x1.3c8fc8p-21f, 0x1.37a76p-21f, 0x1.32d272p-21f, 0x1.2e10b2p-21f, 0x1.2961d2p-21f, 0x1.24c58ap-21f,
    0x1.203b8ep-21f, 0x1.1bc398p-21f, 0x1.175d5cp-21f, 0x1.130896p-21f, 0x1.0ec5p-21f, 0x1.0a9256p-21f,
    0x1.067056p-21f, 0x1.025ebep-21f, 0x1.fcba96p-22f, 0x1.f4d77cp-22f, 0x1.ed13bp-22f, 0x1.e56eb6p-22f,
    0x1.dde814p-22f, 0x1.d67f5p-22f, 0x1.cf33f2p-22f, 0x1.c80588p-22f, 0x1.c0f3ap-22f, 0x1.b9fdc6p-22f,
    0x1.b3238cp-22f, 0x1.ac6486p-22f, 0x1.a5c044p-22f, 0x1.9f366p-22f, 0x1.98c66ep-22f, 0x1.92700cp-22f,
    0x1.8c32cep-22f, 0x1.860e54p-22f, 0x1.80023cp-22f, 0x1.7a0e22p-22f, 0x1.7431aap-22f, 0x1.6e6c76p-22f,
    0x1.68be28p-22f, 0x1.632666p-22f, 0x1.5da4d8p-22f, 0x1.583922p-22f, 0x1.52e2f2p-22f, 0x1.4da1eep-22f,
    0x1.4875c6p-22f, 0x1.435e26p-22f, 0x1.3e5abap-22f, 0x1.396b36p-22f, 0x1.348f48p-22f, 0x1.2fc6a2p-22f,
    0x1.2b10fap-22f, 0x1.266e02p-22f, 0x1.21dd7p-22f, 0x1.1d5efep-22f, 0x1.18f262p-22f, 0x1.149754p-22f,
    0x1.104d9p-22f, 0x1.0c14dp-22f, 0x1.07ecd2p-22f, 0x1.03d554p-22f, 0x1.ff9c24p-23f, 0x1.f7ad9cp-23f,
    0x1.efde8ep-23f, 0x1.e82e7ep-23f, 0x1.e09cfp-23f, 0x1.d9296ep-23f, 0x1.d1d37ep-23f, 0x1.ca9aaap-23f,
    0x1.c37e82p-23f, 0x1.bc7e92p-23f, 0x1.b59a6ap-23f, 0x1.aed19cp-23f, 0x1.a823bap-23f, 0x1.a1905ap-23f,
    0x1.9b1714p-23f, 0x1.94b78p-23f, 0x1.8e7136p-23f, 0x1.8843d4p-23f, 0x1.822ef6p-23f, 0x1.7c323cp-23f,
    0x1.764d46p-23f, 0x1.707fb2p-23f, 0x1.6ac928p-23f, 0x1.65294cp-23f, 0x1.5f9fcp-23f, 0x1.5a2c3p-23f,
    0x1.54ce42p-23f, 0x1.4f85a2p-23f, 0x1.4a51fap-23f, 0x1.4532f6p-23f, 0x1.402846p-23f, 0x1.3b319ap-23f,
    0x1.364eap-23f, 0x1.317f0ap-23f, 0x1.2cc28ep-23f, 0x1.2818dcp-23f, 0x1.2381aep-23f, 0x1.1efcb6p-23f,
    0x1.1a89bp-23f, 0x1.162852p-23f, 0x1.11d856p-23f, 0x1.0d9978p-23f, 0x1.096b74p-23f, 0x1.054e06p-23f,
    0x1.0140eep-23f, 0x1.fa87d4p-24f, 0x1.f2ad74p-24f, 0x1.eaf24p-24f, 0x1.e355bap-24f,
};


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


/* ---------- word detector ---------- */

/* Layers of the detector: input a (-1: the image), input b upsampled by 2 and
 * concatenated after a (-1: none), 2x2 max pooling after the convolution.
 * Encoder e1-e8, decoder d4-d1; the output layer reads d1. */
static const struct { int8_t a, b, pool; } DET_LAYERS[HW_DETECTOR_LAYERS] = {
    {-1, -1, 1}, {0, -1, 1}, {1, -1, 1}, {2, -1, 1}, {3, -1, 0}, {4, -1, 0},
    {5, -1, 1}, {6, -1, 0}, {5, 7, 0}, {2, 8, 0}, {1, 9, 0}, {0, 10, 0},
};


/* sizes (rows, columns) of the layer outputs; offsets in elements of the work
 * arrays: int32 accumulators, padded input, layer outputs, concatenation */
typedef struct {
    int h[HW_DETECTOR_LAYERS], w[HW_DETECTOR_LAYERS];
    size_t out[HW_DETECTOR_LAYERS], acc, input, cat, act;
} det_layout;


static void detector_layout(const hw_detector *net, det_layout *l)
{
    size_t acc = (size_t)HW_DETECTOR_HEIGHT * HW_DETECTOR_WIDTH * net->channels[0], act = 0, cat = 0;
    for (int i = 0; i < HW_DETECTOR_LAYERS; i++) {
        const int a = DET_LAYERS[i].a, b = DET_LAYERS[i].b;
        const int hi = a < 0 ? HW_DETECTOR_HEIGHT : l->h[a], wi = a < 0 ? HW_DETECTOR_WIDTH : l->w[a];
        const int p = DET_LAYERS[i].pool ? 2 : 1;
        l->h[i] = hi / p;
        l->w[i] = wi / p;
        acc = max_size(acc, (size_t)hi * wi * net->channels[i]);
        if (b >= 0)
            cat = max_size(cat, (size_t)hi * wi * (net->channels[a] + net->channels[b]));
        l->out[i] = act;
        act += ((size_t)l->h[i] * l->w[i] * net->channels[i] + 1) / 2 * 2;
    }
    const int last = HW_DETECTOR_LAYERS - 1;
    l->acc = max_size(acc, (size_t)l->h[last] * l->w[last] * HW_DETECTOR_OUTPUTS);
    l->input = (size_t)(HW_DETECTOR_HEIGHT + 2) * (HW_DETECTOR_WIDTH + 2);
    l->cat = (cat + 1) / 2 * 2;
    l->act = act;
}


size_t hw_detector_work_size(const hw_detector *net)
{
    det_layout l;
    detector_layout(net, &l);
    return l.acc * sizeof(int32_t) + (l.input + l.act + l.cat) * sizeof(int16_t);
}


/* Copies tensor t (h/up x w/up x c, nearest upsampling by up = 1 or 2) into
 * channels [offset, offset + c) of out (h x w x C) */
static void gather(const int16_t *t, int c, int up, int h, int w, int C, int offset, int16_t *out)
{
    const int wt = w / up;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            memcpy(out + ((size_t)y * w + x) * C + offset, t + ((size_t)(y / up) * wt + x / up) * c,
                   (size_t)c * sizeof(int16_t));
}


void hw_detector_maps(const hw_detector *net, const uint8_t *img, void *work, float *out)
{
    det_layout l;
    detector_layout(net, &l);
    int32_t *acc = work;
    int16_t *input = (int16_t *)(acc + l.acc), *act = input + l.input, *cat = act + l.act;
    const int HI = HW_DETECTOR_HEIGHT, WI = HW_DETECTOR_WIDTH;

    /* input zero-padded, 0-255 -> 0-2040 */
    memset(input, 0, sizeof(int16_t) * l.input);
    for (int y = 0; y < HI; y++)
        for (int x = 0; x < WI; x++)
            input[(y + 1) * (WI + 2) + x + 1] = (int16_t)(img[y * WI + x] << 3);

    for (int i = 0; i < HW_DETECTOR_LAYERS; i++) {
        const int a = DET_LAYERS[i].a, b = DET_LAYERS[i].b, p = DET_LAYERS[i].pool ? 2 : 1;
        const int O = net->channels[i];
        if (a < 0) {
            conv_first(input, HI, WI, net->w[i], net->b[i], O, acc);
            pool(acc, HI, WI, O, p, p, net->m[i], act + l.out[i]);
            continue;
        }
        const int h = l.h[a], w = l.w[a];
        int C = net->channels[a];
        const int16_t *in = act + l.out[a];
        if (b >= 0) {
            gather(act + l.out[a], net->channels[a], 1, h, w, C + net->channels[b], 0, cat);
            gather(act + l.out[b], net->channels[b], 2, h, w, C + net->channels[b], C, cat);
            C += net->channels[b];
            in = cat;
        }
        conv(in, h, w, C, net->w[i], net->b[i], O, acc);
        pool(acc, h, w, O, p, p, net->m[i], act + l.out[i]);
    }

    /* output layer: accumulator * scale + bias, maps channel-major */
    const int last = HW_DETECTOR_LAYERS - 1, h = l.h[last], w = l.w[last];
    static const int32_t zero[HW_DETECTOR_OUTPUTS];
    conv(act + l.out[last], h, w, net->channels[last], net->wo, zero, HW_DETECTOR_OUTPUTS, acc);
    for (int o = 0; o < HW_DETECTOR_OUTPUTS; o++)
        for (int i = 0; i < h * w; i++)
            out[(size_t)o * h * w + i] = (float)acc[(size_t)i * HW_DETECTOR_OUTPUTS + o] * net->so[o] + net->bo[o];
}


int hw_components(const uint8_t *mask, int h, int w, int32_t *labels, int32_t *work)
{
    int n = 0;
    memset(labels, 0, (size_t)h * w * sizeof(int32_t));
    for (int i = 0; i < h * w; i++) {
        if (!mask[i] || labels[i])
            continue;
        int top = 0;
        labels[i] = ++n;
        work[top++] = i;
        while (top > 0) {
            const int j = work[--top], y = j / w, x = j % w;
            for (int dy = -1; dy <= 1; dy++) {
                if (y + dy < 0 || y + dy >= h)
                    continue;
                for (int dx = -1; dx <= 1; dx++) {
                    const int k = j + dy * w + dx;
                    if (x + dx >= 0 && x + dx < w && mask[k] && !labels[k]) {
                        labels[k] = n;
                        work[top++] = k;
                    }
                }
            }
        }
    }
    return n;
}
