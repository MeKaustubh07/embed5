#include <metal_stdlib>
using namespace metal;

// Brute-force search over 32-byte (256-bit) Embed 5 binary codes: one thread per page.
// Codes are the API's ubinary bytes (sign bits, MSB first), stored row-major as 2 x uint4 per page.
//
//   hamming      distance = popcount(code ^ query), 0..256: what "32 bytes per page" costs to scan
//   hamming_hist the same scan plus a 257-bin histogram of the distances, for the counting top-k below
//   kth          one thread walks the histogram: the smallest distance t with count(d <= t) >= k
//   compact      every page with d <= t appends (page, d) to a candidate list (one atomic counter); the host
//                orders the few candidates by (distance, page): an exact top-k with a deterministic tie-break
//   asym         the float question against the page's bits read as +-1, through a lookup table: for each of the
//                32 byte positions, the score of all 256 byte values (built on the host, 32 KB). Scoring a page
//                is 32 lookups and adds, so the scan stays memory-bound like hamming

inline uint pc4(uint4 v) { uint4 c = popcount(v); return c.x + c.y + c.z + c.w; }

constant uint NBINS = 257;
constant uint CAND0 = 260;      // state buffer: [0, 257) histogram | 257: threshold | 258: count | 260..: pairs

kernel void hamming(device ushort* out [[buffer(0)]], device const uint4* codes [[buffer(1)]],
                    constant uint4* q [[buffer(2)]], constant uint& N [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    out[i] = ushort(pc4(codes[2 * i] ^ q[0]) + pc4(codes[2 * i + 1] ^ q[1]));
}

kernel void hamming_hist(device ushort* out [[buffer(0)]], device const uint4* codes [[buffer(1)]],
                         constant uint4* q [[buffer(2)]], device atomic_uint* state [[buffer(3)]],
                         constant uint& N [[buffer(4)]],
                         uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                         uint tg [[threads_per_threadgroup]]) {
    threadgroup atomic_uint h[NBINS];             // per-group histogram: one global atomic per bin per group
    for (uint b = t; b < NBINS; b += tg) atomic_store_explicit(&h[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < N) {
        uint d = pc4(codes[2 * i] ^ q[0]) + pc4(codes[2 * i + 1] ^ q[1]);
        out[i] = ushort(d);
        atomic_fetch_add_explicit(&h[d], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = t; b < NBINS; b += tg) {
        uint c = atomic_load_explicit(&h[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&state[b], c, memory_order_relaxed);
    }
}

kernel void kth(device uint* state [[buffer(0)]], constant uint& K [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    uint acc = 0, b = 0;
    for (; b < NBINS; b++) { acc += state[b]; if (acc >= K) break; }
    state[NBINS] = min(b, NBINS - 1);
}

kernel void compact(device const ushort* dist [[buffer(0)]], device uint* state [[buffer(1)]],
                    constant uint& N [[buffer(2)]], constant uint& CAP [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    uint d = dist[i];
    if (d <= state[NBINS]) {
        uint slot = atomic_fetch_add_explicit((device atomic_uint*)&state[NBINS + 1], 1u, memory_order_relaxed);
        if (slot < CAP) { state[CAND0 + 2 * slot] = i; state[CAND0 + 2 * slot + 1] = d; }
    }
}

// ---- Hamming top-k for up to 8 questions per pass: each page's code is read once for all of them ----
// distances [Q, N]; state per question q at q * stride: the 257-bin histogram, threshold, count, candidate pairs
// (the layout of the single-question state, repeated)

kernel void hamming8_hist(device ushort* out [[buffer(0)]], device const uint4* codes [[buffer(1)]],
                          constant uint4* q [[buffer(2)]], device atomic_uint* state [[buffer(3)]],
                          constant uint& N [[buffer(4)]], constant uint& Q [[buffer(5)]],
                          constant uint& stride [[buffer(6)]],
                          uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint tg [[threads_per_threadgroup]]) {
    threadgroup atomic_uint h[8 * NBINS];
    for (uint b = t; b < Q * NBINS; b += tg) atomic_store_explicit(&h[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < N) {
        uint4 a = codes[2 * i], b = codes[2 * i + 1];
        for (uint k = 0; k < Q; k++) {
            uint d = pc4(a ^ q[2 * k]) + pc4(b ^ q[2 * k + 1]);
            out[k * N + i] = ushort(d);
            atomic_fetch_add_explicit(&h[k * NBINS + d], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = t; b < Q * NBINS; b += tg) {
        uint c = atomic_load_explicit(&h[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&state[(b / NBINS) * stride + b % NBINS], c, memory_order_relaxed);
    }
}

kernel void kth8(device uint* state [[buffer(0)]], constant uint& K [[buffer(1)]], constant uint& Q [[buffer(2)]],
                 constant uint& stride [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t >= Q) return;
    device uint* s = state + t * stride;
    uint acc = 0, b = 0;
    for (; b < NBINS; b++) { acc += s[b]; if (acc >= K) break; }
    s[NBINS] = min(b, NBINS - 1);
}

kernel void compact8(device const ushort* dist [[buffer(0)]], device uint* state [[buffer(1)]],
                     constant uint& N [[buffer(2)]], constant uint& Q [[buffer(3)]], constant uint& CAP [[buffer(4)]],
                     constant uint& stride [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    for (uint k = 0; k < Q; k++) {
        device uint* s = state + k * stride;
        uint d = dist[k * N + i];
        if (d <= s[NBINS]) {
            uint slot = atomic_fetch_add_explicit((device atomic_uint*)&s[NBINS + 1], 1u, memory_order_relaxed);
            if (slot < CAP) { s[CAND0 + 2 * slot] = i; s[CAND0 + 2 * slot + 1] = d; }
        }
    }
}

// ---- the float question against the page's bits, bit-serial ----
// The question quantized to qi in [-127, 127], stored as u = qi + 128 in 8 bit-planes laid out like the codes.
// sum_d qi_d * (+-1)_d ranks like sum_d qi_d * bit_d (the rest is a constant per question), and
// sum_d qi_d * bit_d = sum_k 2^k popcount(plane_k & code) - 128 popcount(code): 9 AND+popcounts per page, no
// lookups. Scores fit in [-32512, 32512]. Top-k is a two-level radix select on (score + 32768): a 256-bin
// histogram of the high byte during the scan, then one of the low byte inside the threshold bucket only.
// state (int): [0, 256) high-byte histogram | [256, 512) low-byte histogram | 512: bucket | 513: count above it |
//              514: threshold | 515: candidate count | 520..: (page, score) pairs

constant uint AS_HI = 0, AS_LO = 256, AS_BUCKET = 512, AS_ABOVE = 513, AS_T = 514, AS_COUNT = 515, AS_CAND = 520;

inline int asym_bits_score(uint4 a, uint4 b, constant uint4* planes) {
    int s = -128 * int(pc4(a) + pc4(b));
    for (uint k = 0; k < 8; k++) s += int(pc4(a & planes[2 * k]) + pc4(b & planes[2 * k + 1])) << k;
    return s;
}

kernel void asym_bits(device short* out [[buffer(0)]], device const uint4* codes [[buffer(1)]],
                      constant uint4* planes [[buffer(2)]], device atomic_int* state [[buffer(3)]],
                      constant uint& N [[buffer(4)]],
                      uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                      uint tg [[threads_per_threadgroup]]) {
    threadgroup atomic_int h[256];
    for (uint b = t; b < 256; b += tg) atomic_store_explicit(&h[b], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < N) {
        int s = asym_bits_score(codes[2 * i], codes[2 * i + 1], planes);
        out[i] = short(s);
        atomic_fetch_add_explicit(&h[uint(s + 32768) >> 8], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = t; b < 256; b += tg) {
        int c = atomic_load_explicit(&h[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&state[AS_HI + b], c, memory_order_relaxed);
    }
}

// the bucket holding the k-th highest score, and how many pages are in the buckets above it
kernel void asym_bucket(device int* state [[buffer(0)]], constant uint& K [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    int acc = 0, b = 255;
    for (; b > 0; b--) { if (acc + state[AS_HI + b] >= int(K)) break; acc += state[AS_HI + b]; }
    state[AS_BUCKET] = b;
    state[AS_ABOVE] = acc;
}

kernel void asym_lo_hist(device const short* scores [[buffer(0)]], device atomic_int* state [[buffer(1)]],
                         constant uint& N [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    uint v = uint(int(scores[i]) + 32768);
    if (int(v >> 8) == atomic_load_explicit(&state[AS_BUCKET], memory_order_relaxed))
        atomic_fetch_add_explicit(&state[AS_LO + (v & 255)], 1, memory_order_relaxed);
}

// the exact threshold: the k-th highest score
kernel void asym_threshold(device int* state [[buffer(0)]], constant uint& K [[buffer(1)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    int need = int(K) - state[AS_ABOVE], acc = 0, b = 255;
    for (; b > 0; b--) { acc += state[AS_LO + b]; if (acc >= need) break; }
    state[AS_T] = ((state[AS_BUCKET] << 8) | b) - 32768;
}

kernel void asym_compact(device const short* scores [[buffer(0)]], device int* state [[buffer(1)]],
                         constant uint& N [[buffer(2)]], constant uint& CAP [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    int s = scores[i];
    if (s >= state[AS_T]) {
        uint slot = atomic_fetch_add_explicit((device atomic_int*)&state[AS_COUNT], 1, memory_order_relaxed);
        if (slot < CAP) { state[AS_CAND + 2 * slot] = int(i); state[AS_CAND + 2 * slot + 1] = s; }
    }
}

// ---- the float question through a lookup table (the first version, kept for the benchmark) ----
// the code is read as two uint4 (as hamming does); byte j of the page is bits 8(j%4).. of 32-bit word j/4
// (little-endian), and lut[256 j + v] is the score of byte value v at position j
template <typename L> inline float asym_score(uint4 a, uint4 b, L lut) {
    uint w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float s = 0.0f;
    for (uint k = 0; k < 8; k++) {
        uint v = w[k], base = 1024 * k;
        s += lut[base + (v & 0xff)] + lut[base + 256 + ((v >> 8) & 0xff)]
           + lut[base + 512 + ((v >> 16) & 0xff)] + lut[base + 768 + (v >> 24)];
    }
    return s;
}

kernel void asym(device float* out [[buffer(0)]], device const uint4* codes [[buffer(1)]],
                 device const float* lut [[buffer(2)]], constant uint& N [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    out[i] = asym_score(codes[2 * i], codes[2 * i + 1], lut);
}

kernel void asym_const(device float* out [[buffer(0)]], device const uint4* codes [[buffer(1)]],
                       constant float* lut [[buffer(2)]], constant uint& N [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= N) return;
    out[i] = asym_score(codes[2 * i], codes[2 * i + 1], lut);
}
