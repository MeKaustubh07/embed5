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
