// EXL3 format: upstream quant/codebook.cuh and tests/testlib/trellis.py.
// Inherits this repository's MIT license. JIT header for mx.fast.metal_kernel.
template <int CB>
inline half exl3_codebook(uint x) {
    if (CB == 2) {
        x *= 0x83DCD12Du;
        uint b = (x & 255u) + ((x >> 8) & 255u)
               + ((x >> 16) & 255u) + (x >> 24);
        // Preserve CUDA half FMA semantics: exact fp32 inputs, one half rounding.
        return half(metal::fma(float(1024u + b),
            float(as_type<half>(ushort(0x1eee))),
            float(as_type<half>(ushort(0xc931)))));
    }
    x = CB == 0 ? x * 89226354u + 64248484u : x * 0xCBAC1FEDu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return half(float(as_type<half>(ushort(x)))
              + float(as_type<half>(ushort(x >> 16))));
}

template <typename P>
inline uint exl3_u32(const device P* packed, uint offset) {
    return uint(ushort(packed[offset])) | (uint(ushort(packed[offset + 1])) << 16);
}
