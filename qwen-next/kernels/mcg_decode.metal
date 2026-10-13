// Sushi MCG ring semantics: pinned bd4b3954f5a42a87effcdf8a68ea91a4b6a1a1b9.
// Upstream MIT notice is preserved in ../sources/SUSHI-LICENSE.
// Experimental standalone kernel; compilation is not numerical qualification.
#include <metal_stdlib>
using namespace metal;

// Host contract: one expert; positive K,N multiples of16; N/start/width
// tile-aligned; 16<=halfwords<=128 even; 8<=window<=16; products fit uint32;
// validated packed/output buffers and no aliasing. Reads packed[in/16,out/16,n].
kernel void qwen_mcg_decode(
    device const ushort* packed [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& K [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& start [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& halfwords [[buffer(6)]],
    constant uint& window [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= K * width) return;
    uint row = gid / width, col = gid % width + start;
    uint r = row % 16, c = col % 16;
    uint slot = ((c % 8) * 4 + (r % 8) / 2) * 8
              + (r % 2) + (r / 8) * 2 + (c / 8) * 4;
    uint ring_bits = halfwords * 16;
    uint end = ((slot + 1) * halfwords) / 16;
    uint bit_start = (end + ring_bits - 16) % ring_bits;
    uint word = bit_start / 32, shift = bit_start % 32;
    uint base = ((row / 16) * (N / 16) + col / 16) * halfwords;
    uint a = uint(packed[base + word * 2])
           | (uint(packed[base + word * 2 + 1]) << 16);
    uint code;
    if (shift <= 16) {
        code = a >> (16 - shift);
    } else {
        uint next = (word + 1) % (halfwords / 2);
        uint b = uint(packed[base + next * 2])
               | (uint(packed[base + next * 2 + 1]) << 16);
        code = (a << (shift - 16)) | (b >> (48 - shift));
    }
    uint mask = (1u << window) - 1u;
    uint mixed = (code & mask) * 0xCBAC1FEDu;
    uint pair = (mixed & 0x8fff8fffu) ^ 0x3b603b60u;
    output[gid] = half(float(as_type<half>(ushort(pair)))
                     + float(as_type<half>(ushort(pair >> 16))));
}
