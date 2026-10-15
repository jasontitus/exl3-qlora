#include <metal_stdlib>
using namespace metal;
// Clockwise 90-degree rotation. p: rows, cols, input stride, output stride,
// input offset, output offset; all strides/offsets count F32 elements.
// Output shape is cols x rows. No aliasing. Preserve every input bit.
kernel void strided_rotate90(device const float* x [[buffer(0)]],
 device float* y [[buffer(1)]], constant uint* p [[buffer(2)]],
 uint2 tid [[thread_position_in_grid]]) {
 uint rows=p[0], cols=p[1];
 uint col=tid.x, row=tid.y;
 if(col >= cols || row >= rows) return;
 y[p[5]+col*p[3]+(rows-1-row)] = x[p[4]+row*p[2]+col];
}
