#include <metal_stdlib>
using namespace metal;
kernel void row_sum(device const float* input [[buffer(0)]], device float* output [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
const uint row=gid; if(row>=rows)return; float total=0.0f, correction=0.0f; for(uint col=0;col<cols;++col) { float adjusted=max(0.5f * input[row * cols + col] + 0.25f, 0.0f)-correction; float updated=total+adjusted; correction=(updated-total)-adjusted; total=updated; }; output[row]=total;
}
