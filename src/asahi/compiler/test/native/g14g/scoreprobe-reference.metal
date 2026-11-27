kernel void scoreprobe(device const float *x [[buffer(0)]],
                       device float *y [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
   y[i] = x[i] * 2.0f;
}
