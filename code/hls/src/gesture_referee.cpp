#include "typedefs.h"
#include "weights.h"

void gesture_referee(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    // Internal Buffer: 6 axes x 120 samples
    static data_t imu_buf[6][120];
    #pragma HLS ARRAY_PARTITION variable=imu_buf complete dim=1

    // 1. INTAKE: Stream 720 values from DMA
    for (int j = 0; j < 120; j++) {
        for (int i = 0; i < 6; i++) {
            #pragma HLS PIPELINE II=1
            axis_t temp = in_stream.read();
            // Normalize raw sensor data (assume 16-bit signed to +/- 8g)
            imu_buf[i][j] = temp.data * (data_t)0.000244;
        }
    }

    // 2. INFERENCE: 1D-Convolution (8 Filters)
    data_t filter_outputs[8] = {0};
    for (int f = 0; f < 8; f++) {
        data_t sum_acc = 0;
        for (int t = 1; t < 119; t++) { // Sliding window
            #pragma HLS PIPELINE II=1
            data_t conv_val = 0;
            for (int c = 0; c < 6; c++) {
                conv_val += imu_buf[c][t-1] * g_conv_w[f][c][0];
                conv_val += imu_buf[c][t]   * g_conv_w[f][c][1];
                conv_val += imu_buf[c][t+1] * g_conv_w[f][c][2];
            }
            // ReLU + Global Accumulation (GAP)
            if (conv_val > 0) sum_acc += conv_val;
        }
        filter_outputs[f] = sum_acc;
    }

    // 3. DECISION: Find Max Filter (Simplification of FC layer)
    int move_id = 0;
    data_t max_val = -999;
    for (int i = 0; i < 4; i++) {
        if (filter_outputs[i] > max_val) {
            max_val = filter_outputs[i];
            move_id = i;
        }
    }

    // 4. OUTPUT
    axis_t res;
    res.data = (data_t)move_id;
    res.last = 1;
    out_stream.write(res);
}
