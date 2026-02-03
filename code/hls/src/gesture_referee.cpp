#include "typedefs.h"

void gesture_referee(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    data_t buffer[6][120]; // 6 axes, 120 samples
    
    // 1. INTAKE (Pipelined)
    for (int j = 0; j < 120; j++) {
        for (int i = 0; i < 6; i++) {
            #pragma HLS PIPELINE II=1
            buffer[i][j] = in_stream.read().data * (data_t)0.000244;
        }
    }

    // 2. LOGIC (Simple Sum for Week 3 placeholder)
    data_t sum = 0;
    for(int k=0; k<120; k++) {
        #pragma HLS UNROLL factor=4
        sum += buffer[0][k];
    }

    // 3. OUTPUT
    axis_t res;
    res.data = sum;
    res.last = 1;
    out_stream.write(res);
}