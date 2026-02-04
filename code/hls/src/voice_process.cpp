#include "typedefs.h"
#include "weights.h"

void voice_processor(hls::stream<axis_t> &audio_in, hls::stream<axis_t> &voice_out) {
    #pragma HLS INTERFACE axis port=audio_in
    #pragma HLS INTERFACE axis port=voice_out
    #pragma HLS INTERFACE s_axilite port=return bundle=VOICE_CTRL

    static data_t voice_buf[40][50];
    #pragma HLS ARRAY_PARTITION variable=voice_buf cyclic factor=4 dim=1

    // 1. INTAKE: 2000 MFCC coefficients
    for (int t = 0; t < 50; t++) {
        for (int f = 0; f < 40; f++) {
            #pragma HLS PIPELINE II=1
            voice_buf[f][t] = audio_in.read().data;
        }
    }

    // 2. INFERENCE: 1D-CNN (16 Filters)
    data_t pkmn_scores[16] = {0};
    for (int f = 0; f < 16; f++) {
        data_t sum_acc = 0;
        for (int t = 1; t < 49; t++) {
            #pragma HLS PIPELINE II=1
            data_t conv_val = 0;
            for (int c = 0; c < 40; c++) {
                #pragma HLS UNROLL factor=10
                conv_val += voice_buf[c][t-1] * v_conv_w[f][c][0];
                conv_val += voice_buf[c][t]   * v_conv_w[f][c][1];
                conv_val += voice_buf[c][t+1] * v_conv_w[f][c][2];
            }
            if (conv_val > 0) sum_acc += conv_val;
        }
        pkmn_scores[f] = sum_acc;
    }

    // 3. OUTPUT (Simulated Classification)
    axis_t res;
    res.data = (pkmn_scores[0] > pkmn_scores[1]) ? (data_t)0 : (data_t)1;
    res.last = 1;
    voice_out.write(res);
}
