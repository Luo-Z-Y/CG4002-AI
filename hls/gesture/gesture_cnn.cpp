#include "gesture_cnn.h"
#include "gesture_cnn_weights.h"
#include <ap_int.h>
#include <ap_fixed.h>

// =============================
// Tunables up front
// =============================
static const int FC1_W_PART = 16;   // keep as your original
static const int FC2_W_PART = 8;    // keep as your original

// Wider accumulators (keep F=8 so adders aren’t huge)
typedef ap_fixed<24, 16, AP_TRN, AP_SAT> conv_acc_t; // conv sums
typedef ap_fixed<32, 24, AP_TRN, AP_SAT> fc_acc_t;   // FC sums

static inline data_t relu(data_t x) {
#pragma HLS INLINE
    return (x > 0) ? x : (data_t)0;
}

static inline data_t q88_from_axis(ap_uint<32> w) {
#pragma HLS INLINE
    ap_int<16> raw = (ap_int<16>)w.range(15, 0);
    data_t v;
    v.range(15, 0) = raw;
    return v;
}

void gesture_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=return

    // Match your original banking strategy
#pragma HLS ARRAY_PARTITION variable=conv1_w cyclic factor=3 dim=1
#pragma HLS ARRAY_PARTITION variable=conv2_w cyclic factor=3 dim=1
#pragma HLS ARRAY_PARTITION variable=fc1_w  cyclic factor=FC1_W_PART dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_w  cyclic factor=FC2_W_PART dim=1

    // Input buffer
    data_t input_buffer[NUM_SENSORS][WINDOW_SIZE];
#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1

    // 1) Read data
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int c = 0; c < NUM_SENSORS; c++) {
#pragma HLS PIPELINE II=1
            axis_t temp = in_stream.read();
            input_buffer[c][t] = q88_from_axis(temp.data);
        }
    }

    // 2) Layer 1: conv + relu + maxpool
    data_t layer1[16][30];
#pragma HLS ARRAY_PARTITION variable=layer1 complete dim=1

    for (int o = 0; o < 16; o++) {
        for (int t = 0; t < 30; t++) {
            data_t max_val = (data_t)-128;

            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;
                conv_acc_t acc = (conv_acc_t)conv1_b[o];

                for (int i = 0; i < 6; i++) {
#pragma HLS PIPELINE II=1
                    for (int k = 0; k < 3; k++) {
#pragma HLS UNROLL
                        int in_t = curr_t + k - 1;
                        if (in_t >= 0 && in_t < WINDOW_SIZE) {
                            int w_idx = o * (6 * 3) + i * 3 + k;
                            acc += (conv_acc_t)(input_buffer[i][in_t] * (data_t)conv1_w[w_idx]);
                        }
                    }
                }

                data_t val = (acc > 0) ? (data_t)acc : (data_t)0;
                if (val > max_val) max_val = val;
            }
            layer1[o][t] = max_val;
        }
    }

    // 3) Layer 2
    data_t layer2[32][15];
#pragma HLS ARRAY_PARTITION variable=layer2 complete dim=1

    for (int o = 0; o < 32; o++) {
        for (int t = 0; t < 15; t++) {
            data_t max_val = (data_t)-128;

            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;
                conv_acc_t acc = (conv_acc_t)conv2_b[o];

                for (int i = 0; i < 16; i++) {
#pragma HLS PIPELINE II=1
                    for (int k = 0; k < 3; k++) {
#pragma HLS UNROLL
                        int in_t = curr_t + k - 1;
                        if (in_t >= 0 && in_t < 30) {
                            int w_idx = o * (16 * 3) + i * 3 + k;
                            acc += (conv_acc_t)(layer1[i][in_t] * (data_t)conv2_w[w_idx]);
                        }
                    }
                }

                data_t val = (acc > 0) ? (data_t)acc : (data_t)0;
                if (val > max_val) max_val = val;
            }
            layer2[o][t] = max_val;
        }
    }

    // 4) FC1 (no flat_idx++; no division/mod)
    data_t dense1[32];
    for (int d = 0; d < 32; d++) {
        fc_acc_t acc = (fc_acc_t)fc1_b[d];

        for (int c = 0; c < 32; c++) {
            for (int t = 0; t < 15; t++) {
#pragma HLS PIPELINE II=1
                int flat_idx = c * 15 + t;                  // simple multiply+add
                int w_idx    = d * FLATTEN_SIZE + flat_idx; // linear weight index
                acc += (fc_acc_t)(layer2[c][t] * (data_t)fc1_w[w_idx]);
            }
        }
        dense1[d] = (acc > 0) ? (data_t)acc : (data_t)0;
    }

    // 5) FC2
    data_t final_scores[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        fc_acc_t acc = (fc_acc_t)fc2_b[c];
        for (int d = 0; d < 32; d++) {
#pragma HLS PIPELINE II=1
            int w_idx = c * 32 + d;
            acc += (fc_acc_t)(dense1[d] * (data_t)fc2_w[w_idx]);
        }
        final_scores[c] = (data_t)acc;
    }

    // 6) Argmax + output
    int best_class = 0;
    data_t best_score = final_scores[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (final_scores[i] > best_score) {
            best_score = final_scores[i];
            best_class = i;
        }
    }

    axis_t result_packet;
    result_packet.data = (ap_uint<32>)best_class;
    result_packet.keep = 0xF;
    result_packet.strb = 0xF;
    result_packet.last = 1;
    out_stream.write(result_packet);
}
