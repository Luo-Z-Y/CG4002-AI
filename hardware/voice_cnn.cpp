#include "voice_cnn.h"
#include "voice_cnn_weights.h"

static data_t relu(data_t x) {
    return (x > 0) ? x : (data_t)0;
}

void voice_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return

    // Input buffer: [40, 50]
    data_t input_buf[VOICE_NUM_MFCC][VOICE_NUM_FRAMES];
    #pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1

    // Read 2000 float words from AXIS and cast to fixed-point
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            #pragma HLS PIPELINE II=1
            axis_t pkt = in_stream.read();
            union { unsigned int i; float f; } cvt;
            cvt.i = pkt.data;
            input_buf[c][t] = (data_t)cvt.f;
        }
    }

    // Conv1 + ReLU + MaxPool(2): [16, 25]
    data_t layer1[VOICE_CONV1_OUT_CH][VOICE_CONV1_OUT_T];
    #pragma HLS ARRAY_PARTITION variable=layer1 complete dim=1

    for (int o = 0; o < VOICE_CONV1_OUT_CH; o++) {
        for (int t = 0; t < VOICE_CONV1_OUT_T; t++) {
            data_t max_val = -128;
            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;
                data_t sum = conv1_b[o];

                for (int i = 0; i < VOICE_NUM_MFCC; i++) {
                    for (int k = 0; k < 3; k++) {
                        int in_t = curr_t + k - 1; // padding=1
                        if (in_t >= 0 && in_t < VOICE_NUM_FRAMES) {
                            int w_idx = o * (VOICE_NUM_MFCC * 3) + i * 3 + k;
                            sum += input_buf[i][in_t] * conv1_w[w_idx];
                        }
                    }
                }

                data_t v = relu(sum);
                if (v > max_val) {
                    max_val = v;
                }
            }
            layer1[o][t] = max_val;
        }
    }

    // Conv2 + ReLU: [32, 25]
    data_t layer2[VOICE_CONV2_OUT_CH][VOICE_CONV2_OUT_T];
    #pragma HLS ARRAY_PARTITION variable=layer2 complete dim=1

    for (int o = 0; o < VOICE_CONV2_OUT_CH; o++) {
        for (int t = 0; t < VOICE_CONV2_OUT_T; t++) {
            data_t sum = conv2_b[o];
            for (int i = 0; i < VOICE_CONV1_OUT_CH; i++) {
                for (int k = 0; k < 3; k++) {
                    int in_t = t + k - 1; // padding=1
                    if (in_t >= 0 && in_t < VOICE_CONV1_OUT_T) {
                        int w_idx = o * (VOICE_CONV1_OUT_CH * 3) + i * 3 + k;
                        sum += layer1[i][in_t] * conv2_w[w_idx];
                    }
                }
            }
            layer2[o][t] = relu(sum);
        }
    }

    // Global average pool over time: [32]
    data_t pooled[VOICE_CONV2_OUT_CH];
    for (int c = 0; c < VOICE_CONV2_OUT_CH; c++) {
        data_t s = 0;
        for (int t = 0; t < VOICE_CONV2_OUT_T; t++) {
            s += layer2[c][t];
        }
        pooled[c] = s / (data_t)VOICE_CONV2_OUT_T;
    }

    // FC: [3]
    data_t logits[VOICE_NUM_CLASSES];
    for (int c = 0; c < VOICE_NUM_CLASSES; c++) {
        data_t s = fc_b[c];
        for (int i = 0; i < VOICE_CONV2_OUT_CH; i++) {
            s += pooled[i] * fc_w[c * VOICE_CONV2_OUT_CH + i];
        }
        logits[c] = s;
    }

    // Argmax
    int best_class = 0;
    data_t best_score = logits[0];
    for (int c = 1; c < VOICE_NUM_CLASSES; c++) {
        if (logits[c] > best_score) {
            best_score = logits[c];
            best_class = c;
        }
    }

    axis_t out_pkt;
    out_pkt.data = best_class;
    out_pkt.keep = 0xF;
    out_pkt.strb = 0xF;
    out_pkt.last = 1;
    out_stream.write(out_pkt);
}
