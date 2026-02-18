#include "voice_cnn.h"
#include "voice_cnn_weights.h"

static data_t relu(data_t x) {
    return (x > 0) ? x : (data_t)0;
}

void voice_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return

    // Input: [40, 50]
    data_t input_buf[VOICE_NUM_MFCC][VOICE_NUM_FRAMES];
    #pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1

    // Read [t][c] stream order, store channel-major internally.
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            #pragma HLS PIPELINE II=1
            axis_t pkt = in_stream.read();
            union { unsigned int i; float f; } cvt;
            cvt.i = pkt.data;
            input_buf[c][t] = (data_t)cvt.f;
        }
    }

    // =============================
    // Block1: Pointwise 1x1 (40->16)
    // =============================
    data_t b1_pw[VOICE_B1_CH][VOICE_NUM_FRAMES];
    #pragma HLS ARRAY_PARTITION variable=b1_pw complete dim=1

    for (int o = 0; o < VOICE_B1_CH; o++) {
        for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
            data_t s = pw1_b[o];
            for (int i = 0; i < VOICE_NUM_MFCC; i++) {
                int w_idx = o * VOICE_NUM_MFCC + i;
                s += input_buf[i][t] * pw1_w[w_idx];
            }
            b1_pw[o][t] = relu(s);
        }
    }

    // =============================
    // Block1: Depthwise k=3 + ReLU + MaxPool2 => [16,25]
    // =============================
    data_t b1_out[VOICE_B1_CH][VOICE_B1_T];
    #pragma HLS ARRAY_PARTITION variable=b1_out complete dim=1

    for (int c = 0; c < VOICE_B1_CH; c++) {
        for (int t = 0; t < VOICE_B1_T; t++) {
            data_t max_val = -128;
            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;
                data_t s = dw1_b[c];
                for (int k = 0; k < 3; k++) {
                    int in_t = curr_t + k - 1;
                    if (in_t >= 0 && in_t < VOICE_NUM_FRAMES) {
                        int w_idx = c * 3 + k;
                        s += b1_pw[c][in_t] * dw1_w[w_idx];
                    }
                }
                data_t v = relu(s);
                if (v > max_val) {
                    max_val = v;
                }
            }
            b1_out[c][t] = max_val;
        }
    }

    // =============================
    // Block2: Pointwise 1x1 (16->32)
    // =============================
    data_t b2_pw[VOICE_B2_CH][VOICE_B1_T];
    #pragma HLS ARRAY_PARTITION variable=b2_pw complete dim=1

    for (int o = 0; o < VOICE_B2_CH; o++) {
        for (int t = 0; t < VOICE_B1_T; t++) {
            data_t s = pw2_b[o];
            for (int i = 0; i < VOICE_B1_CH; i++) {
                int w_idx = o * VOICE_B1_CH + i;
                s += b1_out[i][t] * pw2_w[w_idx];
            }
            b2_pw[o][t] = relu(s);
        }
    }

    // =============================
    // Block2: Depthwise k=3 + ReLU + MaxPool2 => [32,12]
    // =============================
    data_t b2_out[VOICE_B2_CH][VOICE_B2_T];
    #pragma HLS ARRAY_PARTITION variable=b2_out complete dim=1

    for (int c = 0; c < VOICE_B2_CH; c++) {
        for (int t = 0; t < VOICE_B2_T; t++) {
            data_t max_val = -128;
            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;
                data_t s = dw2_b[c];
                for (int k = 0; k < 3; k++) {
                    int in_t = curr_t + k - 1;
                    if (in_t >= 0 && in_t < VOICE_B1_T) {
                        int w_idx = c * 3 + k;
                        s += b2_pw[c][in_t] * dw2_w[w_idx];
                    }
                }
                data_t v = relu(s);
                if (v > max_val) {
                    max_val = v;
                }
            }
            b2_out[c][t] = max_val;
        }
    }

    // Global average pool over time [32]
    data_t pooled[VOICE_B2_CH];
    for (int c = 0; c < VOICE_B2_CH; c++) {
        data_t s = 0;
        for (int t = 0; t < VOICE_B2_T; t++) {
            s += b2_out[c][t];
        }
        pooled[c] = s / (data_t)VOICE_B2_T;
    }

    // FC [5]
    data_t logits[VOICE_NUM_CLASSES];
    for (int c = 0; c < VOICE_NUM_CLASSES; c++) {
        data_t s = fc_b[c];
        for (int i = 0; i < VOICE_B2_CH; i++) {
            s += pooled[i] * fc_w[c * VOICE_B2_CH + i];
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
