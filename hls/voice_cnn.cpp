#include "voice_cnn.h"
#include "voice_cnn_weights.h"
#include <ap_int.h>

// ============================================================
// Tunable optimisation parameters (keep up front)
// ============================================================

// Conv pipeline targets (II=4 is a good sweet spot here without exploding DSP)
static const int CONV1_II = 4;
static const int CONV2_II = 4;

// ============================================================
// Accumulator types (do NOT increase DSP; only widens adders a bit)
// ============================================================
// data_t is Q8.8 (ap_fixed<16,8,...>)
// Use accumulators with >=16 fractional bits so MAC does not collapse to Q*.8 mid-sum.
typedef ap_fixed<40, 20, AP_TRN, AP_SAT> conv_acc_t;
typedef ap_fixed<48, 24, AP_TRN, AP_SAT> pool_acc_t;
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> mul_acc_t; // exact Q16.16 product carrier

// ReLU
static inline data_t relu(data_t x) {
#pragma HLS INLINE
    return (x > (data_t)0) ? x : (data_t)0;
}

// Interpret AXIS[15:0] as signed Q8.8 -> data_t (bit-cast)
static inline data_t q88_from_axis(ap_uint<32> w) {
#pragma HLS INLINE
    ap_int<16> raw = (ap_int<16>)w.range(15, 0);
    data_t v;
    v.range(15, 0) = raw;
    return v;
}

// Force multiply into DSP and keep wider product precision before accumulation.
static inline mul_acc_t mul_dsp(data_t a, data_t b) {
#pragma HLS INLINE
    mul_acc_t p = (mul_acc_t)a * (mul_acc_t)b;
#pragma HLS bind_op variable=p op=mul impl=dsp
    return p;
}

void voice_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=return

    // ------------------------------------------------------------
    // Keep large constant weights in BRAM ROM (reduces LUT ROM)
    // ------------------------------------------------------------
#pragma HLS BIND_STORAGE variable=conv1_w type=rom_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv2_w type=rom_2p impl=bram
#pragma HLS BIND_STORAGE variable=fc_w    type=rom_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv1_b type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=conv2_b type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=fc_b    type=rom_1p impl=bram

    // ------------------------------------------------------------
    // Buffers in BRAM (avoid register/LUT explosion)
    // ------------------------------------------------------------
    data_t input_pad[VOICE_NUM_MFCC][VOICE_NUM_FRAMES + 2];
#pragma HLS BIND_STORAGE variable=input_pad type=ram_2p impl=bram

    // NOTE: We REMOVE b1_out and write pooled conv1 output directly into b1_pad.
    data_t b1_pad[VOICE_B1_CH][VOICE_B1_T + 2];
#pragma HLS BIND_STORAGE variable=b1_pad type=ram_2p impl=bram

    // pooled features after global average pooling (32 values)
    data_t pooled[VOICE_B2_CH];
#pragma HLS ARRAY_PARTITION variable=pooled complete dim=1  // tiny array, helps FC read

    // ============================================================
    // 1) Init input padding + read input samples
    // ============================================================
    for (int c = 0; c < VOICE_NUM_MFCC; c++) {
#pragma HLS PIPELINE II=1
        input_pad[c][0] = (data_t)0;
        input_pad[c][VOICE_NUM_FRAMES + 1] = (data_t)0;
    }

    // Expected stream order: [t][c] (time-major)
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
#pragma HLS PIPELINE II=1
            axis_t pkt = in_stream.read();
            input_pad[c][t + 1] = q88_from_axis(pkt.data);
        }
    }

    // ============================================================
    // 2) Conv1 + ReLU + MaxPool2 -> write DIRECTLY into b1_pad[o][t+1]
    //    (saves b1_out buffer + copy loops)
    // ============================================================
    // init b1_pad boundaries (padding=1 for conv2)
    for (int c = 0; c < VOICE_B1_CH; c++) {
#pragma HLS PIPELINE II=1
        b1_pad[c][0] = (data_t)0;
        b1_pad[c][VOICE_B1_T + 1] = (data_t)0;
    }

    for (int o = 0; o < VOICE_B1_CH; o++) {
        for (int t = 0; t < VOICE_B1_T; t++) {
#pragma HLS PIPELINE II=CONV1_II

            data_t max_val = (data_t)-128;

            // pool over p=0,1 (curr_t = 2t or 2t+1)
            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;   // 0..49
                int pad_t  = curr_t + 1;  // 1..50 (centre in padded input)

                // Wider accumulator for 40*3 = 120-term sum
                conv_acc_t acc = (conv_acc_t)conv1_b[o];

                for (int i = 0; i < VOICE_NUM_MFCC; i++) {
                    for (int k = 0; k < 3; k++) {
                        int w_idx = o * (VOICE_NUM_MFCC * 3) + i * 3 + k;
                        data_t x  = input_pad[i][pad_t + (k - 1)];
                        acc += (conv_acc_t)mul_dsp(x, conv1_w[w_idx]);
                    }
                }

                data_t v = relu((data_t)acc);
                if (v > max_val) max_val = v;
            }

            // Store into padded buffer at t+1 (so conv2 is branch-free)
            b1_pad[o][t + 1] = max_val;
        }
    }

    // ============================================================
    // 3) Conv2 + ReLU, then GlobalAvgPool ON-THE-FLY (no b2_out buffer)
    // ============================================================
    // invT is compile-time constant (VOICE_B2_T is constant)
    const data_t invT = (data_t)(1.0f / VOICE_B2_T);

    for (int o = 0; o < VOICE_B2_CH; o++) {
        pool_acc_t sum_t = (pool_acc_t)0;  // accumulate over 25 timesteps

        for (int t = 0; t < VOICE_B2_T; t++) {
#pragma HLS PIPELINE II=CONV2_II
            int pad_t = t + 1;

            // Wider accumulator for 16*3 = 48-term sum
            conv_acc_t acc = (conv_acc_t)conv2_b[o];

            for (int i = 0; i < VOICE_B1_CH; i++) {
                for (int k = 0; k < 3; k++) {
                    int w_idx = o * (VOICE_B1_CH * 3) + i * 3 + k;
                    data_t x  = b1_pad[i][pad_t + (k - 1)];
                    acc += (conv_acc_t)mul_dsp(x, conv2_w[w_idx]);
                }
            }

            data_t y = relu((data_t)acc);
            sum_t += (pool_acc_t)y;  // accumulate for avg pool
        }

        pooled[o] = (data_t)(sum_t * (pool_acc_t)invT);
    }

    // ============================================================
    // 4) FC: 32 -> 3
    // ============================================================
    data_t logits[VOICE_NUM_CLASSES];
#pragma HLS ARRAY_PARTITION variable=logits complete dim=1

    for (int c = 0; c < VOICE_NUM_CLASSES; c++) {
        conv_acc_t acc = (conv_acc_t)fc_b[c]; // small reduction, conv_acc_t is fine

        for (int i = 0; i < VOICE_B2_CH; i++) {
#pragma HLS PIPELINE II=1
            acc += (conv_acc_t)mul_dsp(pooled[i], fc_w[c * VOICE_B2_CH + i]);
        }

        logits[c] = (data_t)acc;
    }

    // ============================================================
    // 5) Argmax + output
    // ============================================================
    int best_class = 0;
    data_t best_score = logits[0];
    for (int c = 1; c < VOICE_NUM_CLASSES; c++) {
        if (logits[c] > best_score) {
            best_score = logits[c];
            best_class = c;
        }
    }

    axis_t out_pkt;
    out_pkt.data = (ap_uint<32>)best_class;
    out_pkt.keep = 0xF;
    out_pkt.strb = 0xF;
    out_pkt.last = 1;
    out_stream.write(out_pkt);
}
