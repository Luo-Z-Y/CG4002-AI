#include "voice_cnn.h"
#include "voice_cnn_weights.h"
#include <ap_int.h>

// Keep Q8.8 for storage/IO, but use much wider internal accumulation.
// 12 integer bits can still saturate (e.g., pooling 25 * ~127 > 2048).
typedef ap_fixed<48, 20, AP_TRN, AP_SAT> acc_t;

// ReLU
static inline data_t relu(data_t x) {
    return (x > (data_t)0) ? x : (data_t)0;
}

// Interpret AXIS[15:0] as signed Q8.8 and bit-cast into data_t (ap_fixed<16,8>).
// This avoids float->fixed conversion hardware entirely (big LUT saver).
static inline data_t q88_from_axis(ap_uint<32> w) {
    ap_int<16> raw = (ap_int<16>)w.range(15, 0);
    data_t v;
    v.range(15, 0) = raw;   // bit-level assignment into ap_fixed storage
    return v;
}

// Force multiply into DSP (trades DSP for LUT reduction)
static inline acc_t mul_dsp(data_t a, data_t b) {
    acc_t p = (acc_t)a * (acc_t)b;
#pragma HLS bind_op variable=p op=mul impl=dsp
    return p;
}

void voice_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=return

    // Keep DSP under control if needed (you can raise/lower later)
#pragma HLS ALLOCATION operation instances=mul limit=240

    // --------------------------------------------------------------------
    // Force large constant weights into BRAM-backed ROM (reduces LUT ROM).
    // If your tool complains about BIND_STORAGE, tell me your exact error
    // and I’ll swap to RESOURCE pragmas for your Vitis HLS build.
    // --------------------------------------------------------------------
#pragma HLS BIND_STORAGE variable=conv1_w type=rom_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv2_w type=rom_2p impl=bram
#pragma HLS BIND_STORAGE variable=fc_w    type=rom_2p impl=bram
#pragma HLS BIND_STORAGE variable=conv1_b type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=conv2_b type=rom_1p impl=bram
#pragma HLS BIND_STORAGE variable=fc_b    type=rom_1p impl=bram

    // --------------------------------------------------------------------
    // Buffers: also put these in BRAM RAM to avoid huge register/LUT usage.
    // --------------------------------------------------------------------
    data_t input_pad[VOICE_NUM_MFCC][VOICE_NUM_FRAMES + 2];
#pragma HLS BIND_STORAGE variable=input_pad type=ram_2p impl=bram

    data_t b1_out[VOICE_B1_CH][VOICE_B1_T];
#pragma HLS BIND_STORAGE variable=b1_out type=ram_2p impl=bram

    // padded b1 to remove boundary checks in conv2 (saves control/mux LUT)
    data_t b1_pad[VOICE_B1_CH][VOICE_B1_T + 2];
#pragma HLS BIND_STORAGE variable=b1_pad type=ram_2p impl=bram

    data_t b2_out[VOICE_B2_CH][VOICE_B2_T];
#pragma HLS BIND_STORAGE variable=b2_out type=ram_2p impl=bram

    // ============================================================
    // 1) Initialise padding and read input samples (Q8.8 int16 packed in AXIS)
    // input_pad[c][0] and input_pad[c][VOICE_NUM_FRAMES+1] = 0
    // ============================================================
    for (int c = 0; c < VOICE_NUM_MFCC; c++) {
#pragma HLS PIPELINE II=1
        input_pad[c][0] = (data_t)0;
        input_pad[c][VOICE_NUM_FRAMES + 1] = (data_t)0;
    }

    // Stream order expected: [t][c] (time-major)
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
#pragma HLS PIPELINE II=1
            axis_t pkt = in_stream.read();
            input_pad[c][t + 1] = q88_from_axis(pkt.data);
        }
    }

    // ============================================================
    // 2) Block1: Conv k=3 (40 -> VOICE_B1_CH) + ReLU + MaxPool2 => [VOICE_B1_CH, 25]
    // padding=1 handled via input_pad (no boundary checks)
    // ============================================================
    for (int o = 0; o < VOICE_B1_CH; o++) {
        for (int t = 0; t < VOICE_B1_T; t++) {
#pragma HLS PIPELINE II=4

            data_t max_val = (data_t)-128;

            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;   // 0..49
                int pad_t  = curr_t + 1;  // 1..50

                acc_t s = (acc_t)conv1_b[o];

                for (int i = 0; i < VOICE_NUM_MFCC; i++) {
                    // k=0..2 corresponds to offsets -1,0,+1
                    for (int k = 0; k < 3; k++) {
                        int w_idx = o * (VOICE_NUM_MFCC * 3) + i * 3 + k;
                        s += (acc_t)mul_dsp(input_pad[i][pad_t + (k - 1)], conv1_w[w_idx]);
                    }
                }

                data_t v = relu((data_t)s);
                if (v > max_val) max_val = v;
            }

            b1_out[o][t] = max_val;
        }
    }

    // ============================================================
    // 3) Build padded b1 for conv2 to remove boundary checks
    // b1_pad[c][0] = b1_pad[c][VOICE_B1_T+1] = 0, copy to [t+1]
    // ============================================================
    for (int c = 0; c < VOICE_B1_CH; c++) {
#pragma HLS PIPELINE II=1
        b1_pad[c][0] = (data_t)0;
        b1_pad[c][VOICE_B1_T + 1] = (data_t)0;
    }
    for (int c = 0; c < VOICE_B1_CH; c++) {
        for (int t = 0; t < VOICE_B1_T; t++) {
#pragma HLS PIPELINE II=1
            b1_pad[c][t + 1] = b1_out[c][t];
        }
    }

    // ============================================================
    // 4) Block2: Conv k=3 (VOICE_B1_CH -> VOICE_B2_CH) + ReLU => [VOICE_B2_CH, VOICE_B2_T]
    // VOICE_B2_T should be 25 for your PyTorch AdaptiveAvgPool1d(1) design.
    // Now no boundary checks because we use b1_pad.
    // ============================================================
    for (int o = 0; o < VOICE_B2_CH; o++) {
        for (int t = 0; t < VOICE_B2_T; t++) {
#pragma HLS PIPELINE II=4

            int pad_t = t + 1; // centre index in padded buffer
            acc_t s = (acc_t)conv2_b[o];

            for (int i = 0; i < VOICE_B1_CH; i++) {
                for (int k = 0; k < 3; k++) {
                    int w_idx = o * (VOICE_B1_CH * 3) + i * 3 + k;
                    s += (acc_t)mul_dsp(b1_pad[i][pad_t + (k - 1)], conv2_w[w_idx]);
                }
            }

            b2_out[o][t] = relu((data_t)s);
        }
    }

    // ============================================================
    // 5) Global average pool over T=VOICE_B2_T (25)
    // ============================================================
    data_t pooled[VOICE_B2_CH];
    const data_t invT = (data_t)(1.0f / VOICE_B2_T);

    for (int c = 0; c < VOICE_B2_CH; c++) {
        acc_t s = (acc_t)0;
        for (int t = 0; t < VOICE_B2_T; t++) {
#pragma HLS PIPELINE II=1
            s += (acc_t)b2_out[c][t];
        }
        pooled[c] = (data_t)(s * (acc_t)invT);
    }

    // ============================================================
    // 6) FC: [VOICE_B2_CH] -> [VOICE_NUM_CLASSES]
    // ============================================================
    data_t logits[VOICE_NUM_CLASSES];

    for (int c = 0; c < VOICE_NUM_CLASSES; c++) {
        acc_t s = (acc_t)fc_b[c];
        for (int i = 0; i < VOICE_B2_CH; i++) {
#pragma HLS PIPELINE II=1
            s += (acc_t)mul_dsp(pooled[i], fc_w[c * VOICE_B2_CH + i]);
        }
        logits[c] = (data_t)s;
    }

    // ============================================================
    // 7) Argmax + output
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
