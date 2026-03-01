#include <iostream>
#include <string>
#include <cstdint>
#include <ap_int.h>
#include "voice_cnn.h"
#include "voice_cnn_weights.h"
#include "voice_tb_cases.h"

typedef ap_fixed<48, 20, AP_TRN, AP_SAT> acc_t;

static unsigned int q88_to_u32(float f) {
    int q = static_cast<int>(f * 256.0f + (f >= 0.0f ? 0.5f : -0.5f));
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    return static_cast<unsigned int>(static_cast<unsigned short>(q));
}

static inline data_t relu_ref(data_t x) {
    return (x > (data_t)0) ? x : (data_t)0;
}

static inline data_t q88_word_to_data(uint32_t w) {
    ap_int<16> raw = (ap_int<16>)(w & 0xFFFF);
    data_t v;
    v.range(15, 0) = raw;
    return v;
}

static int voice_ref(const data_t in_tc[VOICE_NUM_FRAMES][VOICE_NUM_MFCC]) {
    data_t input_pad[VOICE_NUM_MFCC][VOICE_NUM_FRAMES + 2];
    data_t b1_out[VOICE_B1_CH][VOICE_B1_T];
    data_t b1_pad[VOICE_B1_CH][VOICE_B1_T + 2];
    data_t b2_out[VOICE_B2_CH][VOICE_B2_T];
    data_t pooled[VOICE_B2_CH];
    data_t logits[VOICE_NUM_CLASSES];

    for (int c = 0; c < VOICE_NUM_MFCC; c++) {
        input_pad[c][0] = (data_t)0;
        input_pad[c][VOICE_NUM_FRAMES + 1] = (data_t)0;
    }
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            input_pad[c][t + 1] = in_tc[t][c];
        }
    }

    for (int c = 0; c < VOICE_B1_CH; c++) {
        b1_pad[c][0] = (data_t)0;
        b1_pad[c][VOICE_B1_T + 1] = (data_t)0;
    }

    for (int o = 0; o < VOICE_B1_CH; o++) {
        for (int t = 0; t < VOICE_B1_T; t++) {
            data_t max_val = (data_t)-128;
            for (int p = 0; p < 2; p++) {
                int curr_t = t * 2 + p;
                int pad_t = curr_t + 1;
                acc_t s = (acc_t)conv1_b[o];
                for (int i = 0; i < VOICE_NUM_MFCC; i++) {
                    for (int k = 0; k < 3; k++) {
                        int w_idx = o * (VOICE_NUM_MFCC * 3) + i * 3 + k;
                        s += (acc_t)input_pad[i][pad_t + (k - 1)] * (acc_t)conv1_w[w_idx];
                    }
                }
                data_t v = relu_ref((data_t)s);
                if (v > max_val) max_val = v;
            }
            b1_out[o][t] = max_val;
        }
    }

    for (int c = 0; c < VOICE_B1_CH; c++) {
        b1_pad[c][0] = (data_t)0;
        b1_pad[c][VOICE_B1_T + 1] = (data_t)0;
    }
    for (int c = 0; c < VOICE_B1_CH; c++) {
        for (int t = 0; t < VOICE_B1_T; t++) {
            b1_pad[c][t + 1] = b1_out[c][t];
        }
    }

    for (int o = 0; o < VOICE_B2_CH; o++) {
        for (int t = 0; t < VOICE_B2_T; t++) {
            int pad_t = t + 1;
            acc_t s = (acc_t)conv2_b[o];
            for (int i = 0; i < VOICE_B1_CH; i++) {
                for (int k = 0; k < 3; k++) {
                    int w_idx = o * (VOICE_B1_CH * 3) + i * 3 + k;
                    s += (acc_t)b1_pad[i][pad_t + (k - 1)] * (acc_t)conv2_w[w_idx];
                }
            }
            b2_out[o][t] = relu_ref((data_t)s);
        }
    }

    const data_t invT = (data_t)(1.0f / VOICE_B2_T);
    for (int c = 0; c < VOICE_B2_CH; c++) {
        acc_t s = (acc_t)0;
        for (int t = 0; t < VOICE_B2_T; t++) {
            s += (acc_t)b2_out[c][t];
        }
        pooled[c] = (data_t)(s * (acc_t)invT);
    }

    for (int c = 0; c < VOICE_NUM_CLASSES; c++) {
        acc_t s = (acc_t)fc_b[c];
        for (int i = 0; i < VOICE_B2_CH; i++) {
            s += (acc_t)pooled[i] * (acc_t)fc_w[c * VOICE_B2_CH + i];
        }
        logits[c] = (data_t)s;
    }

    int best_class = 0;
    data_t best_score = logits[0];
    for (int c = 1; c < VOICE_NUM_CLASSES; c++) {
        if (logits[c] > best_score) {
            best_score = logits[c];
            best_class = c;
        }
    }
    return best_class;
}

static int run_case(const std::string &name, data_t in_tc[VOICE_NUM_FRAMES][VOICE_NUM_MFCC]) {
    hls::stream<axis_t> in_stream("voice_in");
    hls::stream<axis_t> out_stream("voice_out");

    const int expected = voice_ref(in_tc);

    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            axis_t pkt;
            float v = static_cast<float>(in_tc[t][c]);
            pkt.data = q88_to_u32(v);
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (t == VOICE_NUM_FRAMES - 1 && c == VOICE_NUM_MFCC - 1) ? 1 : 0;
            in_stream.write(pkt);
        }
    }

    voice_cnn(in_stream, out_stream);

    if (out_stream.empty()) {
        std::cout << "[FAIL] " << name << ": no output packet\n";
        return 1;
    }

    axis_t out = out_stream.read();
    int pred = static_cast<int>(out.data);
    bool class_ok = (pred >= 0 && pred < VOICE_NUM_CLASSES);
    bool tlast_ok = (out.last == 1);
    bool single_pkt_ok = out_stream.empty();
    bool match_ref = (pred == expected);

    std::cout << "[" << (class_ok && tlast_ok && single_pkt_ok && match_ref ? "PASS" : "FAIL")
              << "] " << name
              << " pred=" << pred
              << " expected=" << expected
              << " class_ok=" << class_ok
              << " tlast_ok=" << tlast_ok
              << " single_pkt_ok=" << single_pkt_ok
              << "\n";

    return (class_ok && tlast_ok && single_pkt_ok && match_ref) ? 0 : 1;
}

static int run_dataset_cases(double pass_threshold) {
    const int words_per_case = VOICE_NUM_FRAMES * VOICE_NUM_MFCC;
    int total = VOICE_TB_NUM_CASES;
    int correct = 0;
    int protocol_failures = 0;
    int mismatches = 0;
    int ref_label_mismatch = 0;
    int ref_pred_mismatch = 0;
    const int verbose_failures = 20;
    int cm[VOICE_NUM_CLASSES][VOICE_NUM_CLASSES] = {};

    for (int n = 0; n < VOICE_TB_NUM_CASES; n++) {
        hls::stream<axis_t> in_stream("voice_in_ds");
        hls::stream<axis_t> out_stream("voice_out_ds");
        data_t in_tc[VOICE_NUM_FRAMES][VOICE_NUM_MFCC];

        for (int i = 0; i < words_per_case; i++) {
            axis_t pkt;
            pkt.data = voice_tb_input_q88[n][i];
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (i == words_per_case - 1) ? 1 : 0;
            in_stream.write(pkt);

            int t = i / VOICE_NUM_MFCC;
            int c = i % VOICE_NUM_MFCC;
            in_tc[t][c] = q88_word_to_data(voice_tb_input_q88[n][i]);
        }

        const int expected_ref = voice_ref(in_tc);
        const int expected_ds = voice_tb_expected[n];
        if (expected_ref != expected_ds) {
            ref_label_mismatch++;
            if (ref_label_mismatch <= verbose_failures) {
                std::cout << "[WARN] case " << n
                          << " dataset_label=" << expected_ds
                          << " voice_ref=" << expected_ref
                          << "\n";
            }
        }

        voice_cnn(in_stream, out_stream);

        if (out_stream.empty()) {
            protocol_failures++;
            continue;
        }

        axis_t out = out_stream.read();
        int pred = static_cast<int>(out.data);
        bool class_ok = (pred >= 0 && pred < VOICE_NUM_CLASSES);
        bool tlast_ok = (out.last == 1);
        bool single_pkt_ok = out_stream.empty();

        if (!(class_ok && tlast_ok && single_pkt_ok)) {
            protocol_failures++;
            continue;
        }

        if (expected_ds >= 0 && expected_ds < VOICE_NUM_CLASSES) {
            cm[expected_ds][pred] += 1;
        }

        if (pred == expected_ds) {
            correct++;
        } else {
            mismatches++;
            if (mismatches <= verbose_failures) {
                std::cout << "[MIS] case " << n
                          << " pred=" << pred
                          << " dataset_label=" << expected_ds
                          << " voice_ref=" << expected_ref
                          << "\n";
            }
        }

        if (pred != expected_ref) {
            ref_pred_mismatch++;
        }
    }

    const double acc = (total > 0) ? (100.0 * static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
    const bool pass = (acc >= pass_threshold) && (protocol_failures == 0);

    std::cout << "[" << (pass ? "PASS" : "FAIL") << "] dataset_"
              << VOICE_TB_NUM_CASES << "_samples"
              << " acc=" << acc << "% threshold=" << pass_threshold << "%"
              << " correct=" << correct << "/" << total
              << " protocol_failures=" << protocol_failures
              << " mismatches=" << mismatches
              << " ref_label_mismatch=" << ref_label_mismatch
              << " ref_pred_mismatch=" << ref_pred_mismatch
              << "\n";

    std::cout << "Confusion matrix [true][pred]:\n";
    for (int t = 0; t < VOICE_NUM_CLASSES; t++) {
        std::cout << "  true " << t << " : ";
        for (int p = 0; p < VOICE_NUM_CLASSES; p++) {
            std::cout << cm[t][p];
            if (p + 1 < VOICE_NUM_CLASSES) std::cout << " ";
        }
        std::cout << "\n";
    }

    return pass ? 0 : 1;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   Voice CNN HLS Testbench (Robust)\n";
    std::cout << "========================================\n";
    std::cout << "Weight fingerprint:\n";
    std::cout << "  conv1_w[0..2]=[" << (float)conv1_w[0] << ", " << (float)conv1_w[1] << ", " << (float)conv1_w[2] << "]\n";
    std::cout << "  conv2_w[0..2]=[" << (float)conv2_w[0] << ", " << (float)conv2_w[1] << ", " << (float)conv2_w[2] << "]\n";
    std::cout << "  fc_w[0..2]=[" << (float)fc_w[0] << ", " << (float)fc_w[1] << ", " << (float)fc_w[2] << "]\n";
    std::cout << "  fc_b[0..2]=[" << (float)fc_b[0] << ", " << (float)fc_b[1] << ", " << (float)fc_b[2] << "]\n";
    std::cout << "  first_labels=[" 
              << voice_tb_expected[0] << ", "
              << voice_tb_expected[1] << ", "
              << voice_tb_expected[2] << ", "
              << voice_tb_expected[3] << ", "
              << voice_tb_expected[4] << "]\n";

    int failures = 0;

    static data_t in_ramp[VOICE_NUM_FRAMES][VOICE_NUM_MFCC];
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            in_ramp[t][c] = (data_t)(0.01f * static_cast<float>(t + c));
        }
    }
    failures += run_case("ramp", in_ramp);

    static data_t in_zero[VOICE_NUM_FRAMES][VOICE_NUM_MFCC];
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            in_zero[t][c] = (data_t)0;
        }
    }
    failures += run_case("all_zero", in_zero);

    static data_t in_impulse[VOICE_NUM_FRAMES][VOICE_NUM_MFCC];
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            in_impulse[t][c] = (data_t)0;
        }
    }
    in_impulse[0][0] = (data_t)1.0f;
    in_impulse[VOICE_NUM_FRAMES / 2][VOICE_NUM_MFCC / 2] = (data_t)-1.0f;
    failures += run_case("impulse", in_impulse);

    static data_t in_lfsr[VOICE_NUM_FRAMES][VOICE_NUM_MFCC];
    uint32_t s = 0x13579BDFu;
    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            int q = static_cast<int>(s & 0xFFFF) - 32768; // full int16 range
            in_lfsr[t][c] = (data_t)(static_cast<float>(q) / 256.0f);
        }
    }
    failures += run_case("lfsr_fullscale", in_lfsr);

    // Dataset-driven criterion: pass when overall accuracy >= 70%.
    failures += run_dataset_cases(70.0);

    std::cout << "----------------------------------------\n";
    if (failures == 0) {
        std::cout << "All test cases passed.\n";
        return 0;
    }

    std::cout << "Testbench failed with " << failures << " failing case(s).\n";
    return 1;
}
