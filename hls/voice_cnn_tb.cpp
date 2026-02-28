#include <iostream>
#include <string>
#include <cstdint>
#include "voice_cnn.h"
#include "voice_cnn_weights.h"
#include "voice_tb_cases.h"

typedef ap_fixed<40, 20, AP_TRN, AP_SAT> conv_acc_t;
typedef ap_fixed<48, 24, AP_TRN, AP_SAT> pool_acc_t;
typedef ap_fixed<32, 16, AP_TRN, AP_SAT> mul_acc_t;

static unsigned int q88_to_u32(float f) {
    int q = static_cast<int>(f * 256.0f + (f >= 0.0f ? 0.5f : -0.5f));
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    return static_cast<unsigned int>(static_cast<unsigned short>(q));
}

static inline data_t relu_ref(data_t x) {
    return (x > (data_t)0) ? x : (data_t)0;
}

static int voice_ref(const data_t in_tc[VOICE_NUM_FRAMES][VOICE_NUM_MFCC]) {
    data_t input_pad[VOICE_NUM_MFCC][VOICE_NUM_FRAMES + 2];
    data_t b1_pad[VOICE_B1_CH][VOICE_B1_T + 2];
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
                conv_acc_t acc = (conv_acc_t)conv1_b[o];
                for (int i = 0; i < VOICE_NUM_MFCC; i++) {
                    for (int k = 0; k < 3; k++) {
                        int w_idx = o * (VOICE_NUM_MFCC * 3) + i * 3 + k;
                        mul_acc_t prod = (mul_acc_t)input_pad[i][pad_t + (k - 1)] * (mul_acc_t)conv1_w[w_idx];
                        acc += (conv_acc_t)prod;
                    }
                }
                data_t v = relu_ref((data_t)acc);
                if (v > max_val) max_val = v;
            }
            b1_pad[o][t + 1] = max_val;
        }
    }

    const data_t invT = (data_t)(1.0f / VOICE_B2_T);
    for (int o = 0; o < VOICE_B2_CH; o++) {
        pool_acc_t sum_t = (pool_acc_t)0;
        for (int t = 0; t < VOICE_B2_T; t++) {
            int pad_t = t + 1;
            conv_acc_t acc = (conv_acc_t)conv2_b[o];
            for (int i = 0; i < VOICE_B1_CH; i++) {
                for (int k = 0; k < 3; k++) {
                    int w_idx = o * (VOICE_B1_CH * 3) + i * 3 + k;
                    mul_acc_t prod = (mul_acc_t)b1_pad[i][pad_t + (k - 1)] * (mul_acc_t)conv2_w[w_idx];
                    acc += (conv_acc_t)prod;
                }
            }
            data_t y = relu_ref((data_t)acc);
            sum_t += (pool_acc_t)y;
        }
        pooled[o] = (data_t)(sum_t * (pool_acc_t)invT);
    }

    for (int c = 0; c < VOICE_NUM_CLASSES; c++) {
        conv_acc_t acc = (conv_acc_t)fc_b[c];
        for (int i = 0; i < VOICE_B2_CH; i++) {
            mul_acc_t prod = (mul_acc_t)pooled[i] * (mul_acc_t)fc_w[c * VOICE_B2_CH + i];
            acc += (conv_acc_t)prod;
        }
        logits[c] = (data_t)acc;
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

    for (int n = 0; n < VOICE_TB_NUM_CASES; n++) {
        hls::stream<axis_t> in_stream("voice_in_ds");
        hls::stream<axis_t> out_stream("voice_out_ds");

        for (int i = 0; i < words_per_case; i++) {
            axis_t pkt;
            pkt.data = voice_tb_input_q88[n][i];
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (i == words_per_case - 1) ? 1 : 0;
            in_stream.write(pkt);
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

        if (pred == voice_tb_expected[n]) {
            correct++;
        }
    }

    const double acc = (total > 0) ? (100.0 * static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
    const bool pass = (acc >= pass_threshold) && (protocol_failures == 0);

    std::cout << "[" << (pass ? "PASS" : "FAIL") << "] dataset_"
              << VOICE_TB_NUM_CASES << "_samples"
              << " acc=" << acc << "% threshold=" << pass_threshold << "%"
              << " correct=" << correct << "/" << total
              << " protocol_failures=" << protocol_failures
              << "\n";

    return pass ? 0 : 1;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "   Voice CNN HLS Testbench (Robust)\n";
    std::cout << "========================================\n";

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
