#include <iostream>
#include "gesture_cnn.h"
#include "gesture_tb_cases.h"

static unsigned int q88_to_u32(float f) {
    int q = static_cast<int>(f * 256.0f + (f >= 0.0f ? 0.5f : -0.5f));
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    return static_cast<unsigned int>(static_cast<unsigned short>(q));
}

int main() {
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");

    std::cout << "========================================" << std::endl;
    std::cout << "   Gesture CNN HLS Testbench" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "[TB] Generating " << (WINDOW_SIZE * NUM_SENSORS) << " samples..." << std::endl;
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int c = 0; c < NUM_SENSORS; c++) {
            float val = static_cast<float>(t + c) * 0.01f;

            axis_t packet;
            packet.data = q88_to_u32(val);
            packet.keep = 0xF;
            packet.strb = 0xF;
            packet.last = (t == WINDOW_SIZE - 1 && c == NUM_SENSORS - 1) ? 1 : 0;
            in_stream.write(packet);
        }
    }

    std::cout << "[TB] Running Inference..." << std::endl;
    gesture_cnn(in_stream, out_stream);

    if (out_stream.empty()) {
        std::cout << "ERROR: output stream is empty" << std::endl;
        return 1;
    }

    axis_t result_packet = out_stream.read();
    int predicted_class = static_cast<int>(result_packet.data);
    bool last_ok = (result_packet.last == 1);
    bool class_ok = (predicted_class >= 0 && predicted_class < NUM_CLASSES);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Prediction Class ID : " << predicted_class << std::endl;
    std::cout << "Class Range Check   : " << (class_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "TLAST Signal Check  : " << (last_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    int failures = (class_ok && last_ok) ? 0 : 1;

    // Dataset-driven check: accuracy on sampled real test cases.
    const int words_per_case = WINDOW_SIZE * NUM_SENSORS;
    int total = GESTURE_TB_NUM_CASES;
    int correct = 0;
    int protocol_failures = 0;

    for (int n = 0; n < GESTURE_TB_NUM_CASES; n++) {
        hls::stream<axis_t> in_ds("input_stream_ds");
        hls::stream<axis_t> out_ds("output_stream_ds");

        for (int i = 0; i < words_per_case; i++) {
            axis_t pkt;
            pkt.data = gesture_tb_input_q88[n][i];
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (i == words_per_case - 1) ? 1 : 0;
            in_ds.write(pkt);
        }

        gesture_cnn(in_ds, out_ds);

        if (out_ds.empty()) {
            protocol_failures++;
            continue;
        }

        axis_t out_pkt = out_ds.read();
        int pred = static_cast<int>(out_pkt.data);
        bool pred_ok = (pred >= 0 && pred < NUM_CLASSES);
        bool tlast_ds_ok = (out_pkt.last == 1);
        bool single_pkt_ok = out_ds.empty();

        if (!(pred_ok && tlast_ds_ok && single_pkt_ok)) {
            protocol_failures++;
            continue;
        }

        if (pred == gesture_tb_expected[n]) {
            correct++;
        }
    }

    double acc = (total > 0) ? (100.0 * static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
    bool acc_ok = (acc >= 70.0);

    std::cout << "[TB] Dataset Accuracy     : " << acc << "% (" << correct << "/" << total << ")" << std::endl;
    std::cout << "[TB] Accuracy >= 70%      : " << (acc_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "[TB] Dataset Protocol Err : " << protocol_failures << std::endl;

    if (!acc_ok || protocol_failures > 0) {
        failures++;
    }

    return (failures == 0) ? 0 : 1;
}
