#include <iostream>
#include "gesture_cnn.h"
#include "gesture_tb_cases.h"

static int run_dataset_cases(double pass_threshold) {
    const int words_per_case = WINDOW_SIZE * NUM_SENSORS;
    int total = GESTURE_TB_NUM_CASES;
    int correct = 0;
    int protocol_failures = 0;

    for (int n = 0; n < GESTURE_TB_NUM_CASES; n++) {
        hls::stream<axis_t> in_stream("gesture_in_ds");
        hls::stream<axis_t> out_stream("gesture_out_ds");

        for (int i = 0; i < words_per_case; i++) {
            axis_t pkt;
            pkt.data = gesture_tb_input_q88[n][i];
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (i == words_per_case - 1) ? 1 : 0;
            in_stream.write(pkt);
        }

        gesture_cnn(in_stream, out_stream);

        if (out_stream.empty()) {
            protocol_failures++;
            continue;
        }

        axis_t out_pkt = out_stream.read();
        int pred = static_cast<int>(out_pkt.data);
        bool class_ok = (pred >= 0 && pred < NUM_CLASSES);
        bool tlast_ok = (out_pkt.last == 1);
        bool single_pkt_ok = out_stream.empty();

        if (!(class_ok && tlast_ok && single_pkt_ok)) {
            protocol_failures++;
            continue;
        }

        if (pred == gesture_tb_expected[n]) {
            correct++;
        }
    }

    const double acc = (total > 0) ? (100.0 * static_cast<double>(correct) / static_cast<double>(total)) : 0.0;
    const bool pass = (acc >= pass_threshold) && (protocol_failures == 0);

    std::cout << "[" << (pass ? "PASS" : "FAIL") << "] dataset_"
              << GESTURE_TB_NUM_CASES << "_samples"
              << " acc=" << acc << "% threshold=" << pass_threshold << "%"
              << " correct=" << correct << "/" << total
              << " protocol_failures=" << protocol_failures
              << "\n";

    return pass ? 0 : 1;
}

int main() {
    std::cout << "========================================\n";
    std::cout << " Gesture CNN HLS Testbench (Dataset Gate)\n";
    std::cout << "========================================\n";

    // Pass criterion: accuracy >= 70% and no protocol failures.
    const int failures = run_dataset_cases(70.0);

    if (failures == 0) {
        std::cout << "All test cases passed.\n";
        return 0;
    }
    std::cout << "Testbench failed.\n";
    return 1;
}
