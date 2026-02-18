#include <iostream>
#include "gesture_cnn.h"

static unsigned int float_to_bits(float f) {
    union {
        unsigned int i;
        float f;
    } cvt;
    cvt.f = f;
    return cvt.i;
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
            packet.data = float_to_bits(val);
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

    return (class_ok && last_ok) ? 0 : 1;
}

