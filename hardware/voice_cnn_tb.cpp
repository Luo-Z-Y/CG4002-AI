#include <iostream>
#include "voice_cnn.h"

static unsigned int float_to_bits(float f) {
    union {
        unsigned int i;
        float f;
    } cvt;
    cvt.f = f;
    return cvt.i;
}

int main() {
    hls::stream<axis_t> in_stream("voice_in");
    hls::stream<axis_t> out_stream("voice_out");

    std::cout << "========================================\n";
    std::cout << "   Voice CNN HLS Testbench\n";
    std::cout << "========================================\n";

    for (int t = 0; t < VOICE_NUM_FRAMES; t++) {
        for (int c = 0; c < VOICE_NUM_MFCC; c++) {
            float v = 0.01f * static_cast<float>(t + c);
            axis_t pkt;
            pkt.data = float_to_bits(v);
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            pkt.last = (t == VOICE_NUM_FRAMES - 1 && c == VOICE_NUM_MFCC - 1) ? 1 : 0;
            in_stream.write(pkt);
        }
    }

    voice_cnn(in_stream, out_stream);

    if (out_stream.empty()) {
        std::cout << "ERROR: no output produced\n";
        return 1;
    }

    axis_t out = out_stream.read();
    int pred = static_cast<int>(out.data);
    bool last_ok = (out.last == 1);
    bool class_ok = (pred >= 0 && pred < VOICE_NUM_CLASSES);

    std::cout << "Predicted class: " << pred << "\n";
    std::cout << "Class range check: " << (class_ok ? "PASS" : "FAIL") << "\n";
    std::cout << "TLAST check: " << (last_ok ? "PASS" : "FAIL") << "\n";

    return (class_ok && last_ok) ? 0 : 1;
}

