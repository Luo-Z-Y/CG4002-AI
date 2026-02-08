#include <iostream>
#include <vector>
#include "gesture_cnn.h"

// Helper to convert float to bits (same as in the core)
unsigned int float_to_bits(float f) {
    union { unsigned int i; float f; } converter;
    converter.f = f;
    return converter.i;
}

int main() {
    // 1. Setup Streams
    hls::stream<axis_t> in_stream("input_stream");
    hls::stream<axis_t> out_stream("output_stream");

    std::cout << "========================================" << std::endl;
    std::cout << "   Gesture CNN HLS Testbench   " << std::endl;
    std::cout << "========================================" << std::endl;

    // 2. Generate Synthetic Data
    // We create a pattern: 60 time steps x 6 sensors
    // In a real test, you could load this from a .dat file
    std::cout << "[TB] Generating " << (WINDOW_SIZE * NUM_SENSORS) << " samples..." << std::endl;

    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int c = 0; c < NUM_SENSORS; c++) {
            
            // Generate dummy value (e.g., a sine wave pattern for visualization)
            // Or just 1.0f to verify data flow
            float val = (float)(t + c) * 0.01f;

            axis_t packet;
            packet.data = float_to_bits(val);
            packet.keep = 0xF;   // Keep all bytes valid
            packet.strb = 0xF;
            packet.user = 0;
            packet.id = 0;
            packet.dest = 0;
            
            // Important: TLAST is NOT handled by the input loop in the core,
            // but DMA usually asserts it on the very last sample. 
            // Our core logic uses fixed loop counts, so it ignores input TLAST,
            // but we set it here for protocol correctness.
            packet.last = (t == WINDOW_SIZE - 1 && c == NUM_SENSORS - 1) ? 1 : 0;

            in_stream.write(packet);
        }
    }

    // 3. Run the IP Core
    // This calls the C++ function exactly as the FPGA would
    std::cout << "[TB] Running Inference..." << std::endl;
    gesture_cnn(in_stream, out_stream);

    // 4. Verify Output
    if (out_stream.empty()) {
        std::cout << "❌ ERROR: Output stream is empty! Logic did not produce a result." << std::endl;
        return 1;
    }

    axis_t result_packet = out_stream.read();
    int predicted_class = (int)result_packet.data;
    bool last_bit = (result_packet.last == 1);

    // 5. Report Results
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Prediction Class ID : " << predicted_class << std::endl;
    std::cout << "TLAST Signal Check  : " << (last_bit ? "PASS" : "FAIL") << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    if (last_bit) {
        std::cout << "✅ Testbench Passed!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Testbench Failed (TLAST not asserted)" << std::endl;
        return 1;
    }
}