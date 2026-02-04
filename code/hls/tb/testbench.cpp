#include <iostream>
#include <iomanip>
#include "../src/typedefs.h"

// Tell the testbench which functions to look for
void gesture_referee(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);
void voice_processor(hls::stream<axis_t> &audio_in, hls::stream<axis_t> &voice_out);

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "   POKEMON AI CORE: STANDALONE TESTBENCH    " << std::endl;
    std::cout << "============================================" << std::endl;

    // ---------------------------------------------------------
    // TEST 1: GESTURE RECOGNITION (IMU DATA)
    // ---------------------------------------------------------
    hls::stream<axis_t> g_in, g_out;
    std::cout << ">> Sending 720 IMU samples (Gesture Test)..." << std::endl;

    for (int i = 0; i < 720; i++) {
        axis_t packet;
        // Simulate a "Z-Axis Spike" for Move 1 (Thrust)
        // Indices 2, 8, 14... are Z-axis in a 6-axis flattened stream
        if (i % 6 == 2 && i > 300 && i < 400) {
            packet.data = (data_t)10.0; // Strong acceleration
        } else {
            packet.data = (data_t)0.1;  // Idle noise
        }
        packet.last = (i == 719);
        g_in.write(packet);
    }

    // Run the Gesture Engine
    gesture_referee(g_in, g_out);

    if (!g_out.empty()) {
        int move_id = (int)g_out.read().data;
        std::cout << ">> [GESTURE ENGINE] Detected Move ID: " << move_id << std::endl;
    }

    // ---------------------------------------------------------
    // TEST 2: VOICE RECOGNITION (MFCC DATA)
    // ---------------------------------------------------------
    hls::stream<axis_t> v_in, v_out;
    std::cout << "\n>> Sending 2000 MFCC coefficients (Voice Test)..." << std::endl;

    for (int i = 0; i < 2000; i++) {
        axis_t packet;
        // Simulate a "High Frequency" pattern for Pokemon 1 (Blastoise)
        if (i > 1500) packet.data = (data_t)1.5;
        else packet.data = (data_t)0.1;

        packet.last = (i == 1999);
        v_in.write(packet);
    }

    // Run the Voice Engine
    voice_processor(v_in, v_out);

    if (!v_out.empty()) {
        int pkmn_id = (int)v_out.read().data;
        std::cout << ">> [VOICE ENGINE] Detected Pokemon ID: " << pkmn_id << std::endl;
    }

    std::cout << "============================================" << std::endl;
    std::cout << "        SIMULATION COMPLETE SUCCESS         " << std::endl;
    std::cout << "============================================" << std::endl;

    return 0;
}
