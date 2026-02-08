#include "gesture_cnn.h"
#include "gesture_cnn_weights.h"

// Helper: ReLU Activation
data_t relu(data_t x) {
    return (x > 0) ? x : (data_t)0;
}

void gesture_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return

    // Internal buffers
    data_t input_buffer[NUM_SENSORS][WINDOW_SIZE];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1

    // ---------------------------------------------------------
    // 1. READ DATA (Stream -> Array)
    // ---------------------------------------------------------
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int c = 0; c < NUM_SENSORS; c++) {
            #pragma HLS PIPELINE II=1
            axis_t temp = in_stream.read();
            
            // Interpret raw float bits -> Cast to Fixed Point
            union { unsigned int i; float f; } converter;
            converter.i = temp.data;
            
            input_buffer[c][t] = (data_t)converter.f; 
        }
    }

    // ---------------------------------------------------------
    // 2. LAYER 1: CONV (6->16) + RELU + MAXPOOL (60->30)
    // ---------------------------------------------------------
    data_t layer1[16][30];
    #pragma HLS ARRAY_PARTITION variable=layer1 complete dim=1

    for (int o = 0; o < 16; o++) {
        for (int t = 0; t < 30; t++) {
            data_t max_val = -128; // Min value for Q8.8

            for (int p = 0; p < 2; p++) { // Pooling Window (Fused)
                int curr_t = t * 2 + p; 
                data_t sum = (data_t)conv1_b[o];

                for (int i = 0; i < 6; i++) {
                    for (int k = 0; k < 3; k++) { 
                        int in_t = curr_t + k - 1; // Padding 1
                        if (in_t >= 0 && in_t < WINDOW_SIZE) {
                            int w_idx = o*(6*3) + i*3 + k;
                            sum += input_buffer[i][in_t] * (data_t)conv1_w[w_idx];
                        }
                    }
                }
                data_t val = relu(sum);
                if (val > max_val) max_val = val;
            }
            layer1[o][t] = max_val;
        }
    }

    // ---------------------------------------------------------
    // 3. LAYER 2: CONV (16->32) + RELU + MAXPOOL (30->15)
    // ---------------------------------------------------------
    data_t layer2[32][15];
    #pragma HLS ARRAY_PARTITION variable=layer2 complete dim=1

    for (int o = 0; o < 32; o++) {
        for (int t = 0; t < 15; t++) {
            data_t max_val = -128;

            for (int p = 0; p < 2; p++) { 
                int curr_t = t * 2 + p;
                data_t sum = (data_t)conv2_b[o];

                for (int i = 0; i < 16; i++) {
                    for (int k = 0; k < 3; k++) {
                        int in_t = curr_t + k - 1;
                        if (in_t >= 0 && in_t < 30) {
                            int w_idx = o*(16*3) + i*3 + k;
                            sum += layer1[i][in_t] * (data_t)conv2_w[w_idx];
                        }
                    }
                }
                data_t val = relu(sum);
                if (val > max_val) max_val = val;
            }
            layer2[o][t] = max_val;
        }
    }

    // ---------------------------------------------------------
    // 4. DENSE LAYERS (Flatten -> FC1 -> FC2)
    // ---------------------------------------------------------
    // FC1
    data_t dense1[32];
    for (int d = 0; d < 32; d++) {
        data_t sum = (data_t)fc1_b[d];
        int flat_idx = 0;
        
        // Flatten Order: Channels first, then Time
        for (int c = 0; c < 32; c++) {
            for (int t = 0; t < 15; t++) {
                int w_idx = d*FLATTEN_SIZE + flat_idx;
                sum += layer2[c][t] * (data_t)fc1_w[w_idx];
                flat_idx++;
            }
        }
        dense1[d] = relu(sum);
    }

    // FC2 (Output)
    data_t final_scores[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) {
        data_t sum = (data_t)fc2_b[c];
        for (int d = 0; d < 32; d++) {
            int w_idx = c*32 + d;
            sum += dense1[d] * (data_t)fc2_w[w_idx];
        }
        final_scores[c] = sum;
    }

    // ---------------------------------------------------------
    // 5. OUTPUT RESULT
    // ---------------------------------------------------------
    int best_class = 0;
    data_t best_score = final_scores[0];

    for (int i = 1; i < NUM_CLASSES; i++) {
        if (final_scores[i] > best_score) {
            best_score = final_scores[i];
            best_class = i;
        }
    }

    // Write Result to Stream
    axis_t result_packet;
    result_packet.data = best_class;
    result_packet.keep = 0xF; 
    result_packet.strb = 0xF;
    result_packet.last = 1;   
    
    out_stream.write(result_packet);
}