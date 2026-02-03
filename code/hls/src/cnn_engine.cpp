#include "typedefs.h"

// A Universal 1D-Convolution Layer with ReLU and Global Average Pooling
// This replaces the need for a massive, power-hungry Fully Connected layer.
void run_1d_cnn(
    data_t input[64][120],  // Max dimensions (will be adjusted by engine)
    const data_t weights[16][64][3], 
    data_t output_scores[16],
    int num_filters,
    int num_channels,
    int window_size
) {
    // 1. Convolution + ReLU + Accumulation for Global Average Pooling
    for (int f = 0; f < num_filters; f++) {
        data_t filter_sum = 0;
        
        for (int t = 1; t < window_size - 1; t++) {
            #pragma HLS PIPELINE II=1
            data_t conv_acc = 0;
            
            for (int c = 0; c < num_channels; c++) {
                #pragma HLS UNROLL factor=6 // Parallelize across sensor axes
                conv_acc += input[c][t-1] * weights[f][c][0];
                conv_acc += input[c][t]   * weights[f][c][1];
                conv_acc += input[c][t+1] * weights[f][c][2];
            }
            
            // ReLU Activation
            data_t relu_out = (conv_acc > 0) ? conv_acc : (data_t)0;
            filter_sum += relu_out;
        }
        
        // Global Average Pooling (Average the activations across time)
        output_scores[f] = filter_sum / (data_t)window_size;
    }
}