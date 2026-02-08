#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// --- CONFIGURATION ---
#define WINDOW_SIZE 60
#define NUM_SENSORS 6
#define NUM_CLASSES 6

// Flatten Size logic: (60 samples -> Pool -> 30 -> Pool -> 15) * 32 channels = 480
#define FLATTEN_SIZE 480 

// 1. Math Type: 16-bit Fixed Point with AUTOMATIC Saturation
// Q8.8 format (8 integer bits, 8 fractional bits)
// Range: -128.0 to +127.99
typedef ap_fixed<16, 8, AP_TRN, AP_SAT> data_t;

// 2. Interface Type: 32-bit AXI Stream
typedef ap_axiu<32, 0, 0, 0> axis_t;

#endif