#ifndef VOICE_TYPEDEFS_H
#define VOICE_TYPEDEFS_H

#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// Voice model configuration
#define VOICE_NUM_MFCC 40
#define VOICE_NUM_FRAMES 50
#define VOICE_NUM_CLASSES 3   // go, no, yes

// Block 1: conv 40->12 (k=3) + maxpool2 => 25
#define VOICE_B1_CH 12
#define VOICE_B1_T 25

// Block 2: conv 12->16 (k=3), keep temporal length 25
// Then AdaptiveAvgPool1d(1) in software is equivalent to global average over T=25.
#define VOICE_B2_CH 16
#define VOICE_B2_T 25

// Fixed-point data type (Q8.8)
typedef ap_fixed<16, 8, AP_TRN, AP_SAT> data_t;

// 32-bit AXI stream
typedef ap_axiu<32, 0, 0, 0> axis_t;

#endif
