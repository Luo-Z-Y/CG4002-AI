#ifndef VOICE_TYPEDEFS_H
#define VOICE_TYPEDEFS_H

#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

// Voice model configuration
#define VOICE_NUM_MFCC 40
#define VOICE_NUM_FRAMES 50
#define VOICE_NUM_CLASSES 3

// Conv1: 40 -> 16, pool2 => time 25
#define VOICE_CONV1_OUT_CH 16
#define VOICE_CONV1_OUT_T 25

// Conv2: 16 -> 32, time 25, then global average pool
#define VOICE_CONV2_OUT_CH 32
#define VOICE_CONV2_OUT_T 25

// Fixed-point data type (Q8.8)
typedef ap_fixed<16, 8, AP_TRN, AP_SAT> data_t;

// 32-bit AXI stream
typedef ap_axiu<32, 0, 0, 0> axis_t;

#endif
