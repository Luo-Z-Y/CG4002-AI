#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// 16-bit Fixed Point: 8 bits for the whole number, 8 bits for decimals.
// This is the optimal "sweet spot" for Ultra96 DSP slices.
typedef ap_fixed<16, 8> data_t;

// AXI4-Stream Structure
// Data: The 16-bit number
// Last: A 1-bit signal that tells the FPGA "This is the end of the packet."
struct axis_t {
    data_t data;
    bool last;
};

#endif
