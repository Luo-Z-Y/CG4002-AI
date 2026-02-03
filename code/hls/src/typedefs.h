#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <ap_fixed.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// 16-bit fixed point: 8 bits integer, 8 bits fractional
typedef ap_fixed<16, 8> data_t;

// AXI-Stream Object (Data + TLAST signal)
struct axis_t {
    data_t data;
    bool last;
};

#endif