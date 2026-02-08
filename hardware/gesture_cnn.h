#ifndef GESTURE_CNN_H
#define GESTURE_CNN_H

#include "typedefs.h"

// Top-Level Function
void gesture_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

#endif