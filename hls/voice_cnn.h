#ifndef VOICE_CNN_H
#define VOICE_CNN_H

#include "voice_typedefs.h"

void voice_cnn(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream);

#endif
