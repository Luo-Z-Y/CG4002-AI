#ifndef VOICE_CNN_WEIGHTS_H
#define VOICE_CNN_WEIGHTS_H

#include "voice_typedefs.h"

// Placeholder weights for build/simulation sanity.
// Replace this file with notebook export after training.

// Block1 pointwise: [16, 40, 1] -> flattened [16*40]
static const data_t pw1_w[640] = {0};
static const data_t pw1_b[16]  = {0};

// Block1 depthwise: [16, 1, 3] -> flattened [16*3]
static const data_t dw1_w[48]  = {0};
static const data_t dw1_b[16]  = {0};

// Block2 pointwise: [32, 16, 1] -> flattened [32*16]
static const data_t pw2_w[512] = {0};
static const data_t pw2_b[32]  = {0};

// Block2 depthwise: [32, 1, 3] -> flattened [32*3]
static const data_t dw2_w[96]  = {0};
static const data_t dw2_b[32]  = {0};

// FC: [5, 32]
static const data_t fc_w[160]  = {0};
static const data_t fc_b[5]    = {0};

#endif
