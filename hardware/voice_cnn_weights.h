#ifndef VOICE_CNN_WEIGHTS_H
#define VOICE_CNN_WEIGHTS_H

#include "voice_typedefs.h"

// Placeholder weights for build/simulation sanity.
// Replace this file with notebook export after training.

static const data_t conv1_w[1920] = {0};   // [16, 40, 3]
static const data_t conv1_b[16]   = {0};

static const data_t conv2_w[1536] = {0};   // [32, 16, 3]
static const data_t conv2_b[32]   = {0};

static const data_t fc_w[96]      = {0};   // [3, 32]
static const data_t fc_b[3]       = {0};

#endif
