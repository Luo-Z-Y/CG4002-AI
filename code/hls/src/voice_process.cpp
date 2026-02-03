#include "typedefs.h"

void voice_processor(hls::stream<axis_t> &audio_in, hls::stream<axis_t> &voice_out) {
    #pragma HLS INTERFACE axis port=audio_in
    #pragma HLS INTERFACE axis port=voice_out
    #pragma HLS INTERFACE s_axilite port=return bundle=VOICE_CTRL

    data_t mfcc_buffer[2000]; // 40 MFCCs * 50 time steps

    // 1. INTAKE
    for (int i = 0; i < 2000; i++) {
        #pragma HLS PIPELINE II=1
        mfcc_buffer[i] = audio_in.read().data;
    }

    // 2. DUMMY CLASSIFIER
    data_t result_id = (mfcc_buffer[0] > 0.5) ? (data_t)1 : (data_t)0;

    // 3. OUTPUT
    axis_t res;
    res.data = result_id;
    res.last = 1;
    voice_out.write(res);
}