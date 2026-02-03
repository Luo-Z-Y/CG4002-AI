#include <iostream>
#include "../src/typedefs.h"

void gesture_referee(hls::stream<axis_t> &in, hls::stream<axis_t> &out);

int main() {
    hls::stream<axis_t> in, out;
    for(int i=0; i<720; i++) {
        axis_t p; p.data = 1.0; p.last = (i==719);
        in.write(p);
    }
    gesture_referee(in, out);
    if(!out.empty()) {
        std::cout << "SUCCESS: Gesture Core Produced Output: " << out.read().data << std::endl;
    }
    return 0;
}