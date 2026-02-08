import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from models import GestureCNN, VoiceCNN

# 1. DATA GENERATOR (Improved for 4 Gesture classes and 3 Voice classes)
def get_synthetic_data(samples=100):
    # Gesture: 4 classes, 6 axes, 120 samples
    X_g = torch.randn(samples * 4, 6, 120) * 0.1
    y_g = torch.cat([torch.full((samples,), i) for i in range(4)]).long()
    for i in range(4):
        X_g[i*samples:(i+1)*samples, i % 6, :] += 2.0 

    # Voice: 3 classes, 40 MFCCs, 50 time steps
    X_v = torch.randn(samples * 3, 40, 50) * 0.1
    y_v = torch.cat([torch.full((samples,), i) for i in range(3)]).long()
    for i in range(3):
        X_v[i*samples:(i+1)*samples, i % 40, :] += 2.0 
    
    return X_g, y_g, X_v, y_v

# 2. TRAINING FUNCTION
def train_brain(model, X, y, epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for e in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print(f"Loss: {loss.item():.4f}")
    return model

# 3. EXPORT TO HLS HEADER (Safe Version)
def export_weights(g_model, v_model, filename=None):
    if filename is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        filename = os.path.join(repo_root, "hls", "src", "weights.h")

    # CREATE THE FOLDER IF IT DOESN'T EXIST
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    g_model.eval()
    v_model.eval()

    with open(filename, "w") as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n#include \"typedefs.h\"\n\n")

        # Export Gesture CNN Weights [8 Filters][6 Channels][3 Kernel]
        gw = g_model.conv1.weight.data.numpy()
        f.write("const data_t g_conv_w[8][6][3] = {\n")
        for filt in gw:
            f.write("    {")
            for chan in filt:
                vals = ", ".join([f"(data_t){v:.6f}" for v in chan])
                f.write(f"{{{vals}}},")
            f.write("},\n")
        f.write("};\n\n")

        # Export Voice CNN Weights [16 Filters][40 Channels][3 Kernel]
        vw = v_model.conv1.weight.data.numpy()
        f.write("const data_t v_conv_w[16][40][3] = {\n")
        for filt in vw:
            f.write("    {")
            for chan in filt:
                vals = ", ".join([f"(data_t){v:.6f}" for v in chan])
                f.write(f"{{{vals}}},")
            f.write("},\n")
        f.write("};\n\n#endif")

if __name__ == "__main__":
    print("🚀 Initializing Standalone AI Training...")
    X_g, y_g, X_v, y_v = get_synthetic_data()
    
    print("Training Gesture CNN...")
    g_brain = train_brain(GestureCNN(), X_g, y_g)
    
    print("Training Voice CNN...")
    v_brain = train_brain(VoiceCNN(), X_v, y_v)
    
    export_weights(g_brain, v_brain)
    print("✅ Weights successfully exported to hls/src/weights.h")