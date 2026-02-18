import argparse
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 60
NUM_CLASSES = 6
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
AUGMENT_FACTOR = 10
SEED = 42

# Canonical class IDs used by your existing gesture pipeline
CLASS_ALIASES = {
    0: ["raise hand and hold", "raise_hand_hold", "raisehandhold"],
    1: ["shaking fist (3 shakes)", "shake_fist_3", "shaking_fist_3_shakes", "shake fist 3"],
    2: ["vertical chop", "vertical_chop"],
    3: ["circular stir", "circular_stir"],
    4: ["horizontal swing", "horizontal_swing"],
    5: ["punch (forward thrust)", "punch", "forward_thrust"],
}

ID_TO_LABEL = {
    0: "Raise hand and hold",
    1: "Shaking fist (3 shakes)",
    2: "Vertical chop",
    3: "Circular stir",
    4: "Horizontal Swing",
    5: "Punch (forward thrust)",
}


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = s.replace(".txt", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_label_id(filename: str):
    n = normalize_name(filename)
    for cid, aliases in CLASS_ALIASES.items():
        for a in aliases:
            if normalize_name(a) == n:
                return cid
    return None


def pick_latest_date_dir(root: Path) -> Path:
    date_dirs = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if re.fullmatch(r"\d{8}", d.name):
            try:
                dt = datetime.strptime(d.name, "%d%m%Y")
                date_dirs.append((dt, d))
            except ValueError:
                continue
    if not date_dirs:
        raise RuntimeError(f"No ddmmyyyy date folders found in {root}")
    date_dirs.sort(key=lambda x: x[0])
    return date_dirs[-1][1]


def read_lines_auto(path: Path):
    # New files can be UTF-16; old files may be UTF-8.
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except Exception:
            continue
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8", errors="ignore")
    return raw.splitlines(True)


def try_parse_6floats(line: str):
    line = line.replace("\x00", "").strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None
    try:
        vals = [float(x) for x in parts]
        return vals
    except ValueError:
        return None


def parse_recordings_from_txt(path: Path):
    lines = read_lines_auto(path)
    blocks = []
    current = []
    recording = False

    for raw in lines:
        line = raw.replace("\x00", "").strip()

        if "YPR(deg)-X" in line:
            if current:
                blocks.append(np.array(current, dtype=np.float32))
                current = []
            recording = True
            continue

        if not recording:
            continue

        vals = try_parse_6floats(line)
        if vals is not None:
            current.append(vals)
            continue

        # End markers
        low = line.lower()
        if ("collecting data" in low) or (line == ""):
            if current:
                blocks.append(np.array(current, dtype=np.float32))
                current = []
            recording = False

    if current:
        blocks.append(np.array(current, dtype=np.float32))

    # Keep only meaningful recordings
    blocks = [b for b in blocks if b.shape[0] >= 10 and b.shape[1] == 6]
    return blocks


def build_augmented_dataframe(date_dir: Path, augment_factor: int):
    txt_files = sorted(date_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No txt files in {date_dir}")

    raw_records = []
    skipped = []

    for fp in txt_files:
        cid = infer_label_id(fp.name)
        if cid is None:
            skipped.append(fp.name)
            continue

        blocks = parse_recordings_from_txt(fp)
        for b in blocks:
            raw_records.append((b, cid, ID_TO_LABEL[cid]))

    if not raw_records:
        raise RuntimeError("Parsed zero recordings. Check txt format and class mapping.")

    rows = []
    global_id = 0

    for raw, cid, label in raw_records:
        seq_len = raw.shape[0]
        if seq_len < 10:
            continue

        for a in range(augment_factor):
            x = raw.copy()
            if a > 0:
                x += np.random.normal(0, 0.05, x.shape).astype(np.float32)
                x *= np.float32(np.random.uniform(0.8, 1.2))
                x += np.random.uniform(-0.1, 0.1, size=(1, 6)).astype(np.float32)

            if a > 5 and seq_len > 20:
                c0 = np.random.randint(0, max(1, int(seq_len * 0.1)))
                c1 = seq_len - np.random.randint(0, max(1, int(seq_len * 0.1)))
                if c1 > c0 + 5:
                    x = x[c0:c1]

            r = signal.resample(x, WINDOW_SIZE).astype(np.float32)
            for t in range(WINDOW_SIZE):
                rows.append(
                    {
                        "measurement_id": global_id,
                        "sequence_id": t,
                        "label_id": cid,
                        "label": label,
                        "gyro_x": float(r[t, 0]),
                        "gyro_y": float(r[t, 1]),
                        "gyro_z": float(r[t, 2]),
                        "acc_x": float(r[t, 3]),
                        "acc_y": float(r[t, 4]),
                        "acc_z": float(r[t, 5]),
                    }
                )
            global_id += 1

    df = pd.DataFrame(rows)
    info = {
        "raw_recordings": len(raw_records),
        "augmented_windows": global_id,
        "skipped_files": skipped,
    }
    return df, info


def df_to_windows(df_part, ids, feature_cols, label_col):
    X_list, y_list = [], []
    for mid in ids:
        g = df_part[df_part["measurement_id"] == mid]
        if len(g) != WINDOW_SIZE:
            continue
        x = g[feature_cols].to_numpy(dtype=np.float32)  # [60,6]
        y = int(g.iloc[0][label_col])
        X_list.append(x)
        y_list.append(y)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


class GestureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten_size = 15 * 32
        self.fc1 = nn.Linear(self.flatten_size, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


def export_pytorch_weights(model, out_header: Path):
    params = model.to("cpu").state_dict()
    with open(out_header, "w") as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write('#include "typedefs.h"\n\n')
        total = 0
        for name, tensor in params.items():
            clean = name.replace(".", "_").replace("weight", "w").replace("bias", "b")
            data = tensor.numpy().flatten()
            total += len(data)
            f.write(f"// PyTorch Layer: {name} (Shape: {tuple(tensor.shape)})\n")
            f.write(f"static const data_t {clean}[{len(data)}] = {{\n")
            for i, v in enumerate(data):
                f.write(f"{float(v):.6f}")
                if i < len(data) - 1:
                    f.write(", ")
                if (i + 1) % 10 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")
        f.write("#endif // WEIGHTS_H\n")
    return total


def main():
    parser = argparse.ArgumentParser(description="Retrain gesture CNN on latest dated txt folder")
    parser.add_argument("--gesture-root", default="data/gesture")
    parser.add_argument("--date", default="latest", help="ddmmyyyy or 'latest'")
    parser.add_argument("--augment-factor", type=int, default=AUGMENT_FACTOR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo = Path(__file__).resolve().parents[2]
    gesture_root = repo / args.gesture_root

    if args.date == "latest":
        date_dir = pick_latest_date_dir(gesture_root)
    else:
        date_dir = gesture_root / args.date
        if not date_dir.exists():
            raise RuntimeError(f"Date folder not found: {date_dir}")

    print(f"Using gesture date folder: {date_dir}")

    df_aug, info = build_augmented_dataframe(date_dir, args.augment_factor)
    print("Raw recordings:", info["raw_recordings"])
    print("Augmented windows:", info["augmented_windows"])
    if info["skipped_files"]:
        print("Skipped unknown files:", info["skipped_files"])

    # Save merged augmented CSV for traceability and existing tooling compatibility
    out_aug_dated = repo / "data" / f"augmented_imudata_{date_dir.name}.csv"
    out_aug_main = repo / "data" / "augmented_imudata.csv"
    df_aug.to_csv(out_aug_dated, index=False)
    df_aug.to_csv(out_aug_main, index=False)
    print(f"Saved: {out_aug_dated}")
    print(f"Updated: {out_aug_main}")

    feature_cols = ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]

    # Valid windows
    df = df_aug.copy()
    df["sequence_id"] = pd.to_numeric(df["sequence_id"], errors="coerce")
    valid_ids, labels_per_id = [], []

    for mid, g in df.groupby("measurement_id", sort=False):
        if len(g) != WINDOW_SIZE:
            continue
        if g["sequence_id"].isna().any() or g["label_id"].isna().any():
            continue
        if g["sequence_id"].nunique() != WINDOW_SIZE:
            continue
        lid = int(g.iloc[0]["label_id"])
        if not (g["label_id"] == lid).all():
            continue
        valid_ids.append(mid)
        labels_per_id.append(lid)

    valid_ids = np.array(valid_ids)
    labels_per_id = np.array(labels_per_id)
    print("Valid windows:", len(valid_ids))

    train_ids, test_ids, _, _ = train_test_split(
        valid_ids,
        labels_per_id,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels_per_id,
    )

    df_train = df[df["measurement_id"].isin(train_ids)].copy()
    df_test = df[df["measurement_id"].isin(test_ids)].copy()
    df_train.sort_values(["measurement_id", "sequence_id"], inplace=True)
    df_test.sort_values(["measurement_id", "sequence_id"], inplace=True)

    out_train = repo / "data" / "augmented_imudata_train.csv"
    out_test = repo / "data" / "augmented_imudata_test.csv"
    df_train.to_csv(out_train, index=False)
    df_test.to_csv(out_test, index=False)

    X_train_raw, y_train = df_to_windows(df_train, train_ids, feature_cols, "label_id")
    X_test_raw, y_test = df_to_windows(df_test, test_ids, feature_cols, "label_id")

    # Fit scaler on TRAIN only
    flat_train = X_train_raw.reshape(-1, 6)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-6

    X_train = ((X_train_raw - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)).astype(np.float32)
    X_test = ((X_test_raw - mean.reshape(1, 1, 6)) / std.reshape(1, 1, 6)).astype(np.float32)

    # Convert to [N, C, T]
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    np.save(repo / "data" / "mean.npy", mean.astype(np.float32))
    np.save(repo / "data" / "std.npy", std.astype(np.float32))
    print("Updated scaler: data/mean.npy, data/std.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureCNN().to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train).to(device), torch.tensor(y_train, dtype=torch.long).to(device)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test).to(device), torch.tensor(y_test, dtype=torch.long).to(device)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {device} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            pred = out.argmax(dim=1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()

        tr_acc = 100.0 * correct / max(total, 1)

        model.eval()
        te_correct = 0
        te_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                pred = out.argmax(dim=1)
                te_total += yb.size(0)
                te_correct += (pred == yb).sum().item()

        te_acc = 100.0 * te_correct / max(te_total, 1)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} | Loss {run_loss/max(len(train_loader),1):.4f} | "
                f"Train {tr_acc:.1f}% | Test {te_acc:.1f}%"
            )

    # Export weights for hardware
    out_header_hw = repo / "hardware" / "gesture_cnn_weights.h"
    out_header_nb = repo / "notebooks" / "gesture_cnn_weights.h"
    total_params = export_pytorch_weights(model, out_header_hw)
    export_pytorch_weights(model, out_header_nb)

    print(f"Exported gesture weights ({total_params} params):")
    print(f"  {out_header_hw}")
    print(f"  {out_header_nb}")


if __name__ == "__main__":
    main()
