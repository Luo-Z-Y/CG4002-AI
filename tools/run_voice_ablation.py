from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from voice_cnn_training import FocalCrossEntropyLoss, build_voice_model, mixup_batch


def augment_feature_np(
    feat: np.ndarray,
    noise_std: float = 0.03,
    gain_jitter: float = 0.15,
    max_time_shift: int = 4,
    time_mask_max: int = 8,
    freq_mask_max: int = 5,
    freq_tilt_std: float = 0.10,
) -> np.ndarray:
    x = feat.copy()
    x *= np.float32(np.random.uniform(1.0 - gain_jitter, 1.0 + gain_jitter))
    x += np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)

    if freq_tilt_std > 0:
        tilt = np.linspace(-1.0, 1.0, x.shape[0], dtype=np.float32)[:, None]
        x *= (1.0 + np.float32(np.random.normal(0.0, freq_tilt_std)) * tilt).astype(np.float32)

    if max_time_shift > 0:
        shift = int(np.random.randint(-max_time_shift, max_time_shift + 1))
        x = np.roll(x, shift=shift, axis=1)

    for _ in range(2):
        width = int(np.random.randint(0, time_mask_max + 1))
        if 0 < width < x.shape[1]:
            start = int(np.random.randint(0, x.shape[1] - width + 1))
            x[:, start:start + width] = 0.0

    for _ in range(2):
        width = int(np.random.randint(0, freq_mask_max + 1))
        if 0 < width < x.shape[0]:
            start = int(np.random.randint(0, x.shape[0] - width + 1))
            x[start:start + width, :] = 0.0

    return x.astype(np.float32)


class AugmentedFeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool = False) -> None:
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)
        self.train = train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.x[idx].copy()
        if self.train:
            feat = augment_feature_np(feat)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(int(self.y[idx]), dtype=torch.long)


@dataclass(frozen=True)
class AblationConfig:
    name: str
    dropout_p: float
    weight_decay: float
    early_stopping_patience: int


DEFAULT_CONFIGS = [
    AblationConfig("baseline", dropout_p=0.20, weight_decay=2e-3, early_stopping_patience=8),
    AblationConfig("low_wd", dropout_p=0.20, weight_decay=1e-3, early_stopping_patience=8),
    AblationConfig("high_patience", dropout_p=0.20, weight_decay=2e-3, early_stopping_patience=10),
    AblationConfig("low_dropout", dropout_p=0.15, weight_decay=2e-3, early_stopping_patience=8),
    AblationConfig("low_dropout_low_wd", dropout_p=0.15, weight_decay=1e-3, early_stopping_patience=8),
    AblationConfig("low_dropout_low_wd_high_patience", dropout_p=0.15, weight_decay=1e-3, early_stopping_patience=10),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused voice CNN ablation sweep.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("../data/audio/20260406_combined"),
        help="Voice artefact directory containing manifests/features.",
    )
    parser.add_argument(
        "--variant",
        choices=("deployed", "experimental"),
        default="experimental",
        help="Voice model variant to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--bn-momentum", type=float, default=0.05)
    parser.add_argument("--bn-eps", type=float, default=1e-3)
    parser.add_argument("--train-aug-multiplier", type=int, default=6)
    parser.add_argument("--dashboard-sample-weight", type=float, default=2.5)
    parser.add_argument("--old-source-sample-weight", type=float, default=1.25)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--loss-class-weight-power", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("../data/audio/20260406_combined/voice_ablation_results.csv"),
        help="CSV path for ablation results.",
    )
    return parser.parse_args()


def prepare_split(
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    manifest_df = pd.read_csv(artifact_dir / "voice_manifest.csv")
    train_df = pd.read_csv(artifact_dir / "voice_train_manifest.csv")
    val_df = pd.read_csv(artifact_dir / "voice_val_manifest.csv")
    test_df = pd.read_csv(artifact_dir / "voice_test_manifest.csv")
    x_base = np.load(artifact_dir / "voice_features.npy").astype(np.float32)
    label_payload = json.loads((artifact_dir / "voice_labels.json").read_text())
    labels = label_payload["labels"] if isinstance(label_payload, dict) else label_payload

    path_to_index = {path: idx for idx, path in enumerate(manifest_df["path"].tolist())}

    def lookup_indices(df: pd.DataFrame) -> np.ndarray:
        return np.asarray([path_to_index[path] for path in df["path"].tolist()], dtype=np.int64)

    train_idx = lookup_indices(train_df)
    val_idx = lookup_indices(val_df)
    test_idx = lookup_indices(test_df)

    return (
        train_df,
        val_df,
        test_df,
        x_base[train_idx],
        x_base[val_idx],
        x_base[test_idx],
        labels,
    )


def build_sample_weights(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    labels: list[str],
    class_weight_power: float,
    dashboard_sample_weight: float,
    old_source_sample_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_classes = len(labels)
    train_class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = np.power(train_class_counts.sum() / np.maximum(train_class_counts, 1.0), class_weight_power)
    class_weights = (class_weights / class_weights.mean()).astype(np.float32)

    hard_class_sample_boosts = {
        "Charizard": 1.2,
        "Greninja": 1.25,
        "Lugia": 1.1,
        "Mewtwo": 1.35,
        "Pikachu": 1.3,
    }
    class_sample_boosts = np.ones(num_classes, dtype=np.float32)
    for idx, label in enumerate(labels):
        class_sample_boosts[idx] = float(hard_class_sample_boosts.get(label, 1.0))

    train_is_dashboard = train_df["path"].astype(str).str.contains("/dashboard/data/", regex=False).to_numpy()
    train_is_old = (train_df["source"] == "old").to_numpy()
    train_source_weights = np.ones(len(train_df), dtype=np.float32)
    train_source_weights[train_is_dashboard] *= dashboard_sample_weight
    train_source_weights[train_is_old] *= old_source_sample_weight

    train_hard_class_weights = class_sample_boosts[y_train].astype(np.float32)
    train_sample_weights = (class_weights[y_train] * train_source_weights * train_hard_class_weights).astype(np.float32)
    return class_weights, train_sample_weights


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = torch.argmax(model(xb), dim=1)
            y_true.extend(yb.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())
    return 100.0 * accuracy_score(y_true, y_pred)


def run_one(
    config: AblationConfig,
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_val_raw: np.ndarray,
    y_val: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    device: torch.device,
) -> dict[str, float | int | str]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_mean = x_train_raw.mean(axis=(0, 2), keepdims=True)
    train_std = x_train_raw.std(axis=(0, 2), keepdims=True) + 1e-6
    x_train = ((x_train_raw - train_mean) / train_std).astype(np.float32)
    x_val = ((x_val_raw - train_mean) / train_std).astype(np.float32)
    x_test = ((x_test_raw - train_mean) / train_std).astype(np.float32)

    class_weights, train_sample_weights = build_sample_weights(
        train_df,
        y_train,
        labels,
        class_weight_power=args.class_weight_power,
        dashboard_sample_weight=args.dashboard_sample_weight,
        old_source_sample_weight=args.old_source_sample_weight,
    )

    train_sampler = WeightedRandomSampler(
        weights=torch.tensor(train_sample_weights, dtype=torch.double),
        num_samples=len(y_train) * args.train_aug_multiplier,
        replacement=True,
    )

    train_loader = DataLoader(
        AugmentedFeatureDataset(x_train, y_train, train=True),
        batch_size=args.batch_size,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.loss_class_weight_power > 0:
        loss_class_weights = np.power(class_weights, args.loss_class_weight_power).astype(np.float32)
        loss_class_weights = loss_class_weights / loss_class_weights.mean()
        criterion_weight = torch.tensor(loss_class_weights, dtype=torch.float32, device=device)
    else:
        criterion_weight = None

    model = build_voice_model(
        len(labels),
        variant=args.variant,
        dropout_p=config.dropout_p,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
    ).to(device)
    criterion = FocalCrossEntropyLoss(
        gamma=args.focal_gamma,
        weight=criterion_weight,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
    )

    best_val = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            xb_mix, y_a, y_b, lam = mixup_batch(xb, yb, args.mixup_alpha)
            out = model(xb_mix)
            loss = lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_acc = evaluate_loader(model, val_loader, device)
        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.early_stopping_patience is not None and epochs_without_improvement >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    test_acc = evaluate_loader(model, test_loader, device)
    return {
        "name": config.name,
        "variant": args.variant,
        "dropout_p": config.dropout_p,
        "weight_decay": config.weight_decay,
        "early_stopping_patience": config.early_stopping_patience,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val, 4),
        "test_acc": round(test_acc, 4),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train_df, val_df, test_df, x_train_raw, x_val_raw, x_test_raw, labels = prepare_split(artifact_dir)

    results = []
    print(f"Using device: {device}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Variant: {args.variant}")
    print(f"Train/val/test: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    for config in DEFAULT_CONFIGS:
        print(
            f"[RUN] {config.name}: "
            f"dropout={config.dropout_p}, weight_decay={config.weight_decay}, patience={config.early_stopping_patience}"
        )
        result = run_one(
            config,
            args,
            train_df=train_df,
            x_train_raw=x_train_raw,
            y_train=pd.read_csv(artifact_dir / "voice_train_manifest.csv")["label_id"].to_numpy(dtype=np.int64),
            x_val_raw=x_val_raw,
            y_val=pd.read_csv(artifact_dir / "voice_val_manifest.csv")["label_id"].to_numpy(dtype=np.int64),
            x_test_raw=x_test_raw,
            y_test=pd.read_csv(artifact_dir / "voice_test_manifest.csv")["label_id"].to_numpy(dtype=np.int64),
            labels=labels,
            device=device,
        )
        results.append(result)
        print(
            f"[DONE] {config.name}: best_val={result['best_val_acc']:.2f}, "
            f"test={result['test_acc']:.2f}, best_epoch={result['best_epoch']}"
        )

    results_df = pd.DataFrame(results).sort_values(["test_acc", "best_val_acc"], ascending=False)
    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print("\nAblation results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {out_csv}")


if __name__ == "__main__":
    main()
