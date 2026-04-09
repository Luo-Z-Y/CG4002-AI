from __future__ import annotations

"""Training helpers for the deployed and experimental voice CNN variants."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


EXPECTED_HLS_STATE_NAMES = (
    "conv1.weight",
    "conv1.bias",
    "conv2.weight",
    "conv2.bias",
    "fc.weight",
    "fc.bias",
)

VOICE_MODEL_VARIANT_SPECS = {
    "deployed": {
        "conv1_channels": 16,
        "conv2_channels": 32,
        "hls_subdir": "voice",
    },
    "experimental": {
        "conv1_channels": 20,
        "conv2_channels": 40,
        "hls_subdir": "voice_new",
    },
}


class _VoiceCNNBase(nn.Module):
    """Shared 2-conv voice backbone used by both deployed and experimental variants."""

    def __init__(
        self,
        num_classes: int,
        conv1_channels: int,
        conv2_channels: int,
        dropout_p: float = 0.25,
        bn_momentum: float = 0.05,
        bn_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.conv1_channels = conv1_channels
        self.conv2_channels = conv2_channels
        self.conv1 = nn.Conv1d(40, conv1_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(dropout_p * 0.5)

        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_channels, eps=bn_eps, momentum=bn_momentum)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.drop2 = nn.Dropout(dropout_p)

        self.fc = nn.Linear(conv2_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.squeeze(-1)
        return self.fc(x)


class VoiceCNN(_VoiceCNNBase):
    """Training-time voice model that matches the current HLS deployment layout."""

    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.25,
        bn_momentum: float = 0.05,
        bn_eps: float = 1e-3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            conv1_channels=16,
            conv2_channels=32,
            dropout_p=dropout_p,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )


class VoiceCNNExperimental(_VoiceCNNBase):
    """Slightly wider 2-conv variant kept small enough for realistic FPGA porting."""

    def __init__(
        self,
        num_classes: int,
        dropout_p: float = 0.25,
        bn_momentum: float = 0.05,
        bn_eps: float = 1e-3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            conv1_channels=20,
            conv2_channels=40,
            dropout_p=dropout_p,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )


def build_voice_model(
    num_classes: int,
    variant: str = "deployed",
    dropout_p: float = 0.25,
    bn_momentum: float = 0.05,
    bn_eps: float = 1e-3,
) -> nn.Module:
    """Build a named voice-model variant."""

    if variant == "deployed":
        return VoiceCNN(
            num_classes,
            dropout_p=dropout_p,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )
    if variant == "experimental":
        return VoiceCNNExperimental(
            num_classes,
            dropout_p=dropout_p,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
        )
    raise ValueError(f"Unknown voice model variant: {variant}")


def voice_hls_subdir_for_variant(variant: str) -> str:
    """Return the HLS export subdirectory for a voice-model variant."""

    try:
        return VOICE_MODEL_VARIANT_SPECS[variant]["hls_subdir"]
    except KeyError as exc:
        raise ValueError(f"Unknown voice model variant: {variant}") from exc


def is_hls_shape_compatible_model(model: nn.Module) -> bool:
    """Return whether the model matches the current HLS voice IP channel sizes."""

    return (
        hasattr(model, "conv1")
        and hasattr(model, "conv2")
        and hasattr(model, "fc")
        and tuple(model.conv1.weight.shape) == (16, 40, 3)
        and tuple(model.conv2.weight.shape) == (32, 16, 3)
        and model.fc.in_features == 32
    )


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup to one batch."""

    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample(()).item())
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


class FocalCrossEntropyLoss(nn.Module):
    """Cross-entropy with focal weighting for harder classes/examples."""

    def __init__(
        self,
        gamma: float = 1.5,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone())
        else:
            self.weight = None
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = torch.pow(1.0 - pt, self.gamma) * ce
        return loss.mean()


def _fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight.detach().cpu()
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, dtype=weight.dtype)
    else:
        bias = conv.bias.detach().cpu()

    gamma = bn.weight.detach().cpu()
    beta = bn.bias.detach().cpu()
    running_mean = bn.running_mean.detach().cpu()
    running_var = bn.running_var.detach().cpu()

    scale = gamma / torch.sqrt(running_var + bn.eps)
    fused_weight = weight * scale.view(-1, 1, 1)
    fused_bias = beta + (bias - running_mean) * scale
    return fused_weight.contiguous(), fused_bias.contiguous()


def build_hls_export_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Fuse BatchNorm into conv layers and return the 2-conv state layout."""

    was_training = model.training
    model.eval()
    try:
        conv1_w, conv1_b = _fuse_conv_bn(model.conv1, model.bn1)
        conv2_w, conv2_b = _fuse_conv_bn(model.conv2, model.bn2)
        state = {
            "conv1.weight": conv1_w,
            "conv1.bias": conv1_b,
            "conv2.weight": conv2_w,
            "conv2.bias": conv2_b,
            "fc.weight": model.fc.weight.detach().cpu().contiguous(),
            "fc.bias": model.fc.bias.detach().cpu().contiguous(),
        }
    finally:
        model.train(was_training)
    return state


def assert_hls_state_shapes(
    state_dict: Dict[str, torch.Tensor],
    num_classes: int,
    variant: str = "deployed",
) -> None:
    try:
        spec = VOICE_MODEL_VARIANT_SPECS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown voice model variant: {variant}") from exc

    conv1_channels = int(spec["conv1_channels"])
    conv2_channels = int(spec["conv2_channels"])
    expected_shapes = {
        "conv1.weight": (conv1_channels, 40, 3),
        "conv1.bias": (conv1_channels,),
        "conv2.weight": (conv2_channels, conv1_channels, 3),
        "conv2.bias": (conv2_channels,),
        "fc.weight": (num_classes, conv2_channels),
        "fc.bias": (num_classes,),
    }
    missing = sorted(set(expected_shapes) - set(state_dict))
    extra = sorted(set(state_dict) - set(expected_shapes))
    if missing or extra:
        raise RuntimeError(f"HLS state_dict mismatch. Missing={missing}, extra={extra}")
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(state_dict[name].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(f"HLS shape mismatch for {name}: expected {expected_shape}, got {actual_shape}")
