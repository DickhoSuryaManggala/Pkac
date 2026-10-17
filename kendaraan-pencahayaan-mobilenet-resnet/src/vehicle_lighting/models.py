"""Pembuat model klasifikasi berbasis transfer learning.

Hanya dua arsitektur yang dibandingkan agar prosedur identik:
    - MobileNetV2 (bobot ImageNet)
    - ResNet50   (bobot ImageNet)
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("mobilenet_v2", "resnet50")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Bangun model dengan kepala (head) diganti sesuai jumlah kelas."""
    if name == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"model tidak dikenal: {name} (pilih dari {MODEL_NAMES})")
    return m


def head_parameters(model: nn.Module, name: str):
    """Parameter kepala klasifikasi (dipakai untuk warmup: backbone dibekukan)."""
    return model.classifier.parameters() if name == "mobilenet_v2" else model.fc.parameters()
