"""Manipulasi pencahayaan dan peningkatan citra.

Semua fungsi menerima citra uint8 RGB (H, W, 3) dan mengembalikan uint8 RGB
dengan ukuran yang sama.

Prinsip metodologis: pencahayaan diisolasi sebagai SATU-SATUNYA variabel.
Karena itu noise bawaan ditetapkan 0 secara default (lihat configs/experiment.yaml).
"""
from __future__ import annotations

import cv2
import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Rec.709


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Konversi sRGB -> ruang linear."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Konversi ruang linear -> sRGB."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def apply_lighting(img: np.ndarray, exposure: float = 1.0, gamma: float = 1.0,
                   noise_sigma: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulasikan pencahayaan lebih gelap pada ruang linear.

    exposure    : pengali cahaya (<1 = lebih gelap).
    gamma       : >1 menekan tonal tengah (efek low-key tambahan).
    noise_sigma : default 0. Jangan dinaikkan bila pencahayaan harus menjadi
                  satu-satunya variabel yang berubah.
    """
    lin = srgb_to_linear(img.astype(np.float32) / 255.0)
    lin = np.power(np.clip(lin * exposure, 0, 1), gamma)
    if noise_sigma > 0:
        rng = rng or np.random.default_rng()
        lin = lin + rng.normal(0, noise_sigma, lin.shape).astype(np.float32)
    out = linear_to_srgb(lin)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def mean_luminance(img: np.ndarray) -> float:
    """Rata-rata luminansi persepsual (0-1) citra uint8 RGB."""
    lin = srgb_to_linear(img.astype(np.float32) / 255.0)
    y = lin @ _LUMA
    return float(linear_to_srgb(np.array(y.mean())))


def auto_gamma(img: np.ndarray, target: float = 0.45, lo: float = 0.3, hi: float = 1.0) -> np.ndarray:
    """Gamma correction adaptif: pilih gamma agar rata-rata luminansi mendekati target."""
    m = float(np.clip(img.astype(np.float32).mean() / 255.0, 1e-3, 0.999))
    g = float(np.clip(np.log(target) / np.log(m), lo, hi))
    lut = (np.power(np.arange(256) / 255.0, g) * 255.0 + 0.5).astype(np.uint8)
    return cv2.LUT(img, lut)


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE pada kanal L (LAB) agar warna tidak bergeser."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def enhance(img: np.ndarray, method: str | None) -> np.ndarray:
    """Terapkan teknik peningkatan citra opsional (None = tanpa perubahan)."""
    if method in (None, "none"):
        return img
    if method == "gamma":
        return auto_gamma(img)
    if method == "clahe":
        return clahe_enhance(img)
    raise ValueError(f"enhance tidak dikenal: {method}")
