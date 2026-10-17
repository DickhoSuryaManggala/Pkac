"""Manifest, split berbasis grup (anti kebocoran data), varian pencahayaan, dan Dataset.

Dua alur dukungan:
  - BDD100K (dipakai): manifest ditulis oleh scripts/bdd100k_to_dataset.py,
    kondisi pencahayaan berasal dari atribut ``timeofday`` asli.
  - Generik data/raw/<kelas>/*.jpg: manifest dibangun build_manifest() dan
    kondisi redup/rendah disintesis oleh make_variants().
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from .config import resolve
from .lighting import apply_lighting, enhance

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIGHTINGS = ["normal", "dim", "low"]


def _list_images(d: Path):
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def _group_id(path: Path, label: str, regex: str | None) -> str:
    stem = path.stem
    if regex:
        m = re.match(regex, stem)
        if m:
            stem = m.group(1) if m.groups() else m.group(0)
    return f"{label}/{stem}"


def group_split(groups_by_class: dict[str, list[str]], fracs: dict, seed: int) -> dict[str, str]:
    """Bagi grup (bukan gambar) per kelas secara stratified -> {group_id: split}."""
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for label, groups in groups_by_class.items():
        gs = sorted(set(groups))
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(n * fracs["train"]))
        n_va = int(round(n * fracs["val"]))
        if n >= 3:  # pastikan tiap split terisi
            n_tr = min(n_tr, n - 2)
            n_va = max(1, min(n_va, n - n_tr - 1))
        for i, g in enumerate(gs):
            out[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


def build_manifest(cfg: dict, seed: int | None = None) -> pd.DataFrame:
    """Scan data mentah -> manifest (belum ada varian sintetis)."""
    d = cfg["data"]
    raw = resolve(cfg, d["raw_dir"])
    seed = cfg["project"]["seeds"][0] if seed is None else seed
    rows = []
    if d["raw_layout"] == "class":
        for c in d["classes"]:
            for p in _list_images(raw / c):
                rows.append(dict(source=str(p), label=c, lighting="normal", origin="real",
                                 group_id=_group_id(p, c, d["group_regex"])))
    elif d["raw_layout"] == "lighting_class":
        for lt in LIGHTINGS:
            for c in d["classes"]:
                for p in _list_images(raw / lt / c):
                    rows.append(dict(source=str(p), label=c, lighting=lt, origin="real",
                                     group_id=_group_id(p, c, d["group_regex"])))
    else:
        raise ValueError("raw_layout harus 'class' atau 'lighting_class'")
    if not rows:
        raise FileNotFoundError(f"Tidak ada citra di {raw}. Cek raw_dir/classes/raw_layout.")
    df = pd.DataFrame(rows)
    gbc = df.groupby("label")["group_id"].apply(list).to_dict()
    mp = group_split(gbc, d["split"], seed)
    df["split"] = df["group_id"].map(mp)
    return df


def make_variants(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Tulis semua citra lewat pipeline yang sama (re-encode JPEG kualitas sama).

    layout 'class'          : buat normal/dim/low sintetis dari tiap citra asli.
    layout 'lighting_class' : hanya re-encode citra asli (tanpa sintesis).
    """
    L = cfg["lighting"]
    out_root = resolve(cfg, cfg["data"]["processed_dir"])
    q = int(L["jpeg_quality"])
    rng = np.random.default_rng(cfg["project"]["seeds"][0])
    recs = []
    for r in df.itertuples():
        bgr = cv2.imread(r.source)
        if bgr is None:
            print("PERINGATAN: citra tidak terbaca, dilewati:", r.source)
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        levels = L["levels"] if cfg["data"]["raw_layout"] == "class" else {r.lighting: None}
        for lt, prm in levels.items():
            x = img if prm is None else apply_lighting(
                img, prm["exposure"], prm["gamma"], L.get("noise_sigma", 0.0), rng)
            name = f"{Path(r.source).stem}.jpg"
            dst = out_root / lt / r.label / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
            recs.append(dict(path=str(dst.relative_to(cfg["_root"])), source=r.source, label=r.label,
                             group_id=r.group_id, split=r.split, lighting=lt,
                             origin="real" if prm is None else "synthetic"))
    return pd.DataFrame(recs)


def check_leakage(df: pd.DataFrame) -> list[str]:
    """Kembalikan daftar masalah kebocoran (list kosong = aman).

    Aturan: satu ``group_id`` tidak boleh muncul di lebih dari satu split.
    """
    issues = []
    multi = df.groupby("group_id")["split"].nunique()
    bad = multi[multi > 1]
    if len(bad):
        issues.append(f"{len(bad)} group_id muncul di >1 split, contoh: {list(bad.index[:3])}")
    return issues


# ----------------------------------------------------------------------------- Dataset
class LightingDataset(torch.utils.data.Dataset):
    """Dataset citra kendaraan untuk satu split x sekumpulan kondisi pencahayaan.

    Kelas didefinisikan di tingkat modul (bukan di dalam fungsi) agar bisa
    di-pickle oleh DataLoader dengan num_workers > 0 di Windows (spawn).
    """

    def __init__(self, cfg, sub: pd.DataFrame, transform, label2idx: dict,
                 enhance_method: str | None = None, brightness_aug: bool = False):
        self.cfg = cfg
        self.df = sub.reset_index(drop=True)
        self.transform = transform
        self.label2idx = label2idx
        self.enhance_method = enhance_method
        self.brightness_aug = brightness_aug

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.cvtColor(cv2.imread(str(resolve(self.cfg, r.path))), cv2.COLOR_BGR2RGB)
        if self.brightness_aug:
            lo, hi = self.cfg["train"]["brightness_aug_range"]
            e = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            img = apply_lighting(img, exposure=e, gamma=1.0)
        img = enhance(img, self.enhance_method)
        return self.transform(Image.fromarray(img)), self.label2idx[r.label], i


def make_dataset(cfg, df, split, lightings, train, enhance_method=None, brightness_aug=False,
                 label2idx=None) -> LightingDataset:
    """Buat torch Dataset untuk kombinasi split x kondisi pencahayaan."""
    size = cfg["project"]["image_size"]
    tcfg = cfg["train"]
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        tf = [T.RandomResizedCrop(size, scale=tuple(tcfg["random_resized_crop_scale"]))]
        if tcfg.get("hflip", True):
            tf.append(T.RandomHorizontalFlip())
    else:
        tf = [T.Resize(int(size * 256 / 224)), T.CenterCrop(size)]
    transform = T.Compose(tf + [T.ToTensor(), norm])
    sub = df[(df.split == split) & (df.lighting.isin(lightings))].reset_index(drop=True)
    return LightingDataset(cfg, sub, transform, label2idx, enhance_method=enhance_method,
                           brightness_aug=bool(brightness_aug and train))
