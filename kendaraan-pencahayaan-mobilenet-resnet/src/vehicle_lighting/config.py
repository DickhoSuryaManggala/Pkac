"""Pemuatan konfigurasi. Semua parameter eksperimen hanya berasal dari
configs/experiment.yaml; tidak boleh ada angka yang di-hardcode di modul lain.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/experiment.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Muat YAML konfigurasi dan sisipkan kunci ``_root`` (path proyek absolut)."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict, rel: str | Path) -> Path:
    """Ubah path relatif (terhadap root proyek) menjadi path absolut."""
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p
