"""Susun laporan markdown dari results/summary (angka terlacak, tanpa mengarang).

Jalankan setelah aggregate:
    python -m vehicle_lighting.report

Keluaran: docs/laporan/laporan.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config, resolve


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(data kosong)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                                       for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main():
    cfg = load_config()
    summary = resolve(cfg, cfg["results_dir"]) / "summary"
    if not summary.exists():
        raise SystemExit("Belum ada results/summary. Jalankan aggregate dahulu.")
    out_dir = resolve(cfg, "docs/laporan")
    out_dir.mkdir(parents=True, exist_ok=True)

    ms = pd.read_csv(summary / "mean_std.csv") if (summary / "mean_std.csv").exists() else pd.DataFrame()
    parts = ["# Laporan Hasil Eksperimen",
             "",
             "Topik: **Analisis Pengaruh Variasi Kondisi Pencahayaan terhadap Generalisasi "
             "Model MobileNetV2 dan ResNet50 pada Klasifikasi Jenis Kendaraan**.",
             "",
             "_Dokumen ini dibuat otomatis dari results/summary. Isi angka hanya dari hasil nyata._",
             "",
             "## 1. Performa per regime x model x kondisi uji (mean +/- std antar seed)",
             ""]
    if not ms.empty:
        view = ms.copy()
        for m in ["accuracy", "f1_macro"]:
            if f"{m}_mean" in view:
                view[f"{m}"] = view.apply(
                    lambda r: f"{r[f'{m}_mean']:.3f} +/- {r.get(f'{m}_std', 0):.3f}", axis=1)
        keep = [c for c in ["regime", "model", "test_lighting", "accuracy", "f1_macro", "n_seeds"]
                if c in view.columns]
        parts.append(_md_table(view[keep]))
    else:
        parts.append("_(mean_std.csv tidak ditemukan)_\n")

    drop = summary / "performance_drop.csv"
    parts += ["", "## 2. Penurunan performa vs kondisi normal", ""]
    parts.append(_md_table(pd.read_csv(drop)) if drop.exists() else "_(belum tersedia)_\n")

    eff = summary / "efficiency.csv"
    parts += ["", "## 3. Efisiensi model (parameter, ukuran, latensi CPU)", ""]
    parts.append(_md_table(pd.read_csv(eff)) if eff.exists() else "_(belum tersedia)_\n")

    mcn = summary / "mcnemar.csv"
    parts += ["", "## 4. Uji McNemar (MobileNetV2 vs ResNet50 pada test set sama)", ""]
    parts.append(_md_table(pd.read_csv(mcn)) if mcn.exists() else "_(belum tersedia)_\n")

    parts += ["", "## 5. Visualisasi",
              "Gambar tersimpan di `results/summary/` (semua angka di atas dapat ditelusuri ke `results/<run>/metrics.json`).",
              "",
              "| Prioritas | Komponen | Berkas gambar |",
              "|---|---|---|",
              "| 1 | Accuracy per kondisi | `accuracy_by_condition.png` |",
              "| 2 | Macro-F1 per kondisi | `f1_by_condition.png` |",
              "| 3 | Penurunan performa | `performance_drop.png` |",
              "| 4 | Confusion matrix | `confusion_<model>_<regime>_<kondisi>.png` |",
              "| 5 | Precision/Recall/F1 per kelas | `per_class_<model>_<regime>.png`, `per_class_metrics.csv` |",
              "| 6 | Waktu inferensi / efisiensi | `efficiency.png`, `efficiency.csv` |",
              "| 7 | Signifikansi (McNemar) | `mcnemar_<regime>.png`, `mcnemar.csv` |",
              ""]

    parts += ["", "## 6. Keterbatasan",
              "- Kondisi pencahayaan berasal dari atribut `timeofday` BDD100K, sehingga "
              "berpotensi terkonfound dengan kamera, lokasi, dan cuaca.",
              "- Split berbasis grup (video id) mencegah kebocoran, tetapi distribusi "
              "kondisi antar split dapat tidak persis sama.",
              "- Kesimpulan hanya berlaku untuk dataset dan protokol yang tercatat di "
              "`configs/experiment.yaml`.", ""]

    (out_dir / "laporan.md").write_text("\n".join(parts), encoding="utf-8")
    print("Laporan ->", out_dir / "laporan.md")


if __name__ == "__main__":
    main()
