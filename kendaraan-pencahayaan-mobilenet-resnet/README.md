# Pencahayaan x Generalisasi MobileNetV2 & ResNet50 (Klasifikasi Kendaraan)

Kerangka kode untuk penelitian:
*Analisis Pengaruh Variasi Kondisi Pencahayaan terhadap Generalisasi Model MobileNetV2 dan ResNet50 pada Klasifikasi Jenis Kendaraan.*

Dataset: **BDD100K** (`../bdd100k/bdd100k`). Environment: **conda**.

## 1. Setup conda
```bash
conda env create -f environment.yml
conda activate kendaraan-pencahayaan
python -m pytest -q tests          # semua uji harus lolos
```

## 2. Bangun dataset dari BDD100K
Crop kendaraan + kelompokkan kondisi pencahayaan asli (`timeofday`), lalu split berbasis grup video
(anti kebocoran):
```bash
python scripts/bdd100k_to_dataset.py --bdd-root ../bdd100k/bdd100k --per-class 3000 --seed 42
python scripts/check_lighting.py --n 4          # statistik luminansi + contact sheet
```
Keluaran: `data/manifest.csv`, `data/crops/`, `data/dataset_stats.json`,
`results/summary/luminance_stats.csv`, `results/summary/lighting_samples.png`.

## 3. Latih & evaluasi
```bash
# smoke test cepat (1 epoch, tanpa bobot pretrained)
python scripts/run_experiments.py --regimes normal --models mobilenet_v2 \
    --seeds 42 --epochs 1 --no-pretrained

# eksperimen utama: 2 model x 2 regime x 3 seed = 12 run
python scripts/run_experiments.py --regimes normal mixed

# eksperimen lanjutan (opsional, bila ada penurunan berarti)
python scripts/run_experiments.py --regimes normal_brightaug normal_gamma normal_clahe
```
Satu konfigurasi juga bisa dijalankan langsung:
```bash
python -m vehicle_lighting.train --model resnet50 --regime normal --seed 42 --save-ckpt
```

## 4. Ringkas hasil & visualisasi
```bash
python -m vehicle_lighting.aggregate     # results/summary/*.csv + *.png
python -m vehicle_lighting.report        # docs/laporan/laporan.md
```
`aggregate` menghasilkan tabel dan gambar berikut di `results/summary/`:

| Prioritas | Komponen | Berkas |
|---|---|---|
| 1 (Utama) | Accuracy per kondisi | `accuracy_by_condition.png`, kolom `accuracy_mean` di `mean_std.csv` |
| 2 (Utama) | Macro F1 per kondisi | `f1_by_condition.png`, kolom `f1_macro_mean` di `mean_std.csv` |
| 3 (Utama) | Performance drop dari normal | `performance_drop.png`, `performance_drop.csv` |
| 4 (Utama) | Confusion matrix | `confusion_<model>_<regime>_<kondisi>.png` (baris dinormalkan) |
| 5 (Pendukung) | Precision & recall per kelas | `per_class_<model>_<regime>.png`, `per_class_metrics.csv` |
| 6 (Tambahan) | Waktu inferensi / efisiensi | `efficiency.png`, `efficiency.csv` |
| 7 (Opsional) | Uji signifikansi (McNemar) | `mcnemar_<regime>.png`, `mcnemar.csv` |

## 5. Versi Jupyter Notebook (opsional)
Versi `.ipynb` dari seluruh pipeline ada di `notebooks/` dengan fungsi yang sama:
```bash
python -m ipykernel install --user --name kendaraan-pencahayaan --display-name "Python (kendaraan-pencahayaan)"
jupyter lab
```
Lihat `notebooks/README.md` untuk daftar notebook (`00`-`04`).

## Struktur
| Lokasi | Fungsi |
|---|---|
| `configs/experiment.yaml` | satu-satunya sumber parameter |
| `src/vehicle_lighting/` | lighting, data & split anti-kebocoran, model, train, aggregate, viz, report |
| `scripts/` | bdd100k_to_dataset, prepare_data, check_lighting, make_synthetic_dataset, run_experiments |
| `notebooks/` | versi Jupyter (.ipynb) dengan fungsi sama + generator `_build_notebooks.py` |
| `tests/` | pytest untuk pencahayaan, kebocoran data, dan pra-proses BDD100K |
| `docs/referensi/` | 3 dokumen referensi (docx + md) dan catatan verifikasi |

## Skenario eksperimen
| Skenario | Latih | Uji | Regime |
|---|---|---|---|
| A | normal | normal | `normal` |
| B | normal | dim | `normal` |
| C | normal | low | `normal` |
| D | normal + dim + low | semua kondisi (uji terpisah) | `mixed` |

## Mode data
1. **BDD100K (default)** - `raw_layout: lighting_class`; jalankan `scripts/bdd100k_to_dataset.py`.
   `scripts/prepare_data.py` tidak dipakai pada mode ini.
2. **Generik `data/raw/<kelas>/*.jpg`** - ubah `raw_layout: class` lalu jalankan `scripts/prepare_data.py`
   (kondisi redup/rendah disintesis). Untuk smoke test: `python scripts/make_synthetic_dataset.py`.

## Peringatan
- Hasil penelitian hanya sah bila dari data nyata dan split berbasis grup.
- Cek manual seluruh sitasi (lihat `docs/referensi/README.md`).
