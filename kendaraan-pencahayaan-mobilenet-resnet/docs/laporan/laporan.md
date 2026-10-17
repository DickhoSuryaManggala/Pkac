# Laporan Hasil Eksperimen

Topik: **Analisis Pengaruh Variasi Kondisi Pencahayaan terhadap Generalisasi Model MobileNetV2 dan ResNet50 pada Klasifikasi Jenis Kendaraan**.

_Dokumen ini dibuat otomatis dari results/summary. Isi angka hanya dari hasil nyata._

## 1. Performa per regime x model x kondisi uji (mean +/- std antar seed)

| regime | model | test_lighting | accuracy | f1_macro | n_seeds |
| --- | --- | --- | --- | --- | --- |
| mixed | mobilenet_v2 | dim | 0.688 +/- 0.007 | 0.701 +/- 0.009 | 3 |
| mixed | mobilenet_v2 | low | 0.726 +/- 0.006 | 0.727 +/- 0.007 | 3 |
| mixed | mobilenet_v2 | normal | 0.753 +/- 0.006 | 0.763 +/- 0.007 | 3 |
| mixed | resnet50 | dim | 0.734 +/- 0.016 | 0.739 +/- 0.005 | 3 |
| mixed | resnet50 | low | 0.766 +/- 0.007 | 0.765 +/- 0.005 | 3 |
| mixed | resnet50 | normal | 0.789 +/- 0.011 | 0.797 +/- 0.011 | 3 |
| normal | mobilenet_v2 | dim | 0.572 +/- 0.257 | 0.520 +/- 0.357 | 3 |
| normal | mobilenet_v2 | low | 0.485 +/- 0.149 | 0.418 +/- 0.259 | 3 |
| normal | mobilenet_v2 | normal | 0.577 +/- 0.273 | 0.531 +/- 0.370 | 3 |
| normal | resnet50 | dim | 0.721 +/- 0.012 | 0.720 +/- 0.008 | 3 |
| normal | resnet50 | low | 0.591 +/- 0.013 | 0.584 +/- 0.015 | 3 |
| normal | resnet50 | normal | 0.781 +/- 0.008 | 0.788 +/- 0.006 | 3 |


## 2. Penurunan performa vs kondisi normal

| regime | model | test_lighting | metric | drop_abs_mean | drop_abs_std | drop_rel_pct_mean | drop_rel_pct_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed | mobilenet_v2 | dim | accuracy | 0.0657 | 0.0133 | 8.7130 | 1.6936 |
| mixed | mobilenet_v2 | dim | f1_macro | 0.0621 | 0.0147 | 8.1310 | 1.8541 |
| mixed | mobilenet_v2 | low | accuracy | 0.0275 | 0.0117 | 3.6392 | 1.5237 |
| mixed | mobilenet_v2 | low | f1_macro | 0.0357 | 0.0125 | 4.6678 | 1.5983 |
| mixed | resnet50 | dim | accuracy | 0.0550 | 0.0102 | 6.9731 | 1.3213 |
| mixed | resnet50 | dim | f1_macro | 0.0574 | 0.0088 | 7.2011 | 1.0145 |
| mixed | resnet50 | low | accuracy | 0.0236 | 0.0169 | 2.9704 | 2.0853 |
| mixed | resnet50 | low | f1_macro | 0.0312 | 0.0154 | 3.8944 | 1.8662 |
| normal | mobilenet_v2 | dim | accuracy | 0.0050 | 0.0154 | -0.3353 | 3.8300 |
| normal | mobilenet_v2 | dim | f1_macro | 0.0108 | 0.0127 | 0.3871 | 3.5485 |
| normal | mobilenet_v2 | low | accuracy | 0.0922 | 0.1242 | 8.4147 | 24.0877 |
| normal | mobilenet_v2 | low | f1_macro | 0.1130 | 0.1115 | 10.9550 | 22.2887 |
| normal | resnet50 | dim | accuracy | 0.0606 | 0.0067 | 7.7636 | 0.8942 |
| normal | resnet50 | dim | f1_macro | 0.0677 | 0.0056 | 8.5962 | 0.7048 |
| normal | resnet50 | low | accuracy | 0.1901 | 0.0056 | 24.3375 | 0.9532 |
| normal | resnet50 | low | f1_macro | 0.2039 | 0.0116 | 25.8804 | 1.5658 |


## 3. Efisiensi model (parameter, ukuran, latensi CPU)

| model | params_m | size_mb | latency_ms_cpu |
| --- | --- | --- | --- |
| mobilenet_v2 | 2.2290 | 8.9160 | 23.2490 |
| resnet50 | 23.5162 | 94.0649 | 71.3595 |


## 4. Uji McNemar (MobileNetV2 vs ResNet50 pada test set sama)

| regime | seed | test_lighting | n | only_mobilenet_correct | only_resnet_correct | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| mixed | seed42 | dim | 222 | 19 | 30 | 0.1524 |
| mixed | seed42 | low | 495 | 42 | 63 | 0.0504 |
| mixed | seed42 | normal | 930 | 87 | 116 | 0.0491 |
| mixed | seed43 | dim | 222 | 23 | 28 | 0.5758 |
| mixed | seed43 | low | 495 | 34 | 52 | 0.0662 |
| mixed | seed43 | normal | 930 | 72 | 104 | 0.0192 |
| mixed | seed44 | dim | 222 | 14 | 29 | 0.0315 |
| mixed | seed44 | low | 495 | 39 | 59 | 0.0544 |
| mixed | seed44 | normal | 930 | 70 | 109 | 0.0044 |
| normal | seed42 | dim | 222 | 12 | 113 | 0.0000 |
| normal | seed42 | low | 495 | 53 | 191 | 0.0000 |
| normal | seed42 | normal | 930 | 39 | 523 | 0.0000 |
| normal | seed43 | dim | 222 | 26 | 23 | 0.7754 |
| normal | seed43 | low | 495 | 62 | 72 | 0.4370 |
| normal | seed43 | normal | 930 | 71 | 110 | 0.0046 |
| normal | seed44 | dim | 222 | 23 | 24 | 1.0000 |
| normal | seed44 | low | 495 | 62 | 72 | 0.4370 |
| normal | seed44 | normal | 930 | 61 | 108 | 0.0004 |


## 5. Visualisasi
Gambar tersimpan di `results/summary/` (semua angka di atas dapat ditelusuri ke `results/<run>/metrics.json`).

| Prioritas | Komponen | Berkas gambar |
|---|---|---|
| 1 | Accuracy per kondisi | `accuracy_by_condition.png` |
| 2 | Macro-F1 per kondisi | `f1_by_condition.png` |
| 3 | Penurunan performa | `performance_drop.png` |
| 4 | Confusion matrix | `confusion_<model>_<regime>_<kondisi>.png` |
| 5 | Precision/Recall/F1 per kelas | `per_class_<model>_<regime>.png`, `per_class_metrics.csv` |
| 6 | Waktu inferensi / efisiensi | `efficiency.png`, `efficiency.csv` |
| 7 | Signifikansi (McNemar) | `mcnemar_<regime>.png`, `mcnemar.csv` |


## 6. Keterbatasan
- Kondisi pencahayaan berasal dari atribut `timeofday` BDD100K, sehingga berpotensi terkonfound dengan kamera, lokasi, dan cuaca.
- Split berbasis grup (video id) mencegah kebocoran, tetapi distribusi kondisi antar split dapat tidak persis sama.
- Kesimpulan hanya berlaku untuk dataset dan protokol yang tercatat di `configs/experiment.yaml`.
