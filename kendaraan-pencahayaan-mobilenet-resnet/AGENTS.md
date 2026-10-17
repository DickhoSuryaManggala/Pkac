# AGENTS.md - Proyek Pencahayaan x Generalisasi MobileNetV2 & ResNet50

Topik: **Analisis Pengaruh Variasi Kondisi Pencahayaan terhadap Generalisasi Model MobileNetV2 dan ResNet50 pada Klasifikasi Jenis Kendaraan**.

## Wajib
- Bahasa: Indonesia untuk percakapan dan dokumentasi; kode, nama variabel, dan komentar teknis boleh Inggris.
- Semua parameter hanya di `configs/experiment.yaml`. Catat tiap keputusan di `docs/log-keputusan.md`.
- Jangan pernah: membocorkan data uji ke latih/validasi, memilih checkpoint dengan data uji, menyetel satu model saja, mengarang angka atau sitasi.
- Jalankan `python -m pytest -q tests` setelah mengubah `src/`.
- Perintah panjang (training) jalankan di background dan laporkan ringkasannya, bukan seluruh log.
- Environment memakai **conda**: `conda activate kendaraan-pencahayaan`.

## Alur perintah
```
setup -> dataset -> training -> evaluation -> (lanjutan) -> report
```

## Cakupan
- Dataset: **BDD100K** (format deteksi) -> crop kendaraan, kondisi pencahayaan asli (`timeofday`).
- Model: MobileNetV2 dan ResNet50 (transfer learning, prosedur identik).
- Skenario: A (latih normal, uji normal), B (latih normal, uji dim), C (latih normal, uji low), D (latih campuran, uji semua).
- Metrik: accuracy, precision/recall/macro-F1, confusion matrix, penurunan performa, McNemar, efisiensi.
