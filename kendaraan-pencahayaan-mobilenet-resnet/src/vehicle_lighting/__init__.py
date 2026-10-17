"""Paket kode eksperimen.

Topik: Analisis Pengaruh Variasi Kondisi Pencahayaan terhadap Generalisasi
Model MobileNetV2 dan ResNet50 pada Klasifikasi Jenis Kendaraan.

Modul:
    config    - pemuatan configs/experiment.yaml (satu-satunya sumber parameter)
    lighting  - simulasi/manipulasi pencahayaan & peningkatan citra
    data      - manifest, split berbasis grup (anti kebocoran), Dataset PyTorch
    models    - pembuat MobileNetV2 & ResNet50 (transfer learning)
    train     - latih satu konfigurasi (model x regime x seed) + evaluasi
    aggregate - gabung metrics.json -> tabel ringkas, penurunan, McNemar, grafik
    report    - susun laporan markdown dari results/summary
"""

__version__ = "0.1.0"
