# Log Keputusan

| Tanggal | Tahap | Keputusan | Alasan |
|---|---|---|---|
| 2026-10-12 | 0 | Environment memakai conda (`environment.yml`, env `kendaraan-pencahayaan`, Python 3.11). | Permintaan pengguna; memisahkan dependensi proyek. |
| 2026-10-12 | 1 | Dataset BDD100K dipakai langsung (bukan `data/raw/<kelas>`). | BDD100K berformat deteksi (bbox + atribut citra); perlu crop kendaraan. |
| 2026-10-12 | 1 | Kelas = mobil, sepeda_motor, bus, truk. Pemetaan: car->mobil, motor->sepeda_motor, bus->bus, truck->truk. | Sesuai 4 kelas pada dokumen referensi; kategori lain (bike, train) diabaikan. |
| 2026-10-12 | 1 | Kondisi pencahayaan dari atribut timeofday ASLI: daytime->normal, dawn/dusk->dim, night->low. | Menghindari kritik "pencahayaan sintetis". Dicatat sebagai keterbatasan (confound kamera/lokasi/cuaca). |
| 2026-10-12 | 1 | Split berbasis grup video id (prefix sebelum '-' pada nama berkas). | Frame dari video sama tidak boleh lintas split (anti-kebocoran). |
| 2026-10-12 | 1 | Maksimum 3000 crop/kelas; proporsi kondisi normal 55% / dim 15% / low 30%; min bbox 32 px; bbox truncated dibuang. | Menyeimbangkan kelas yang sangat timpang; menjaga kualitas crop. |
| 2026-10-12 | 3 | Validasi memakai kondisi pencahayaan latih saja; checkpoint terbaik dipilih dari val. | Mencegah kebocoran seleksi dari kondisi uji. |
| 2026-10-12 | 3 | Prosedur latih identik (AdamW, cosine, warmup 2 epoch freeze backbone, 3 seed). | Perbandingan adil antar model. |
| 2026-10-12 | 4 | Evaluasi mengikuti 7 prioritas: accuracy & macro-F1 per kondisi, performance drop, confusion matrix, precision/recall per kelas, waktu inferensi, McNemar. Metrik per kelas dicatat di metrics.json; gambar dibuat oleh `vehicle_lighting.viz`. | Sesuai permintaan agar hasil dapat divisualisasikan. |
| 2026-10-12 | 0-6 | Dibuat versi Jupyter notebook di `notebooks/` (00-04) yang memanggil paket yang sama; ditambah `viz.py` dan paket Jupyter (jupyterlab, ipykernel, nbformat). | Permintaan pengguna akan versi .ipynb dengan fungsi sama di folder berbeda. |
|  |  |  |  |
