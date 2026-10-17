# Notebooks (versi Jupyter)

Versi `.ipynb` dari pipeline proyek. Semua notebook memanggil paket `vehicle_lighting`
yang sama, jadi fungsinya identik dengan versi CLI di `scripts/` dan `src/`.

| Notebook | Tahap | Padanan CLI |
|---|---|---|
| `00_pipeline_lengkap.ipynb` | End-to-end | semua perintah di bawah, berurutan |
| `01_dataset_bdd100k.ipynb` | Dataset & split | `python scripts/bdd100k_to_dataset.py` |
| `02_verifikasi_pencahayaan.ipynb` | Verifikasi luminansi | `python scripts/check_lighting.py` |
| `03_pelatihan.ipynb` | Latih & evaluasi model | `python scripts/run_experiments.py` |
| `04_evaluasi_visualisasi.ipynb` | Agregasi + 7 visualisasi + laporan | `aggregate` lalu `report` |

## Cara pakai

1. Aktifkan environment dan daftarkan kernel (sekali saja):
   ```powershell
   conda activate kendaraan-pencahayaan
   python -m ipykernel install --user --name kendaraan-pencahayaan --display-name "Python (kendaraan-pencahayaan)"
   ```
2. Jalankan Jupyter dari **root proyek** (bukan dari folder `notebooks/`), agar
   `configs/experiment.yaml` dan path relatif terbaca:
   ```powershell
   cd D:\pkac\kendaraan-pencahayaan-mobilenet-resnet
   jupyter lab
   ```
3. Buka notebook, pilih kernel **Python (kendaraan-pencahayaan)**, lalu jalankan sel
   dari atas ke bawah. Sel pertama selalu menata root proyek, `sys.path`, dan cwd
   secara otomatis.

## Catatan
- Ubah parameter hanya di `configs/experiment.yaml`; notebook membacanya dari sana.
- Notebook `03`/`00` melewati run yang sudah punya `metrics.json`, jadi aman diulang.
- Tanpa `pretrained=False`, bobot ImageNet diunduh sekali pada run pertama (butuh internet).
- Notebook dihasilkan oleh `_build_notebooks.py`; jalankan `python notebooks/_build_notebooks.py`
  untuk membangun ulang setelah mengubah templatnya.
