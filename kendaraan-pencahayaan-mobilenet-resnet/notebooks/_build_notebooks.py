"""Generator notebook Jupyter (.ipynb) untuk pipeline proyek ini.

Jalankan ulang bila ingin memperbarui notebook:
    python notebooks/_build_notebooks.py

Notebook yang dihasilkan memanggil paket `vehicle_lighting` yang sama (via sys.path),
jadi fungsinya identik dengan versi CLI di scripts/ dan src/.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

KERNEL = {"display_name": "Python (kendaraan-pencahayaan)", "language": "python",
          "name": "kendaraan-pencahayaan"}


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {"kernelspec": KERNEL,
                  "language_info": {"name": "python", "version": "3.11"}}
    return n


BOOTSTRAP = r'''
import os, sys, json
from pathlib import Path
from IPython.display import display, Image

# Temukan root proyek (folder berisi configs/experiment.yaml), lalu jadikan cwd.
_cands = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
ROOT = next(p for p in _cands if (p / "configs" / "experiment.yaml").exists())
os.chdir(ROOT)
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vehicle_lighting.config import load_config, resolve
cfg = load_config()
print("ROOT   :", ROOT)
print("config :", ROOT / "configs" / "experiment.yaml")
print("kelas  :", cfg["data"]["classes"])
print("model  :", cfg["models"])
'''.strip("\n")


nb01 = notebook([
    md("""
# 01 - Bangun Dataset dari BDD100K

Mengubah BDD100K (format deteksi) menjadi dataset klasifikasi jenis kendaraan:
crop bbox kendaraan, peta kategori -> kelas, kondisi pencahayaan asli dari `timeofday`,
dan split train/val/test berbasis grup video (anti-kebocoran data).

Keluaran: `data/manifest.csv`, `data/crops/`, `data/dataset_stats.json`.
"""),
    md("## 1. Setup"),
    code(BOOTSTRAP),
    md("## 2. Parameter dataset\nDiambil dari `configs/experiment.yaml`; ubah di sana, bukan di notebook."),
    code(r'''
params = dict(
    bdd_root=str(cfg["data"]["bdd100k"]["root"]),
    per_class=cfg["data"]["bdd100k"]["per_class"],
    min_box=cfg["data"]["bdd100k"]["min_box"],
    seed=cfg["project"]["seeds"][0],
    splits=["train", "val"],
)
params
'''),
    md("""## 3. Jalankan pra-proses

Menjalankan `scripts/bdd100k_to_dataset.py` (fungsi sama dengan versi CLI).
Pemindaian JSON bisa berlangsung lama. Untuk uji cepat, kecilkan `per_class`
dan batasi citra dengan `--limit-images`.
"""),
    code(r'''
import subprocess
cmd = [sys.executable, "scripts/bdd100k_to_dataset.py",
       "--bdd-root", params["bdd_root"],
       "--per-class", str(params["per_class"]),
       "--min-box", str(params["min_box"]),
       "--seed", str(params["seed"]),
       "--splits", *params["splits"]]
print(" ".join(cmd))
r = subprocess.run(cmd, text=True, capture_output=True)
print(r.stdout[-4000:])
if r.returncode:
    print("STDERR:\n", r.stderr[-2000:])
'''),
    md("## 4. Periksa manifest"),
    code(r'''
import pandas as pd
from vehicle_lighting.data import check_leakage

df = pd.read_csv(resolve(cfg, cfg["data"]["manifest"]))
print("total crop:", len(df), "| jumlah grup (video):", df.group_id.nunique())
print("kebocoran data:", check_leakage(df) or "aman")
display(df.head())
display(df.pivot_table(index="label", columns="split", values="path",
                       aggfunc="count", fill_value=0))
display(df.pivot_table(index="label", columns="lighting", values="path",
                       aggfunc="count", fill_value=0))
'''),
    md("""
**Gate tahap 1:** tiap kelas muncul di ketiga split dan tidak ada `group_id` yang
muncul di lebih dari satu split. Lanjut ke `02_verifikasi_pencahayaan.ipynb`.
"""),
])


nb02 = notebook([
    md("""
# 02 - Verifikasi Kondisi Pencahayaan

Mengukur luminansi rata-rata per kondisi (normal/dim/low) dan membuat contact sheet
contoh citra. Tujuannya memastikan urutan luminansi **normal > dim > low**.

Keluaran: `results/summary/luminance_stats.csv`, `results/summary/lighting_samples.png`.
"""),
    md("## 1. Setup"),
    code(BOOTSTRAP),
    md("## 2. Hitung luminansi per citra"),
    code(r'''
import cv2, numpy as np, pandas as pd
from vehicle_lighting.lighting import mean_luminance

df = pd.read_csv(resolve(cfg, cfg["data"]["manifest"]))
df["luminance"] = [
    mean_luminance(cv2.cvtColor(cv2.imread(str(resolve(cfg, p))), cv2.COLOR_BGR2RGB))
    for p in df["path"]
]
out = resolve(cfg, cfg["results_dir"]) / "summary"
out.mkdir(parents=True, exist_ok=True)
stats = df.groupby("lighting")["luminance"].describe()[["count", "mean", "std", "min", "max"]]
stats.to_csv(out / "luminance_stats.csv")
stats
'''),
    md("## 3. Contact sheet (kolom: normal, dim, low)"),
    code(r'''
order = [l for l in ["normal", "dim", "low"] if l in set(df.lighting)]
n = int(min(4, (df.lighting == "normal").sum()))
base = df[df.lighting == "normal"].sample(n, random_state=0)
rows = []
for r in base.itertuples():
    tiles = []
    for lt in order:
        m = df[(df.lighting == lt) & (df.group_id == r.group_id)]
        p = m.iloc[0].path if len(m) else r.path
        tiles.append(cv2.resize(cv2.imread(str(resolve(cfg, p))), (224, 224)))
    rows.append(np.hstack(tiles))
sheet = np.vstack(rows)
cv2.imwrite(str(out / "lighting_samples.png"), sheet)
Image(filename=str(out / "lighting_samples.png"))
'''),
    md("""
**Gate tahap 2:** `mean` luminansi `normal > dim > low`. Lanjut ke `03_pelatihan.ipynb`.
"""),
])


nb03 = notebook([
    md("""
# 03 - Pelatihan & Evaluasi Model

Fungsi `run()` melakukan: latih satu konfigurasi (model x regime x seed), pilih
checkpoint terbaik dari **val kondisi latih**, lalu evaluasi pada SEMUA kondisi uji
tanpa pelatihan ulang. Prosedur identik untuk MobileNetV2 dan ResNet50.
"""),
    md("## 1. Setup"),
    code(BOOTSTRAP),
    code(r'''
from vehicle_lighting.train import run
results_dir = resolve(cfg, cfg["results_dir"])
'''),
    md("""
## 2. Smoke test (1 epoch, tanpa bobot pretrained)

Pastikan pipeline berjalan sebelum menjalankan eksperimen penuh.
"""),
    code(r'''
_ = run(cfg, "mobilenet_v2", "normal", cfg["project"]["seeds"][0], epochs=1, pretrained=False)
'''),
    md("""
## 3. Eksperimen utama

2 model x 2 regime x 3 seed. Run yang sudah punya `metrics.json` akan dilewati,
sehingga cell ini aman dijalankan ulang bila terputus.
"""),
    code(r'''
REGIMES = ["normal", "mixed"]      # tambah normal_brightaug / normal_gamma / normal_clahe bila perlu
MODELS = cfg["models"]
SEEDS = cfg["project"]["seeds"]
EPOCHS = None                      # None = pakai nilai di config
NO_PRETRAINED = False

for reg in REGIMES:
    for m in MODELS:
        for s in SEEDS:
            done = results_dir / f"{m}__{reg}__seed{s}" / "metrics.json"
            if done.exists():
                print("lewati (sudah ada):", done.parent.name)
                continue
            print("==>", m, reg, "seed", s, flush=True)
            run(cfg, m, reg, s, EPOCHS, False if NO_PRETRAINED else None, save_ckpt=True)
'''),
    md("""
Catatan: tanpa `pretrained=False`, bobot ImageNet diunduh sekali pada run pertama
(butuh internet). Lanjut ke `04_evaluasi_visualisasi.ipynb`.
"""),
])


nb04 = notebook([
    md("""
# 04 - Evaluasi, Visualisasi, dan Laporan

Menghasilkan tabel ringkas, penurunan performa, confusion matrix, precision/recall
per kelas, efisiensi, dan uji McNemar, lalu menyusun laporan markdown.
"""),
    md("## 1. Setup"),
    code(BOOTSTRAP),
    md("## 2. Agregasi semua run -> `results/summary/`"),
    code(r'''
from vehicle_lighting import aggregate
aggregate.main()
'''),
    md("## 3. Tabel ringkas"),
    code(r'''
import pandas as pd
out = resolve(cfg, cfg["results_dir"]) / "summary"
for f in ["mean_std.csv", "performance_drop.csv", "per_class_metrics.csv",
          "efficiency.csv", "mcnemar.csv"]:
    p = out / f
    if p.exists():
        print("###", f)
        display(pd.read_csv(p).head(24))
'''),
    md("""
## 4. Visualisasi (tampil di notebook)

Prioritas 1-3: accuracy & macro-F1 per kondisi, penurunan performa.
Prioritas 4-5: confusion matrix, precision/recall per kelas.
Prioritas 6-7: efisiensi dan McNemar.
"""),
    code(r'''
for p in sorted(out.glob("*.png")):
    print(p.name)
    display(Image(filename=str(p)))
'''),
    md("## 5. Susun laporan markdown"),
    code(r'''
from vehicle_lighting import report
report.main()
lap = resolve(cfg, "docs/laporan/laporan.md")
print(lap.read_text(encoding="utf-8")[:2000])
'''),
])


nb00 = notebook([
    md("""
# 00 - Pipeline Lengkap (end-to-end)

Notebook ringkas yang menjalankan seluruh tahap secara berurutan. Untuk penjelasan
tiap tahap, gunakan notebook 01-04.
"""),
    md("## 1. Setup"),
    code(BOOTSTRAP),
    md("## 2. Bangun dataset (lewati bila `data/manifest.csv` sudah ada)"),
    code(r'''
manifest = resolve(cfg, cfg["data"]["manifest"])
if not manifest.exists():
    import subprocess
    cmd = [sys.executable, "scripts/bdd100k_to_dataset.py",
           "--bdd-root", str(cfg["data"]["bdd100k"]["root"]),
           "--per-class", str(cfg["data"]["bdd100k"]["per_class"]),
           "--seed", str(cfg["project"]["seeds"][0])]
    print(" ".join(cmd))
    r = subprocess.run(cmd, text=True, capture_output=True)
    print(r.stdout[-3000:])
else:
    print("manifest sudah ada:", manifest)
'''),
    md("## 3. Verifikasi pencahayaan"),
    code(r'''
import cv2, pandas as pd
from vehicle_lighting.lighting import mean_luminance
df = pd.read_csv(resolve(cfg, cfg["data"]["manifest"]))
df["luminance"] = [mean_luminance(cv2.cvtColor(cv2.imread(str(resolve(cfg, p))),
                                                cv2.COLOR_BGR2RGB)) for p in df["path"]]
print(df.groupby("lighting")["luminance"].mean())
'''),
    md("## 4. Latih + evaluasi (smoke test dulu, lalu matriks penuh)"),
    code(r'''
from vehicle_lighting.train import run
run(cfg, "mobilenet_v2", "normal", cfg["project"]["seeds"][0], epochs=1, pretrained=False)
'''),
    code(r'''
results_dir = resolve(cfg, cfg["results_dir"])
for reg in ["normal", "mixed"]:
    for m in cfg["models"]:
        for s in cfg["project"]["seeds"]:
            if (results_dir / f"{m}__{reg}__seed{s}" / "metrics.json").exists():
                print("lewati:", m, reg, s); continue
            run(cfg, m, reg, s, None, None, save_ckpt=True)
'''),
    md("## 5. Agregasi, visualisasi, laporan"),
    code(r'''
from vehicle_lighting import aggregate, report
aggregate.main()
report.main()
print((resolve(cfg, "docs/laporan/laporan.md")).read_text(encoding="utf-8")[:1200])
'''),
])


def main():
    files = {
        "00_pipeline_lengkap.ipynb": nb00,
        "01_dataset_bdd100k.ipynb": nb01,
        "02_verifikasi_pencahayaan.ipynb": nb02,
        "03_pelatihan.ipynb": nb03,
        "04_evaluasi_visualisasi.ipynb": nb04,
    }
    for name, nb in files.items():
        nbf.write(nb, str(HERE / name))
        print("ditulis:", name)


if __name__ == "__main__":
    main()
