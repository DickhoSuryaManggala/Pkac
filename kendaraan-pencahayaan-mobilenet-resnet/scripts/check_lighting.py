"""Verifikasi kondisi pencahayaan: statistik luminansi per kondisi + contact sheet contoh citra.

  python scripts/check_lighting.py --n 4
Keluaran: results/summary/luminance_stats.csv, results/summary/lighting_samples.png
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2, numpy as np, pandas as pd
from vehicle_lighting.config import load_config, resolve
from vehicle_lighting.lighting import mean_luminance


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4); a = ap.parse_args()
    cfg = load_config()
    df = pd.read_csv(resolve(cfg, cfg["data"]["manifest"]))
    out = resolve(cfg, cfg["results_dir"]) / "summary"; out.mkdir(parents=True, exist_ok=True)
    lum = []
    for r in df.itertuples():
        img = cv2.cvtColor(cv2.imread(str(resolve(cfg, r.path))), cv2.COLOR_BGR2RGB)
        lum.append(mean_luminance(img))
    df["luminance"] = lum
    st = df.groupby("lighting")["luminance"].describe()[["count", "mean", "std", "min", "max"]]
    st.to_csv(out / "luminance_stats.csv"); print(st)
    order = [l for l in ["normal", "dim", "low"] if l in set(df.lighting)]
    base = df[(df.lighting == "normal")].sample(min(a.n, (df.lighting == "normal").sum()), random_state=0)
    rows = []
    for r in base.itertuples():
        tiles = []
        for lt in order:
            m = df[(df.lighting == lt) & (df.group_id == r.group_id)]
            p = m.iloc[0].path if len(m) else r.path
            tiles.append(cv2.resize(cv2.imread(str(resolve(cfg, p))), (224, 224)))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / "lighting_samples.png"), np.vstack(rows))
    print("Kolom:", order, "->", out / "lighting_samples.png")


if __name__ == "__main__":
    main()
