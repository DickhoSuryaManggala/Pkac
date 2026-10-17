"""Jalankan seluruh matriks eksperimen: models x regimes x seeds (lewati yang sudah selesai).

  python scripts/run_experiments.py --regimes normal mixed          # eksperimen utama (A-D)
  python scripts/run_experiments.py --regimes normal_brightaug normal_gamma normal_clahe   # lanjutan
Opsi smoke test: --epochs 1 --seeds 42 --no-pretrained
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vehicle_lighting.config import load_config, resolve
from vehicle_lighting.train import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["normal", "mixed"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    models = a.models or cfg["models"]; seeds = a.seeds or cfg["project"]["seeds"]
    for reg in a.regimes:
        for m in models:
            for s in seeds:
                d = resolve(cfg, cfg["results_dir"]) / f"{m}__{reg}__seed{s}" / "metrics.json"
                if d.exists():
                    print("lewati (sudah ada):", d.parent.name); continue
                run(cfg, m, reg, s, a.epochs, False if a.no_pretrained else None)


if __name__ == "__main__":
    main()
