"""Dataset mainan untuk SMOKE TEST pipeline (BUKAN untuk hasil penelitian).

  python scripts/make_synthetic_dataset.py --per-class 24
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2, numpy as np
from vehicle_lighting.config import load_config, resolve


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-class", type=int, default=24); a = ap.parse_args()
    cfg = load_config(); raw = resolve(cfg, cfg["data"]["raw_dir"]); rng = np.random.default_rng(0)
    shapes = {c: i for i, c in enumerate(cfg["data"]["classes"])}
    for c, k in shapes.items():
        (raw / c).mkdir(parents=True, exist_ok=True)
        for j in range(a.per_class):
            img = np.full((240, 320, 3), rng.integers(90, 200), np.uint8)
            img += rng.integers(0, 25, img.shape, dtype=np.uint8)
            w, h = [(120, 60), (50, 90), (200, 80), (160, 110)][k % 4]
            x, y = int(rng.integers(10, 300 - w)), int(rng.integers(10, 220 - h))
            col = tuple(int(v) for v in rng.integers(30, 255, 3))
            cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
            cv2.imwrite(str(raw / c / f"{c}_{j:03d}.jpg"), img)
    print("Dataset mainan ->", raw)


if __name__ == "__main__":
    main()
