"""Tahap 1-2: scan data mentah -> split berbasis grup -> buat varian pencahayaan -> data/manifest.csv.

  python scripts/prepare_data.py            # pakai configs/experiment.yaml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vehicle_lighting.config import load_config, resolve
from vehicle_lighting.data import build_manifest, check_leakage, make_variants


def main():
    cfg = load_config()
    base = build_manifest(cfg)
    print("Citra asli:", len(base), "| grup:", base.group_id.nunique())
    man = make_variants(cfg, base)
    issues = check_leakage(man)
    if issues:
        raise SystemExit("KEBOCORAN DATA: " + "; ".join(issues))
    dst = resolve(cfg, cfg["data"]["manifest"]); dst.parent.mkdir(parents=True, exist_ok=True)
    man.to_csv(dst, index=False)
    print("\nJumlah per split x kondisi:\n", man.pivot_table(index="split", columns="lighting", values="path", aggfunc="count"))
    print("\nJumlah per kelas x split (kondisi normal):\n",
          man[man.lighting == "normal"].pivot_table(index="label", columns="split", values="path", aggfunc="count", fill_value=0))
    print("\nManifest ->", dst)


if __name__ == "__main__":
    main()
