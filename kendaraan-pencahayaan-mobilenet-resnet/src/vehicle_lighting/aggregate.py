"""Gabungkan results/*/metrics.json -> tabel ringkas, penurunan performa, McNemar, grafik.

Jalankan:
    python -m vehicle_lighting.aggregate

Keluaran di results/summary/:
    all_runs.csv, mean_std.csv, performance_drop.csv, efficiency.csv,
    mcnemar.csv, f1_<regime>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import load_config, resolve
from . import viz


def load_all(res_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(res_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        for lt, t in m["test"].items():
            rows.append(dict(model=m["model"], regime=m["regime"], seed=m["seed"], test_lighting=lt,
                             accuracy=t["accuracy"], precision_macro=t["precision_macro"],
                             recall_macro=t["recall_macro"], f1_macro=t["f1_macro"], n=t["n"],
                             params_m=m["efficiency"]["params_m"], size_mb=m["efficiency"]["size_mb"],
                             latency_ms_cpu=m["efficiency"]["latency_ms_cpu"]))
    return pd.DataFrame(rows)


def mean_std(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["regime", "model", "test_lighting"])
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def drops(df: pd.DataFrame) -> pd.DataFrame:
    """Penurunan performa vs kondisi 'normal' pada seed yang sama (absolut & relatif)."""
    base = df[df.test_lighting == "normal"].set_index(["regime", "model", "seed"])
    rows = []
    for r in df[df.test_lighting != "normal"].itertuples():
        if (r.regime, r.model, r.seed) not in base.index:
            continue
        b = base.loc[(r.regime, r.model, r.seed)]
        for metric in ["accuracy", "f1_macro"]:
            d = b[metric] - getattr(r, metric)
            rows.append(dict(regime=r.regime, model=r.model, seed=r.seed, test_lighting=r.test_lighting,
                             metric=metric, drop_abs=d,
                             drop_rel_pct=100 * d / b[metric] if b[metric] else np.nan))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    out = (d.groupby(["regime", "model", "test_lighting", "metric"])[["drop_abs", "drop_rel_pct"]]
           .agg(["mean", "std"]))
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def mcnemar_table(res_dir: Path, models=("mobilenet_v2", "resnet50")) -> pd.DataFrame:
    """McNemar exact: MobileNetV2 vs ResNet50 pada test set yang sama (per regime, kondisi, seed)."""
    rows = []
    for fa in sorted(res_dir.glob(f"{models[0]}__*__seed*")):
        fb = res_dir / fa.name.replace(models[0], models[1], 1)
        if not (fa / "predictions.csv").exists() or not (fb / "predictions.csv").exists():
            continue
        a, b = pd.read_csv(fa / "predictions.csv"), pd.read_csv(fb / "predictions.csv")
        parts = fa.name.split("__")
        if len(parts) != 3:
            continue
        _, regime, seed = parts
        for lt, ga in a.groupby("lighting"):
            gb = b[b.lighting == lt]
            m = ga.merge(gb, on=["lighting", "path"], suffixes=("_a", "_b"))
            ca, cb = m.y_pred_a == m.y_true_a, m.y_pred_b == m.y_true_b
            n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
            rows.append(dict(regime=regime, seed=seed, test_lighting=lt, n=len(m),
                             only_mobilenet_correct=n01, only_resnet_correct=n10, p_value=p))
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    classes = cfg["data"]["classes"]
    res = resolve(cfg, cfg["results_dir"])
    out = res / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df = load_all(res)
    if df.empty:
        raise SystemExit("Belum ada results/*/metrics.json")
    df.to_csv(out / "all_runs.csv", index=False)
    ms = mean_std(df)
    ms.to_csv(out / "mean_std.csv", index=False)
    dr = drops(df)
    if len(dr):
        dr.to_csv(out / "performance_drop.csv", index=False)
    eff = df.groupby("model")[["params_m", "size_mb", "latency_ms_cpu"]].mean().reset_index()
    eff.to_csv(out / "efficiency.csv", index=False)
    mt = mcnemar_table(res)
    if len(mt):
        mt.to_csv(out / "mcnemar.csv", index=False)

    # ---- prioritas 1-3: akurasi, F1, penurunan performa ----
    viz.grouped_by_condition(ms, "accuracy", out, "accuracy_by_condition.png", ylabel="accuracy")
    viz.grouped_by_condition(ms, "f1_macro", out, "f1_by_condition.png", ylabel="macro-F1")
    viz.drop_chart(dr, out)
    # ---- prioritas 4-5: confusion matrix + precision/recall per kelas ----
    viz.confusion_heatmaps(res, out, classes, models=cfg["models"])
    viz.per_class_metrics(res, out, classes, models=cfg["models"])
    # ---- prioritas 6-7: efisiensi + signifikansi ----
    viz.efficiency_chart(eff, out)
    viz.mcnemar_chart(mt, out)

    print(ms.to_string(index=False))
    print("\nTersimpan di", out)


if __name__ == "__main__":
    main()
