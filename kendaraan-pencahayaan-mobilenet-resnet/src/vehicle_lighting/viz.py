"""Visualisasi hasil eksperimen (sesuai prioritas evaluasi).

Menghasilkan gambar di results/summary/:
    accuracy_by_condition.png   - Accuracy per kondisi (Utama, prioritas 1)
    f1_by_condition.png         - Macro F1 per kondisi (Utama, prioritas 2)
    performance_drop.png        - Penurunan performa vs normal (Utama, prioritas 3)
    confusion_*.png             - Confusion matrix (Utama, prioritas 4)
    per_class_*.png             - Precision/recall/F1 per kelas (Pendukung, prioritas 5)
    efficiency.png              - Waktu inferensi & ukuran model (Tambahan, prioritas 6)
    mcnemar_*.png               - Uji signifikansi statistik (Opsional, prioritas 7)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

COND_ORDER = ["normal", "dim", "low"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _conds(values):
    vals = set(values)
    return [c for c in COND_ORDER if c in vals]


# ---------------------------------------------------------------- 1 & 2: akurasi / F1
def grouped_by_condition(ms: pd.DataFrame, metric: str, out: Path, fname: str,
                         ylabel: str | None = None) -> None:
    """Bar grouped: sumbu-x = kondisi uji, warna = model, panel = regime latih."""
    if ms.empty:
        return
    plt = _plt()
    regimes = sorted(ms.regime.unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4), squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        g = ms[ms.regime == reg]
        conds = _conds(g.test_lighting)
        models = sorted(g.model.unique())
        w = 0.8 / max(1, len(models))
        for k, m in enumerate(models):
            gm = g[g.model == m].set_index("test_lighting").reindex(conds)
            ax.bar(np.arange(len(conds)) + (k - (len(models) - 1) / 2) * w,
                   gm[f"{metric}_mean"], w,
                   yerr=gm[f"{metric}_std"].fillna(0), capsize=3, label=m)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel or metric)
        ax.set_title(f"regime latih: {reg}")
        ax.legend()
    fig.suptitle(f"{ylabel or metric} per kondisi pencahayaan uji")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- 3: performance drop
def drop_chart(drop: pd.DataFrame, out: Path) -> None:
    """Penurunan performa (dim & low) dibanding kondisi normal, per model x metrik."""
    if drop.empty or "drop_abs_mean" not in drop.columns:
        return
    plt = _plt()
    metrics = [m for m in ["accuracy", "f1_macro"] if m in set(drop.metric)]
    regimes = sorted(drop.regime.unique())
    fig, axes = plt.subplots(len(metrics), len(regimes),
                             figsize=(5 * len(regimes), 3.6 * len(metrics)), squeeze=False)
    for i, metric in enumerate(metrics):
        for j, reg in enumerate(regimes):
            ax = axes[i][j]
            g = drop[(drop.metric == metric) & (drop.regime == reg)]
            conds = _conds(g.test_lighting)
            models = sorted(g.model.unique())
            w = 0.8 / max(1, len(models))
            for k, m in enumerate(models):
                gm = g[g.model == m].set_index("test_lighting").reindex(conds)
                ax.bar(np.arange(len(conds)) + (k - (len(models) - 1) / 2) * w,
                       gm.drop_abs_mean, w, yerr=gm.drop_abs_std.fillna(0), capsize=3, label=m)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels(conds)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel(f"drop {metric} (poin)")
            ax.set_title(f"{metric} | regime: {reg}")
            ax.legend()
    fig.suptitle("Penurunan performa vs kondisi normal (makin besar = makin rentan)")
    fig.tight_layout()
    fig.savefig(out / "performance_drop.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- 4: confusion matrix
def _confusion_sums(res_dir: Path, models) -> dict:
    acc: dict = defaultdict(lambda: None)
    for d in sorted(res_dir.glob("*__*__seed*")):
        parts = d.name.split("__")
        if len(parts) != 3:
            continue
        model, regime, _seed = parts
        if models and model not in models:
            continue
        for cf in sorted(d.glob("confusion_*.csv")):
            lt = cf.stem.replace("confusion_", "")
            cm = pd.read_csv(cf, index_col=0).values.astype(float)
            key = (model, regime, lt)
            acc[key] = cm if acc[key] is None else acc[key] + cm
    return acc


def confusion_heatmaps(res_dir: Path, out: Path, classes, models=None) -> None:
    """Heatmap confusion matrix (baris = aktual, kolom = prediksi), dinormalkan per baris."""
    sums = _confusion_sums(res_dir, models)
    if not sums:
        return
    plt = _plt()
    for (model, regime, lt), cm in sorted(sums.items()):
        total = cm.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.divide(cm, cm.sum(axis=1, keepdims=True),
                             out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) != 0)
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("prediksi")
        ax.set_ylabel("aktual")
        ax.set_title(f"{model} | regime {regime} | uji {lt} (n={int(total)})")
        for a in range(len(classes)):
            for b in range(len(classes)):
                ax.text(b, a, f"{norm[a, b]:.2f}\n({int(cm[a, b])})", ha="center", va="center",
                        fontsize=7, color="white" if norm[a, b] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall (baris dinormalkan)")
        fig.tight_layout()
        fig.savefig(out / f"confusion_{model}_{regime}_{lt}.png", dpi=200)
        plt.close(fig)


# ------------------------------------------------------- 5: precision/recall per kelas
def per_class_metrics(res_dir: Path, out: Path, classes, models=None) -> pd.DataFrame:
    """Tabel + grafik precision/recall/F1 per kelas, dirata-ratakan antar seed."""
    rows = []
    for d in sorted(res_dir.glob("*__*__seed*")):
        parts = d.name.split("__")
        if len(parts) != 3:
            continue
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        if models and m["model"] not in models:
            continue
        for lt, t in m["test"].items():
            for c in classes:
                rows.append(dict(
                    model=m["model"], regime=m["regime"], seed=m["seed"], test_lighting=lt,
                    cls=c, precision=t.get("per_class_precision", {}).get(c, np.nan),
                    recall=t.get("per_class_recall", {}).get(c, np.nan),
                    f1=t.get("per_class_f1", {}).get(c, np.nan)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (df.groupby(["model", "regime", "test_lighting", "cls"])[
        ["precision", "recall", "f1"]].mean().reset_index())
    agg.to_csv(out / "per_class_metrics.csv", index=False)

    plt = _plt()
    for (model, regime), g in agg.groupby(["model", "regime"]):
        conds = _conds(g.test_lighting)
        fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), squeeze=False)
        for ax, lt in zip(axes[0], conds):
            gg = g[g.test_lighting == lt].set_index("cls").reindex(classes)
            x = np.arange(len(classes))
            w = 0.26
            for k, col in enumerate(["precision", "recall", "f1"]):
                ax.bar(x + (k - 1) * w, gg[col].fillna(0), w, label=col)
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=20, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{model} | regime {regime} | uji {lt}")
            ax.legend()
        fig.suptitle("Precision / Recall / F1 per kelas kendaraan")
        fig.tight_layout()
        fig.savefig(out / f"per_class_{model}_{regime}.png", dpi=200)
        plt.close(fig)
    return agg


# ----------------------------------------------------------- 6: waktu inferensi/efisiensi
def efficiency_chart(eff: pd.DataFrame, out: Path) -> None:
    """Ukuran model (parameter, MB) dan latensi inferensi CPU per model."""
    if eff.empty:
        return
    plt = _plt()
    metrics = [("params_m", "Parameter (juta)"), ("size_mb", "Ukuran model (MB)"),
               ("latency_ms_cpu", "Latensi CPU (ms/citra)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, label) in zip(axes, metrics):
        if col not in eff.columns:
            continue
        ax.bar(eff.model, eff[col])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Efisiensi model (prioritas tambahan)")
    fig.tight_layout()
    fig.savefig(out / "efficiency.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------- 7: signifikansi (McNemar)
def mcnemar_chart(mt: pd.DataFrame, out: Path) -> None:
    """Heatmap p-value McNemar (regime x kondisi uji). p kecil = beda signifikan."""
    if mt.empty:
        return
    plt = _plt()
    for regime, g in mt.groupby("regime"):
        conds = _conds(g.test_lighting)
        pvals = [g[g.test_lighting == c].p_value.mean() for c in conds]
        fig, ax = plt.subplots(figsize=(1.6 * len(conds) + 2, 2.4))
        im = ax.imshow(np.array([pvals]), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_yticks([0])
        ax.set_yticklabels(["p-value"])
        ax.set_title(f"McNemar (MobileNetV2 vs ResNet50) | regime {regime}\n"
                     f"merah = p<0.05 (beda signifikan)")
        for b, p in enumerate(pvals):
            ax.text(b, 0, f"{p:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / f"mcnemar_{regime}.png", dpi=200)
        plt.close(fig)
