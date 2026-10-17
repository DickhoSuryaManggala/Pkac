"""Latih satu konfigurasi (model x regime x seed) lalu evaluasi pada SEMUA kondisi uji.

Contoh:
    python -m vehicle_lighting.train --model resnet50 --regime normal --seed 42

Prinsip wajib:
  - Prosedur latih IDENTIK antar model (optimizer, jadwal, augmentasi, epoch).
  - Validasi HANYA memakai kondisi pencahayaan yang ada saat latih.
  - Checkpoint dipilih dari val, BUKAN dari kondisi uji (anti kebocoran seleksi).
  - Evaluasi akhir tanpa pelatihan ulang, pada setiap kondisi uji terpisah.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .config import load_config, resolve
from .data import check_leakage, make_dataset
from .models import build_model, head_parameters


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _freeze_backbone(model, name, frozen: bool) -> None:
    head = set(id(p) for p in head_parameters(model, name))
    for p in model.parameters():
        p.requires_grad = (id(p) in head) or (not frozen)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, idx = [], [], []
    for x, y, i in loader:
        out = model(x.to(device))
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
        idx.append(i.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idx)


def metrics_from(y, p, classes) -> dict:
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    labels = list(range(len(classes)))
    prc, rcc, f1c, _ = precision_recall_fscore_support(
        y, p, labels=labels, average=None, zero_division=0)
    return dict(n=int(len(y)), accuracy=float(accuracy_score(y, p)), precision_macro=float(pr),
                recall_macro=float(rc), f1_macro=float(f1),
                per_class_f1={c: float(v) for c, v in zip(classes, f1c)},
                per_class_precision={c: float(v) for c, v in zip(classes, prc)},
                per_class_recall={c: float(v) for c, v in zip(classes, rcc)})


def efficiency(model, size, cpu_runs=50) -> dict:
    """Ukur jumlah parameter, ukuran (MB float32), dan latensi inferensi CPU (ms)."""
    m = copy.deepcopy(model).cpu().eval()
    params = sum(p.numel() for p in m.parameters())
    size_mb = params * 4 / 1e6
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        t = time.perf_counter()
        for _ in range(cpu_runs):
            m(x)
    return dict(params_m=params / 1e6, size_mb=size_mb,
                latency_ms_cpu=(time.perf_counter() - t) / cpu_runs * 1000)


def run(cfg, model_name, regime_name, seed, epochs=None, pretrained=None, device=None,
        save_ckpt=False) -> dict:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T = cfg["train"]
    epochs = epochs or T["epochs"]
    pretrained = T["pretrained"] if pretrained is None else pretrained
    classes = cfg["data"]["classes"]
    label2idx = {c: i for i, c in enumerate(classes)}
    reg = cfg["regimes"][regime_name]
    tr_light = reg["train_lighting"]
    enh = reg.get("enhance")
    bAug = reg.get("brightness_aug", False)

    df = pd.read_csv(resolve(cfg, cfg["data"]["manifest"]))
    issues = check_leakage(df)
    if issues:
        raise RuntimeError("KEBOCORAN DATA: " + "; ".join(issues))

    mk = lambda split, lights, train, aug=False: make_dataset(
        cfg, df, split, lights, train, enh, aug, label2idx)
    ds_tr, ds_va = mk("train", tr_light, True, bAug), mk("val", tr_light, False)
    nw = cfg["project"]["num_workers"]
    dl = lambda ds, sh: DataLoader(ds, T["batch_size"], shuffle=sh, num_workers=nw)
    dl_tr, dl_va = dl(ds_tr, True), dl(ds_va, False)

    model = build_model(model_name, len(classes), pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best, best_state, best_ep, bad, hist = -1, None, 0, 0, []

    for ep in range(epochs):
        frozen = ep < T["warmup_frozen_epochs"]
        _freeze_backbone(model, model_name, frozen)
        model.train()
        if frozen:  # BatchNorm backbone tetap eval selama warmup
            for mod in model.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()
        tot = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        yv, pv, _ = predict(model, dl_va, device)
        f1v = metrics_from(yv, pv, classes)["f1_macro"]
        hist.append(dict(epoch=ep + 1, train_loss=tot / max(1, len(ds_tr)), val_f1_macro=f1v))
        print(f"[{model_name}|{regime_name}|s{seed}] ep{ep + 1}/{epochs} "
              f"loss={hist[-1]['train_loss']:.4f} val_f1={f1v:.4f}", flush=True)
        if f1v > best:
            best, best_ep, bad = f1v, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= T["patience"] and ep >= T["warmup_frozen_epochs"]:
                break
    model.load_state_dict(best_state)

    # ---- evaluasi: semua kondisi uji, tanpa pelatihan ulang ----
    out_dir = resolve(cfg, cfg["results_dir"]) / f"{model_name}__{regime_name}__seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = dict(model=model_name, regime=regime_name, seed=seed, train_lighting=tr_light,
               enhance=enh, brightness_aug=bAug, pretrained=pretrained, best_epoch=best_ep,
               n_train=len(ds_tr), n_val=len(ds_va), test={})
    preds = []
    for lt in cfg["test_lighting"]:
        ds = mk("test", [lt], False)
        if len(ds) == 0:
            continue
        y, p, i = predict(model, dl(ds, False), device)
        res["test"][lt] = metrics_from(y, p, classes)
        cm = confusion_matrix(y, p, labels=range(len(classes)))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / f"confusion_{lt}.csv")
        for k in range(len(y)):
            preds.append(dict(lighting=lt, path=ds.df.iloc[i[k]].path,
                              y_true=classes[y[k]], y_pred=classes[p[k]]))
    res["efficiency"] = efficiency(model, cfg["project"]["image_size"])
    pd.DataFrame(preds).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "_root"}, allow_unicode=True), encoding="utf-8")
    if save_ckpt:
        torch.save(best_state, out_dir / "best.pt")
    return res


def main():
    ap = argparse.ArgumentParser(description="Latih & evaluasi satu konfigurasi")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", required=True, choices=["mobilenet_v2", "resnet50"])
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true", help="hanya untuk smoke test tanpa internet")
    ap.add_argument("--save-ckpt", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = a.seed if a.seed is not None else cfg["project"]["seeds"][0]
    r = run(cfg, a.model, a.regime, seed, a.epochs, False if a.no_pretrained else None,
            save_ckpt=a.save_ckpt)
    print(json.dumps({k: v["f1_macro"] for k, v in r["test"].items()}, indent=2))


if __name__ == "__main__":
    main()
