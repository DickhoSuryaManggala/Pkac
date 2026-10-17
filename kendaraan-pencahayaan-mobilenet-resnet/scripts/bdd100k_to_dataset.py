"""BDD100K -> dataset klasifikasi jenis kendaraan dengan kondisi pencahayaan ASLI.

Skrip ini jembatan antara BDD100K (format deteksi: bbox + atribut citra) dan
kerangka klasifikasi di proyek ini. Hasilnya adalah `data/manifest.csv` dengan
kolom yang dipakai `vehicle_lighting.data.make_dataset`:

    path, label, lighting, group_id, split, origin

Pemetaan kelas (kategori BDD100K -> kelas proyek):
    car   -> mobil
    motor -> sepeda_motor
    bus   -> bus
    truck -> truk

Kondisi pencahayaan diambil dari atribut `timeofday` citra BDD100K (BUKAN sintetis):
    daytime   -> normal
    dawn/dusk -> dim
    night     -> low

Split train/val/test dibuat berbasis GRUP = video id (prefix sebelum tanda '-' pada
nama file, mis. `0000f77c-6257be58.jpg` -> grup `0000f77c`). Semua frame dari video
yang sama selalu berada di split yang sama, sehingga tidak ada kebocoran data.

Keluaran:
    data/crops/<kelas>/<image_stem>__<k>.jpg   (crop kendaraan)
    data/manifest.csv                          (path relatif terhadap root proyek)
    data/dataset_stats.json                    (ringkasan jumlah)

Contoh (dataset ada di ../bdd100k/bdd100k, yaitu D:/pkac/bdd100k/bdd100k):
    python scripts/bdd100k_to_dataset.py --bdd-root ../bdd100k/bdd100k --per-class 3000 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLASS_MAP = {"car": "mobil", "motor": "sepeda_motor", "bus": "bus", "truck": "truk"}
TOD_MAP = {"daytime": "normal", "dawn/dusk": "dim", "night": "low"}
CLASSES = ["mobil", "sepeda_motor", "bus", "truk"]
LIGHTINGS = ["normal", "dim", "low"]
# Sasaran proporsi tiap kondisi per kelas (dim dibuat kecil karena dawn/dusk langka).
TARGET_PROP = {"normal": 0.55, "dim": 0.15, "low": 0.30}
LABEL_FILES = {
    "train": "bdd100k_labels_images_train.json",
    "val": "bdd100k_labels_images_val.json",
}


def iter_records(json_path: Path):
    """Iterator per citra. Pakai ijson bila ada (hemat memori untuk JSON ratusan MB)."""
    try:
        import ijson  # type: ignore
    except ImportError:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        yield from data
        return
    with open(json_path, "rb") as f:
        yield from ijson.items(f, "item")


class Reservoir:
    """Reservoir sampling deterministik -> sampel acak seragam berukuran <= k."""

    def __init__(self, k: int, rng: random.Random):
        self.k = k
        self.rng = rng
        self.n = 0
        self.items: list = []

    def add(self, item):
        self.n += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randrange(self.n)
            if j < self.k:
                self.items[j] = item


def group_of(name: str) -> str:
    return name.split("-")[0]


def collect_candidates(bdd_root: Path, splits, per_class, seed, min_box,
                       drop_truncated, drop_occluded, limit_images):
    """Tahap 1: baca anotasi, kumpulkan kandidat crop per (kelas, pencahayaan)."""
    rng = random.Random(seed)
    buckets = defaultdict(lambda: Reservoir(per_class, rng))
    for split in splits:
        label_path = bdd_root / "labels" / LABEL_FILES[split]
        if not label_path.exists():
            print(f"PERINGATAN: label tidak ditemukan, dilewati: {label_path}")
            continue
        n = 0
        for rec in iter_records(label_path):
            n += 1
            if limit_images and n > limit_images:
                break
            name = rec.get("name")
            tod = (rec.get("attributes") or {}).get("timeofday")
            lighting = TOD_MAP.get(tod) if tod else None
            if not name or lighting is None:
                continue
            g = group_of(name)
            img_rel = f"images/100k/{split}/{name}"
            for lab in rec.get("labels", []):
                cls = CLASS_MAP.get(lab.get("category"))
                if cls is None:
                    continue
                box = lab.get("box2d")
                if not box:
                    continue
                attrs = lab.get("attributes") or {}
                if drop_truncated and attrs.get("truncated"):
                    continue
                if drop_occluded and attrs.get("occluded"):
                    continue
                x1, y1, x2, y2 = (box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2"))
                if None in (x1, y1, x2, y2):
                    continue
                if (x2 - x1) < min_box or (y2 - y1) < min_box:
                    continue
                buckets[(cls, lighting)].add((img_rel, g, float(x1), float(y1), float(x2), float(y2)))
        print(f"  {split}: {n} citra dipindai -> kandidat terkumpul")
    return buckets, rng


def select_final(buckets, per_class, rng):
    """Tahap 2: pilih <= per_class crop per kelas dengan proporsi kondisi terkontrol."""
    final = []
    for cls in CLASSES:
        pools = {}
        for lt in LIGHTINGS:
            items = list(buckets[(cls, lt)].items)
            rng.shuffle(items)
            pools[lt] = items
        chosen = []
        taken = {lt: 0 for lt in LIGHTINGS}
        for lt in ["dim", "normal", "low"]:  # dahulukan dim agar kondisi langka terwakili
            want = min(len(pools[lt]), int(round(per_class * TARGET_PROP[lt])))
            chosen += [(lt, c) for c in pools[lt][:want]]
            taken[lt] = want
        remaining = per_class - len(chosen)
        for lt in ["normal", "low", "dim"]:
            if remaining <= 0:
                break
            avail = len(pools[lt]) - taken[lt]
            add = min(avail, remaining)
            chosen += [(lt, c) for c in pools[lt][taken[lt]:taken[lt] + add]]
            taken[lt] += add
            remaining -= add
        final += [(cls, lt) + c for (lt, c) in chosen]
    return final


def write_crops(bdd_root: Path, out_root: Path, final, quality: int, min_box: int):
    """Tahap 3: buka citra sekali per file, potong bbox, tulis crop JPEG."""
    import cv2

    final.sort(key=lambda r: r[2])  # urut per citra agar caching efisien
    records = []
    per_img_idx = defaultdict(int)
    cache_key, cache_img = None, None
    for cls, lighting, img_rel, g, x1, y1, x2, y2 in final:
        if img_rel != cache_key:
            src = bdd_root / img_rel
            cache_img = cv2.imread(str(src)) if src.exists() else None
            cache_key = img_rel
        if cache_img is None:
            continue
        h, w = cache_img.shape[:2]
        X1, Y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        X2, Y2 = min(w, int(round(x2))), min(h, int(round(y2)))
        if (X2 - X1) < min_box or (Y2 - Y1) < min_box:
            continue
        crop = cache_img[Y1:Y2, X1:X2]
        k = per_img_idx[img_rel]
        per_img_idx[img_rel] += 1
        fname = f"{Path(img_rel).stem}__{k:03d}.jpg"
        dst = out_root / cls / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        try:
            rel = dst.relative_to(ROOT).as_posix()
        except ValueError:
            rel = dst.as_posix()
        records.append(dict(
            path=rel,
            label=cls,
            lighting=lighting,
            group_id=g,
            origin="real",
        ))
    return records


def split_groups(groups, group_classes, fracs, seed):
    """Bagi grup (video) ke split dengan greedy stratified per kelas (anti-kebocoran)."""
    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    total = Counter()
    for g in groups:
        for c in group_classes[g]:
            total[c] += 1
    target = {c: {s: fracs[s] * total[c] for s in fracs} for c in total}
    current = {c: {s: 0.0 for s in fracs} for c in total}
    assign = {}
    for g in order:
        best, best_score = None, float("-inf")
        for s in fracs:
            score = sum((target[c][s] - current[c][s]) / max(1, total[c]) for c in group_classes[g])
            if score > best_score:
                best, best_score = s, score
        assign[g] = best
        for c in group_classes[g]:
            current[c][best] += 1
    return assign


def main():
    ap = argparse.ArgumentParser(description="BDD100K -> dataset klasifikasi (pencahayaan asli)")
    ap.add_argument("--bdd-root", default="../bdd100k/bdd100k", type=str,
                    help="folder BDD100K (berisi images/ dan labels/)")
    ap.add_argument("--out", default="data/crops", type=str, help="folder keluaran crop (relatif ROOT)")
    ap.add_argument("--manifest", default="data/manifest.csv", type=str)
    ap.add_argument("--per-class", type=int, default=3000, help="maksimum crop per kelas")
    ap.add_argument("--min-box", type=int, default=32, help="ukuran minimum sisi bbox (piksel)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=95, help="kualitas JPEG keluaran")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--drop-truncated", action="store_true", default=True)
    ap.add_argument("--keep-truncated", dest="drop_truncated", action="store_false")
    ap.add_argument("--drop-occluded", action="store_true", default=False)
    ap.add_argument("--limit-images", type=int, default=0, help="batasi citra per file (uji cepat)")
    a = ap.parse_args()

    bdd_root = Path(a.bdd_root)
    if not bdd_root.is_absolute():
        bdd_root = ROOT / bdd_root
    out_root = Path(a.out)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    manifest = Path(a.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest

    print(f"BDD100K root : {bdd_root}")
    print(f"keluaran crop: {out_root}")
    print("Tahap 1/3: mengumpulkan kandidat crop ...")
    buckets, rng = collect_candidates(
        bdd_root, a.splits, a.per_class, a.seed, a.min_box,
        a.drop_truncated, a.drop_occluded, a.limit_images)

    print("Tahap 2/3: memilih crop seimbang ...")
    final = select_final(buckets, a.per_class, rng)
    print(f"  kandidat terpilih: {len(final)}")

    print("Tahap 3/3: menulis crop ...")
    records = write_crops(bdd_root, out_root, final, a.quality, a.min_box)

    fracs = {"train": 0.70, "val": 0.15, "test": 0.15}
    group_classes = defaultdict(set)
    for r in records:
        group_classes[r["group_id"]].add(r["label"])
    assign = split_groups(list(group_classes), group_classes, fracs, a.seed)
    for r in records:
        r["split"] = assign[r["group_id"]]

    manifest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "lighting", "group_id", "split", "origin"]
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    # ---- ringkasan ----
    def table(rows, keys):
        c = Counter(tuple(r[k] for k in keys) for r in rows)
        return c

    cls_split = table(records, ["label", "split"])
    cls_light = table(records, ["label", "lighting"])
    cls_light_split = table(records, ["label", "lighting", "split"])

    print("\nJumlah per kelas x split:")
    for cls in CLASSES:
        print(f"  {cls:14s}", {s: cls_split[(cls, s)] for s in ["train", "val", "test"]})
    print("\nJumlah per kelas x kondisi pencahayaan:")
    for cls in CLASSES:
        print(f"  {cls:14s}", {lt: cls_light[(cls, lt)] for lt in LIGHTINGS})
    empty = [k for k in cls_light_split if cls_light_split[k] == 0]
    print("\n(vanishing) sel kelas x kondisi x split yang kosong:",
          [f"{c}/{l}/{s}" for (c, l, s) in sorted(cls_light_split) if cls_light_split[(c, l, s)] == 0][:12])

    stats = dict(
        bdd_root=str(bdd_root),
        per_class=a.per_class,
        seed=a.seed,
        total_crops=len(records),
        groups=len(group_classes),
        per_class_split={c: {s: cls_split[(c, s)] for s in fracs} for c in CLASSES},
        per_class_lighting={c: {lt: cls_light[(c, lt)] for lt in LIGHTINGS} for c in CLASSES},
        per_class_lighting_split={f"{c}|{lt}": {s: cls_light_split[(c, lt, s)] for s in fracs}
                                  for c in CLASSES for lt in LIGHTINGS},
    )
    (ROOT / "data" / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest -> {manifest}")
    print(f"Statistik -> {ROOT / 'data' / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
