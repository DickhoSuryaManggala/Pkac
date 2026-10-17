# Flow Pengerjaan

Alur dari data mentah sampai laporan. Jangan lompat tahap.

```
[1 Dataset] -> [2 Verifikasi pencahayaan] -> [3 Latih] -> [4 Evaluasi] -> [5 Lanjutan*] -> [6 Laporan]
```

| # | Tahap | Perintah | Keluaran utama | Gate |
|---|---|---|---|---|
| 1 | Dataset & split | `scripts/bdd100k_to_dataset.py` | `data/manifest.csv` | tanpa kebocoran `group_id`; tiap kelas ada di semua split |
| 2 | Verifikasi pencahayaan | `scripts/check_lighting.py` | `luminance_stats.csv`, `lighting_samples.png` | luminansi normal > dim > low |
| 3 | Pelatihan | `scripts/run_experiments.py` | `results/<model>__<regime>__seed<N>/` | smoke test lolos; semua seed selesai |
| 4 | Evaluasi | `python -m vehicle_lighting.aggregate` | `results/summary/*` | mean +/- std, penurunan, confusion matrix, McNemar, precision/recall per kelas, efisiensi |
| 5 | Lanjutan (opsional) | `run_experiments.py --regimes normal_brightaug normal_gamma normal_clahe` | tabel baseline vs brightness aug / gamma / CLAHE | kesimpulan jujur per teknik |
| 6 | Laporan | `python -m vehicle_lighting.report` | `docs/laporan/laporan.md` | angka terlacak, keterbatasan tertulis |

## Skenario eksperimen
| Skenario | Latih | Uji | Regime |
|---|---|---|---|
| A | normal | normal | `normal` |
| B | normal | dim | `normal` |
| C | normal | low | `normal` |
| D | normal + dim + low | semua kondisi (uji terpisah) | `mixed` |

## Catatan mode BDD100K (setup saat ini)
- Tahap 1 dibangun oleh `scripts/bdd100k_to_dataset.py` (crop bbox + peta kelas + `timeofday`).
- Kondisi pencahayaan nyata, jadi tidak ada sintesis pencahayaan pada alur utama.

## Perkiraan beban kerja (GPU tunggal, ~5-12 ribu citra)
- Dataset + verifikasi: beberapa jam (pemindaian JSON paling lama).
- Latih: 2 model x 2 regime x 3 seed = 12 run (+ 18 run bila semua teknik lanjutan).
- Evaluasi + laporan: beberapa hari tergantung revisi.

## Checklist mutu sebelum menulis kesimpulan
- [ ] Split berbasis grup; tidak ada video sumber yang sama lintas split
- [ ] Parameter pencahayaan tercatat; luminansi per kondisi terukur
- [ ] Prosedur latih identik untuk kedua model; >= 3 seed
- [ ] Checkpoint dipilih dari val kondisi latih, bukan dari kondisi uji
- [ ] Penurunan performa dihitung per seed, dilaporkan mean +/- std
- [ ] Keterbatasan (timeofday sebagai proxy pencahayaan) tertulis
- [ ] Sitasi diverifikasi manual (lihat `docs/referensi/README.md`)
