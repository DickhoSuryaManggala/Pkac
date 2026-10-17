**SOLUSI PENELITIAN BERDASARKAN RESEARCH GAP**

**Analisis Pengaruh Variasi Kondisi Pencahayaan terhadap Generalisasi Model MobileNetV2 dan ResNet50 pada Klasifikasi Jenis Kendaraan**

# 1. Solusi untuk Setiap Permasalahan (Research Gap)

Berdasarkan ringkasan research gap dari tabel studi literatur, solusi utama yang diusulkan adalah merancang eksperimen yang menguji pengaruh perubahan pencahayaan terhadap performa dan kemampuan generalisasi MobileNetV2 dan ResNet50 dalam klasifikasi jenis kendaraan. Penelitian tidak hanya membandingkan model dengan akurasi tertinggi, tetapi juga menilai model yang lebih tahan terhadap perubahan kondisi pencahayaan.

| **No.** | **Permasalahan (Research Gap)** | **Solusi yang Diusulkan** |
| --- | --- | --- |
| 1 | Belum jelas model mana yang lebih baik antara MobileNetV2 dan ResNet50 pada variasi pencahayaan. | Melakukan eksperimen komparatif menggunakan kedua arsitektur dengan dataset, pembagian data, dan prosedur evaluasi yang sebanding. |
| 2 | Performa model pada kondisi pencahayaan berbeda dari data pelatihan belum diketahui secara memadai. | Melatih model pada kondisi pencahayaan tertentu, kemudian mengujinya pada kondisi berbeda tanpa pelatihan ulang. |
| 3 | Pengaruh pencahayaan belum dipisahkan secara sistematis dari faktor kualitas citra lainnya. | Membuat kelompok citra berdasarkan tingkat pencahayaan dan menjaga resolusi, kelas kendaraan, serta faktor lain tetap sebanding. |
| 4 | Akurasi saja belum cukup untuk menjelaskan ketahanan model. | Menggunakan accuracy, precision, recall, macro-F1, confusion matrix, dan penurunan performa antarkondisi. |
| 5 | Belum diketahui model yang paling sesuai untuk kondisi pencahayaan bervariasi. | Membandingkan ketahanan performa, waktu inferensi, dan ukuran model untuk menilai kompromi antara akurasi dan efisiensi. |

# 2. Metode Penyelesaian yang Direkomendasikan

## Tahap 1 — Menyiapkan dataset kendaraan

Kumpulkan citra dengan kelas yang konsisten, misalnya mobil, sepeda motor, bus, dan truk. Pisahkan data pelatihan, validasi, dan pengujian tanpa kebocoran data antarkelompok.

## Tahap 2 — Membuat variasi kondisi pencahayaan

Kelompokkan citra menjadi kondisi normal, redup, dan rendah. Gunakan citra asli dari berbagai kondisi jika tersedia. Jika menggunakan simulasi digital, terapkan perubahan kecerahan secara terkontrol dan dokumentasikan parameternya.

## Tahap 3 — Melatih dan membandingkan model

Latih MobileNetV2 dan ResNet50 dengan prosedur yang sebanding. Uji performa pada pencahayaan yang sama dengan data pelatihan dan pada pencahayaan berbeda.

## Tahap 4 — Menganalisis hasil dan menentukan model terbaik

Bandingkan accuracy, macro-F1, penurunan performa, confusion matrix, dan waktu inferensi. Model terbaik ditentukan berdasarkan hasil eksperimen.

# 3. Rancangan Eksperimen untuk Menguji Generalisasi

| **Skenario** | **Kondisi Pelatihan** | **Kondisi Pengujian** | **Tujuan** |
| --- | --- | --- | --- |
| A | Normal | Normal | Mengukur performa dasar model. |
| B | Normal | Redup | Mengukur generalisasi terhadap pencahayaan redup. |
| C | Normal | Rendah | Mengukur ketahanan pada pencahayaan rendah. |
| D | Normal, redup, dan rendah | Data uji terpisah dengan variasi pencahayaan | Mengukur generalisasi setelah pelatihan dengan kondisi beragam. |

Catatan metodologis: Gunakan identitas gambar atau sumber video yang terpisah antara pelatihan dan pengujian. Jika gambar yang sama dimanipulasi menjadi beberapa tingkat kecerahan, seluruh versi gambar tersebut harus ditempatkan pada kelompok data yang sama untuk mencegah kebocoran data.

# 4. Teknik Tambahan yang Dapat Diuji

| **Teknik** | **Fungsi** | **Rancangan Evaluasi** |
| --- | --- | --- |
| Gamma correction atau CLAHE | Meningkatkan keterlihatan detail pada citra gelap atau kontras rendah. | Bandingkan performa model dengan citra asli dan citra yang telah diproses. |
| Brightness augmentation | Melatih model dengan variasi kecerahan agar lebih terbiasa terhadap perubahan pencahayaan. | Bandingkan model tanpa augmentasi kecerahan dengan model yang menggunakannya. |
| Transfer learning dan fine-tuning | Memanfaatkan bobot model pretrained lalu menyesuaikannya pada dataset kendaraan. | Gunakan prosedur pelatihan yang konsisten dan bandingkan metrik pada tiap kondisi pencahayaan. |

Teknik tambahan tersebut merupakan alternatif eksperimen. Efektivitasnya belum dapat dipastikan sebelum diuji pada dataset yang digunakan.

# 5. Rekomendasi Akhir

Eksperimen utama: bandingkan MobileNetV2 dan ResNet50 pada tiga kondisi pencahayaan.

Analisis generalisasi: latih model pada pencahayaan normal, kemudian uji pada kondisi redup dan rendah tanpa pelatihan ulang.

Evaluasi: ukur performa setiap model dan besarnya penurunan hasil ketika pencahayaan berubah.

Eksperimen lanjutan: jika ditemukan penurunan yang berarti, uji brightness augmentation atau teknik peningkatan citra.

Dengan pendekatan ini, kontribusi penelitian bukan hanya menentukan model yang lebih akurat, melainkan mengidentifikasi bagaimana variasi pencahayaan memengaruhi generalisasi model dan mengevaluasi model yang lebih tahan terhadap perubahan tersebut. Hasilnya dapat membantu memilih arsitektur yang sesuai untuk klasifikasi kendaraan berbasis citra pada kondisi pencahayaan yang bervariasi.