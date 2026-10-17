**ANALISIS RESEARCH GAP**

**Analisis Pengaruh Variasi Kondisi Pencahayaan terhadap Generalisasi Model MobileNetV2 dan ResNet50 pada Klasifikasi Jenis Kendaraan**

# 1. Identifikasi Kesenjangan Penelitian Berdasarkan Artikel Terdahulu

Berdasarkan tabel studi literatur yang diberikan, kelima artikel telah membahas klasifikasi atau deteksi kendaraan, penggunaan arsitektur CNN, kualitas citra rendah, serta kondisi pencahayaan yang menantang. Namun, berdasarkan ringkasan yang tersedia, masih terdapat peluang untuk menguji secara lebih terstruktur pengaruh variasi pencahayaan terhadap kemampuan generalisasi model MobileNetV2 dan ResNet50.

| **No.** | **Referensi** | **Fokus Penelitian Terdahulu** | **Kesenjangan Penelitian (Research Gap)** |
| --- | --- | --- | --- |
| 1 | Convolutional Neural Network Based Vehicle Classification in Adverse Illuminous Conditions for Intelligent Transportation Systems (2021) | Klasifikasi kendaraan pada kondisi pencahayaan yang kurang baik menggunakan beberapa arsitektur CNN. | Perlu dikaji perbandingan MobileNetV2 dan ResNet50 dengan skenario pencahayaan terkontrol serta evaluasi generalisasi yang konsisten. |
| 2 | Convolutional Neural Network-Based Vehicle Classification in Low-Quality Imaging Conditions for Internet of Things Devices (2023) | Klasifikasi kendaraan pada kondisi citra berkualitas rendah. | Perlu ditelaah apakah pengaruh pencahayaan diuji secara terpisah dari faktor kualitas citra lainnya. Penelitian lanjutan dapat mengisolasi pencahayaan sebagai variabel eksperimen. |
| 3 | Vehicle Type Recognition: A Case Study of MobileNetV2 for an Image Classification Task (2024) | Penerapan MobileNetV2 dengan variasi penyiapan data, normalisasi, augmentasi, dan validasi. | Ringkasan belum menunjukkan perbandingan langsung MobileNetV2 dan ResNet50 secara khusus berdasarkan tingkat pencahayaan dan generalisasi lintas kondisi. |
| 4 | Fast and Accurate Deep Learning Architecture on Vehicle Type Recognition (2022) | Perbandingan arsitektur CNN dan teknik augmentasi, termasuk augmentasi kecerahan. | Masih terbuka peluang untuk membandingkan performa kedua model pada tingkat pencahayaan yang ditetapkan secara sistematis, baik pada kondisi yang dikenal maupun kondisi yang tidak digunakan saat pelatihan. |
| 5 | A Deep-Learning-Based Vehicle Detection Approach for Insufficient and Nighttime Illumination Conditions (2019) | Deteksi kendaraan pada kondisi pencahayaan tidak memadai dan malam hari. | Fokus penelitian adalah deteksi, bukan klasifikasi jenis kendaraan. Penelitian ini dapat menjadi landasan untuk menguji klasifikasi jenis kendaraan ketika pencahayaan berubah. |

# 2. Rumusan Research Gap Utama

## 2.1 Kesenjangan pada Perbandingan Model

Penelitian terdahulu telah menggunakan CNN dan MobileNetV2 untuk klasifikasi kendaraan serta membahas berbagai arsitektur pada kondisi citra yang menantang. Namun, berdasarkan ringkasan tabel, belum terlihat evaluasi yang secara khusus membandingkan MobileNetV2 dan ResNet50 dalam satu rancangan eksperimen dengan variasi pencahayaan yang konsisten.

## 2.2 Kesenjangan pada Evaluasi Generalisasi

Kinerja model pada dataset pengujian yang serupa dengan data pelatihan belum tentu mencerminkan kemampuannya menghadapi citra dengan karakteristik pencahayaan berbeda. Oleh karena itu, diperlukan pengujian yang membedakan performa pada kondisi pencahayaan yang telah dipelajari dan kondisi yang tidak digunakan selama pelatihan.

## 2.3 Kesenjangan pada Pengujian Pencahayaan secara Terkontrol

Citra berkualitas rendah dapat dipengaruhi beberapa faktor, seperti pencahayaan, noise, resolusi, dan ketajaman. Penelitian ini dapat memfokuskan eksperimen pada pencahayaan dengan menjaga faktor lain tetap sebanding, sehingga perubahan performa model dapat dianalisis secara lebih terarah.

# 3. Kebaruan yang Dapat Diusulkan (Potential Novelty)

Penelitian ini berfokus pada analisis komparatif kemampuan generalisasi MobileNetV2 dan ResNet50 dalam mengklasifikasikan jenis kendaraan pada variasi kondisi pencahayaan yang berbeda. Evaluasi tidak hanya dilakukan berdasarkan akurasi klasifikasi, tetapi juga berdasarkan perubahan performa model ketika diuji pada kondisi pencahayaan yang berbeda dari kondisi pelatihannya.

# 4. Rancangan Eksperimen yang Disarankan

| **Komponen** | **Rancangan** |
| --- | --- |
| Model | MobileNetV2 dan ResNet50. |
| Objek | Jenis kendaraan, misalnya mobil, sepeda motor, bus, dan truk, sesuai kelas dataset. |
| Variabel bebas | Kondisi pencahayaan: normal, redup, dan rendah. |
| Variabel terikat | Accuracy, precision, recall, macro-F1, dan penurunan performa. |
| Variabel yang dikendalikan | Dataset, kelas, pembagian data, prosedur pelatihan, dan metode evaluasi. |
| Skenario generalisasi | Melatih model pada kondisi tertentu, lalu menguji pada kondisi pencahayaan berbeda tanpa pelatihan ulang. |
| Analisis tambahan | Confusion matrix dan perbandingan penurunan performa antara kedua model. |

# 5. Paragraf Research Gap untuk Proposal atau Artikel

Berdasarkan hasil telaah terhadap sejumlah penelitian terdahulu, klasifikasi jenis kendaraan berbasis deep learning telah dikembangkan melalui pemanfaatan berbagai arsitektur Convolutional Neural Network (CNN), termasuk MobileNetV2 dan ResNet50. Penelitian sebelumnya juga telah mengkaji klasifikasi kendaraan pada kondisi pencahayaan yang kurang memadai, citra berkualitas rendah, serta penerapan teknik augmentasi untuk meningkatkan performa model. Meskipun demikian, berdasarkan informasi yang dirangkum dalam tabel studi literatur, masih terdapat peluang untuk melakukan evaluasi komparatif terhadap MobileNetV2 dan ResNet50 dengan skenario variasi pencahayaan yang terkontrol. Selain itu, evaluasi performa pada data pengujian yang memiliki kondisi pencahayaan berbeda dari data pelatihan perlu mendapat perhatian, karena akurasi yang tinggi pada satu kondisi belum tentu menunjukkan kemampuan generalisasi yang baik pada kondisi lainnya. Oleh sebab itu, penelitian ini diarahkan untuk menganalisis pengaruh variasi kondisi pencahayaan terhadap kemampuan generalisasi kedua arsitektur tersebut dalam klasifikasi jenis kendaraan, dengan mengevaluasi metrik klasifikasi serta perubahan performa pada setiap kondisi pengujian. Hasil penelitian diharapkan dapat memberikan bukti empiris mengenai perbedaan ketahanan MobileNetV2 dan ResNet50 terhadap perubahan pencahayaan serta membantu menentukan model yang lebih sesuai untuk klasifikasi kendaraan pada kondisi citra yang bervariasi.

# 6. Catatan Metodologis

Untuk menghindari kesimpulan yang bias, pengujian sebaiknya menggunakan citra asli dari kondisi pencahayaan berbeda jika tersedia. Jika kondisi redup dibuat melalui manipulasi digital, jelaskan bahwa eksperimen menguji perubahan pencahayaan sintetis dan tidak sepenuhnya mewakili kondisi malam hari di dunia nyata. Klaim kebaruan perlu dikonfirmasi melalui pembacaan artikel lengkap dan penelusuran literatur terbaru. Research gap dalam dokumen ini merupakan analisis awal berdasarkan ringkasan tabel studi literatur, bukan bukti definitif bahwa aspek tersebut belum pernah diteliti.