# Section 1. Business Understanding

**Tujuan:** Menerjemahkan permasalahan bisnis menjadi masalah yang dapat diselesaikan menggunakan Machine Learning (ML). Pada tahap ini, fokus utama adalah memahami konteks bisnis, mengidentifikasi kebutuhan stakeholder, serta menentukan tujuan dan indikator keberhasilan proyek sebelum melakukan analisis terhadap data.

## 1.1 Context

**Stakeholder: Customer Success (CS)**

Dalam industri telekomunikasi, mempertahankan pelanggan merupakan salah satu faktor penting untuk menjaga stabilitas pendapatan perusahaan. Perusahaan tidak hanya perlu mendapatkan pelanggan baru, tetapi juga memastikan pelanggan yang sudah berlangganan tetap menggunakan layanan dalam jangka panjang.

Salah satu tantangan yang dihadapi oleh tim Customer Success (CS) adalah *Customer Churn*, yaitu kondisi ketika pelanggan memutuskan untuk berhenti menggunakan layanan atau mengakhiri langganannya.

Berdasarkan konteks bisnis tersebut, tim Customer Success membutuhkan pendekatan yang lebih proaktif untuk memahami perilaku pelanggan dan mengidentifikasi pelanggan yang berpotensi berhenti berlangganan.

Dataset **Telco Customer Churn** digunakan dalam proyek ini karena menyediakan informasi historis mengenai profil pelanggan, karakteristik langganan, penggunaan layanan, biaya bulanan, serta status churn pelanggan.

Informasi tersebut dapat dimanfaatkan untuk menemukan pola yang berkaitan dengan perilaku pelanggan yang berhenti berlangganan maupun yang tetap bertahan.

## 1.2 Problem Statements

**Tujuan:** Merumuskan permasalahan bisnis secara spesifik agar dapat diterjemahkan menjadi permasalahan Machine Learning yang terukur.

Tim Customer Success (CS) mengalami kesulitan dalam mengidentifikasi pelanggan yang memiliki risiko churn secara proaktif karena proses pemantauan perilaku pelanggan masih memiliki keterbatasan dan cenderung bersifat manual.

Akibatnya, tindakan penyelamatan pelanggan (*customer retention intervention*), seperti pemberian penawaran promosi, penyesuaian paket langganan, maupun penyelesaian kendala teknis, sering kali baru dilakukan setelah pelanggan menunjukkan ketidakpuasan atau memutuskan untuk berhenti menggunakan layanan.

Pendekatan yang bersifat reaktif tersebut berpotensi menyebabkan beberapa permasalahan bisnis:

- **Penurunan Monthly Recurring Revenue (MRR):** Kehilangan pelanggan dapat mengurangi pendapatan berulang yang diperoleh perusahaan setiap bulannya.
- **Peningkatan Biaya Retensi:** Perusahaan berpotensi mengeluarkan biaya promosi dan insentif secara tidak efisien karena tidak memiliki informasi yang cukup untuk menentukan pelanggan yang benar-benar membutuhkan intervensi.
- **Kehilangan Customer Lifetime Value (CLV):** Pelanggan yang berhenti berlangganan lebih awal mengurangi potensi pendapatan yang seharusnya dapat diperoleh perusahaan selama hubungan berlangganan berlangsung.
- **Keterlambatan Intervensi:** Tim Customer Success tidak memiliki sistem prediksi yang dapat memberikan peringatan dini terhadap pelanggan yang berpotensi churn.

Oleh karena itu, diperlukan suatu model Machine Learning yang mampu mempelajari pola dari data historis pelanggan untuk memprediksi kemungkinan seorang pelanggan melakukan churn.

**Rumusan masalah utama:**

Bagaimana membangun model klasifikasi yang mampu mengidentifikasi pelanggan dengan risiko churn berdasarkan karakteristik profil, layanan, kontrak, dan biaya berlangganan, sehingga tim Customer Success dapat melakukan tindakan retensi secara lebih tepat sasaran dan tepat waktu?

## 1.3 Goals

**Tujuan:** Menetapkan target yang ingin dicapai oleh proyek sebagai bentuk penyelesaian dari permasalahan bisnis yang telah diidentifikasi.

Proyek ini bertujuan untuk membangun model prediksi Customer Churn yang dapat membantu tim Customer Success mengidentifikasi pelanggan yang berpotensi berhenti berlangganan berdasarkan pola yang ditemukan dalam data historis.

Secara lebih spesifik, tujuan proyek ini meliputi:

1. **Mengidentifikasi Pola Churn:** Menemukan karakteristik pelanggan dan faktor-faktor yang memiliki hubungan dengan kemungkinan terjadinya churn, seperti lama berlangganan (*tenure*), jenis kontrak, biaya bulanan (*MonthlyCharges*), serta penggunaan layanan tambahan.
2. **Membangun Model Prediksi:** Mengembangkan model klasifikasi menggunakan beberapa algoritma Machine Learning untuk memprediksi apakah pelanggan berpotensi churn atau tetap bertahan.
3. **Menghasilkan Prediksi Risiko:** Menghasilkan probabilitas churn untuk membantu mengelompokkan pelanggan berdasarkan tingkat risiko yang dimiliki.
4. **Mendukung Pengambilan Keputusan:** Memberikan informasi berbasis data kepada tim Customer Success agar strategi retensi dapat dilakukan secara lebih terarah.
5. **Meningkatkan Efisiensi Retensi:** Mengurangi potensi pemborosan biaya promosi dengan memprioritaskan pelanggan yang memiliki risiko churn lebih tinggi.

## 1.4 Analytical Approach

**Tujuan:** Menentukan pendekatan analitis dan teknis yang digunakan untuk menyelesaikan permasalahan bisnis.

Permasalahan Customer Churn akan diselesaikan menggunakan pendekatan **Supervised Machine Learning dengan metode Binary Classification**.

Pendekatan ini dipilih karena dataset telah memiliki label atau target berupa status churn pelanggan, yaitu:

- **No (0):** Pelanggan tidak melakukan churn atau masih menggunakan layanan.
- **Yes (1):** Pelanggan melakukan churn atau telah berhenti menggunakan layanan.

Model akan mempelajari hubungan antara karakteristik pelanggan dengan status churn berdasarkan data historis yang tersedia.

Tahapan analisis yang akan dilakukan meliputi:

1. **Exploratory Data Analysis (EDA):** Melakukan eksplorasi terhadap distribusi data, karakteristik pelanggan, proporsi churn, serta hubungan antarvariabel untuk memperoleh pemahaman awal mengenai pola perilaku pelanggan.
2. **Data Preprocessing:** Melakukan pemeriksaan kualitas data, menangani missing values, memeriksa duplikasi, serta memastikan tipe data sesuai dengan kebutuhan pemodelan.
3. **Feature Encoding:** Mengubah variabel kategorikal menjadi representasi numerik menggunakan teknik Binary Encoding dan One-Hot Encoding agar dapat diproses oleh algoritma Machine Learning.
4. **Feature Scaling:** Melakukan standardisasi terhadap variabel numerik yang membutuhkan penyetaraan skala, terutama untuk algoritma yang sensitif terhadap perbedaan skala fitur seperti Logistic Regression.
5. **Model Training:** Mengembangkan dan membandingkan beberapa algoritma klasifikasi, yaitu:
   - Logistic Regression
   - Decision Tree
   - Random Forest
6. **Model Evaluation:** Mengevaluasi performa model menggunakan metrik Accuracy, Precision, Recall, F1-Score, dan ROC-AUC untuk menentukan model yang paling sesuai dengan kebutuhan bisnis.
7. **Model Selection:** Memilih model berdasarkan kemampuan mengidentifikasi pelanggan yang benar-benar berisiko churn, dengan mempertimbangkan keseimbangan antara Recall dan Precision.

Model yang telah dipilih diharapkan dapat digunakan sebagai dasar pengembangan sistem peringatan dini (*early warning system*) bagi tim Customer Success.

## 1.5 Metric Evaluation (Business Metric & Machine Learning Evaluation Metric)

**Tujuan:** Menghubungkan indikator keberhasilan bisnis dengan metrik evaluasi Machine Learning agar performa model tidak hanya dinilai berdasarkan ketepatan prediksi, tetapi juga berdasarkan manfaatnya terhadap operasional perusahaan.

Evaluasi proyek dibagi menjadi dua kategori utama:

**A. Business Metrics**

Metrik bisnis digunakan untuk mengukur dampak penerapan model terhadap aktivitas retensi pelanggan.

| Business Metric | Tujuan Pengukuran |
|---|---|
| Churn Rate | Mengukur persentase pelanggan yang berhenti berlangganan dalam periode tertentu. |
| Monthly Recurring Revenue (MRR) | Memantau stabilitas pendapatan berulang bulanan yang berpotensi dipertahankan. |
| Customer Retention Rate | Mengukur kemampuan perusahaan mempertahankan pelanggan dalam periode tertentu. |
| Retention Cost | Mengukur biaya yang dikeluarkan untuk menjalankan program penyelamatan pelanggan. |
| Customer Lifetime Value (CLV) | Mengukur estimasi nilai ekonomi pelanggan selama hubungan berlangganan berlangsung. |

**B. Machine Learning Evaluation Metrics**

Metrik Machine Learning digunakan untuk mengukur kemampuan model dalam mengidentifikasi pelanggan yang berpotensi churn.

| ML Metric | Tujuan Pengukuran |
|---|---|
| Accuracy | Mengukur proporsi keseluruhan prediksi yang benar. |
| Precision | Mengukur seberapa banyak pelanggan yang diprediksi churn ternyata benar-benar churn. |
| Recall | Mengukur seberapa banyak pelanggan yang benar-benar churn berhasil diidentifikasi oleh model. |
| F1-Score | Mengukur keseimbangan antara Precision dan Recall. |
| ROC-AUC | Mengukur kemampuan model membedakan pelanggan yang churn dan tidak churn pada berbagai classification threshold. |

**Prioritas evaluasi:**

Dalam proyek ini, Recall dan F1-Score menjadi metrik yang mendapat perhatian utama.

Recall penting karena perusahaan perlu meminimalkan jumlah pelanggan yang sebenarnya berisiko churn tetapi tidak berhasil diidentifikasi oleh model (*False Negative*).

Sementara itu, Precision tetap perlu diperhatikan agar tim Customer Success tidak menghabiskan terlalu banyak anggaran dan sumber daya untuk menghubungi pelanggan yang sebenarnya tidak berisiko churn (*False Positive*).

Dengan demikian, pemilihan model tidak hanya berorientasi pada Accuracy tertinggi, tetapi juga mempertimbangkan kemampuan model dalam mendukung strategi retensi pelanggan.

## 1.6 Success Criteria

**Tujuan:** Menetapkan ambang batas keberhasilan yang digunakan untuk menentukan apakah model telah memenuhi kebutuhan teknis dan dapat dipertimbangkan untuk mendukung proses bisnis.

Model akan dianggap memenuhi target awal apabila mampu mencapai kriteria evaluasi berikut:

| Indikator | Target Awal | Alasan |
|---|---:|---|
| Recall | ≥ 80% | Meminimalkan pelanggan berisiko churn yang tidak berhasil teridentifikasi. |
| Precision | ≥ 60% | Menjaga agar prediksi churn tetap relevan dan mengurangi intervensi yang tidak diperlukan. |
| F1-Score | ≥ 65% | Menjaga keseimbangan antara Precision dan Recall. |
| ROC-AUC | ≥ 80% | Memastikan model memiliki kemampuan diskriminasi yang baik antara pelanggan churn dan non-churn. |

Selain memenuhi target teknis, terdapat beberapa kriteria keberhasilan dari sisi bisnis:

1. **Early Identification:** Model mampu memberikan identifikasi awal terhadap pelanggan yang memiliki kemungkinan churn lebih tinggi.
2. **Actionable Prediction:** Hasil prediksi dapat diterjemahkan menjadi informasi yang dapat digunakan tim Customer Success untuk menentukan prioritas intervensi.
3. **Resource Efficiency:** Prediksi dapat membantu perusahaan mengalokasikan sumber daya dan anggaran retensi secara lebih terarah.
4. **Business Relevance:** Model menghasilkan informasi yang relevan dengan kebutuhan bisnis, bukan hanya menghasilkan nilai evaluasi yang tinggi secara statistik.

Target evaluasi di atas merupakan sasaran awal proyek dan perlu divalidasi melalui hasil eksperimen serta pertimbangan operasional perusahaan.

Keberhasilan akhir proyek tidak hanya ditentukan oleh performa model secara teknis, tetapi juga oleh kemampuannya memberikan informasi yang dapat mendukung pengambilan keputusan dalam upaya mempertahankan pelanggan.