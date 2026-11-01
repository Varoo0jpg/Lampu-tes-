# INVENTAR PERSAMAAN DAN FORMULA PENELITIAN LAMPIU
## Dokumen kerja untuk audit independen oleh Gemini

**Status:** inventaris awal berbasis materi repo/catatan eksperimen yang tersedia di percakapan. Ini BUKAN hasil audit final terhadap seluruh paper dan seluruh jurnal referensi.

**Tujuan audit Gemini**
1. Cocokkan setiap persamaan penelitian dengan dokumen, jurnal, artikel, standar, atau paper sumber yang benar-benar dirujuk dalam naskah.
2. Tentukan bentuk formula original dari sumber primer.
3. Bandingkan formula original dengan formula di paper, kode, notebook, dan implementasi GUI.
4. Identifikasi modifikasi, penambahan, pengurangan, perubahan notasi, perubahan asumsi, dan kesalahan implementasi.
5. Perbaiki formula hanya jika ada bukti sumber dan alasan metodologis yang jelas.
6. Tambahkan formula yang memang diperlukan untuk menjelaskan metode/evaluasi penelitian, tetapi tandai sebagai usulan dan jangan mengklaim telah digunakan sebelum dibuktikan di kode/eksperimen.

---

# A. BATASAN DAN ATURAN AUDIT

- Jangan mengarang referensi, DOI, nomor halaman, persamaan, parameter, atau hasil eksperimen.
- Gunakan naskah paper dan daftar pustakanya sebagai titik awal. Lacak setiap referensi sampai sumber primer bila memungkinkan.
- Untuk setiap formula, catat: sumber, nomor persamaan/halaman, bentuk original, bentuk di paper, bentuk di kode, perbedaan, dampak, dan rekomendasi.
- Bedakan **formula benar-benar diimplementasikan**, **formula ditulis di paper**, **formula teori yang dirujuk**, dan **formula usulan yang belum diterapkan**.
- Jangan mengubah hasil eksperimen hanya untuk membuatnya sesuai teori. Jika implementasi tidak sesuai formula sumber, laporkan sebagai temuan dan usulkan koreksi terpisah.
- Jangan menyatakan sebuah formula berasal dari YOLO/YAMNet/SVM atau jurnal tertentu tanpa sumber yang dapat diverifikasi.
- Notasi di bawah merupakan bentuk matematis yang merepresentasikan perilaku yang tercatat. Gemini wajib memeriksa kembali kode aktual sebelum menyebutnya persamaan implementasi final.

# B. SISTEM PENGATURAN LAMPU LALU LINTAS ADAPTIF

## B1. Bobot kendaraan per jenis

Konfigurasi bobot yang tercatat pada `GUI/smart_traffic_gui_100.py`:

| Jenis | Bobot tercatat |
|---|---:|
| Sepeda motor | 0,25 |
| Mobil | 1 |
| Truk | 2 |
| Bus | 2 |

Bentuk umum yang merepresentasikan perhitungan bobot lajur:

\[
W_i = \sum_{k=1}^{K} w_k N_{i,k}
\]

Keterangan:
- \(W_i\): total bobot kendaraan pada lajur \(i\).
- \(w_k\): bobot untuk kelas kendaraan \(k\).
- \(N_{i,k}\): jumlah kendaraan kelas \(k\) pada lajur \(i\).
- \(K\): jumlah kelas kendaraan yang dihitung.

**Status:** nilai bobot dan pola penjumlahan tercatat dalam ringkasan pemeriksaan kode sebelumnya. Verifikasi ulang ekspresi persis, kelas yang masuk hitungan, serta penanganan deteksi duplikat di file kode.

## B2. Ambang kepadatan

Nilai ambang yang tercatat:

\[
W_i \geq 5.0
\]

**Penting:** 5,0 adalah nilai ambang di kode yang pernah diperiksa, bukan otomatis ambang kemacetan yang tervalidasi secara empiris. Audit perlu mencari sumber penentuan angka ini: jurnal, kalibrasi, uji lapangan, atau keputusan desain. Jika tidak ada, tulis sebagai parameter heuristik/rekayasa dan jelaskan metode penetapannya.

Terdapat ketidaksesuaian dokumentasi yang perlu diperiksa: catatan README pernah menyebut kombinasi 2 mobil + 2 truk dengan bobot truk 1,5, sedangkan kode yang diperiksa mencatat bobot truk 2. Dengan bobot kode, kombinasi tersebut menghasilkan \(2(1)+2(2)=6\), bukan 5. Audit harus menentukan dokumentasi mana yang usang dan memperbaruinya berdasarkan implementasi yang disepakati.

## B3. Pembagian waktu hijau proporsional

Bentuk formula yang merepresentasikan perilaku kode yang pernah diperiksa:

\[
T_i = \frac{W_i}{\sum_{j=1}^{M} W_j}\,T_{\mathrm{total}}
\]

Pada implementasi yang diperiksa, \(T_{\mathrm{total}}=120\) detik:

\[
T_i = \frac{W_i}{\sum_{j=1}^{M} W_j}\times 120
\]

Keterangan:
- \(T_i\): waktu hijau untuk lajur \(i\).
- \(W_i\): bobot lajur \(i\).
- \(M\): jumlah lajur yang ikut alokasi.
- \(T_{\mathrm{total}}\): total waktu yang dibagi.

**Kasus khusus:** jika \(\sum_j W_j=0\), catatan kode menyebut pembagian waktu sama rata. Bentuk matematisnya:

\[
T_i = \frac{T_{\mathrm{total}}}{M}, \quad \text{jika } \sum_{j=1}^{M}W_j=0
\]

Audit harus memeriksa apakah 120 detik merupakan total siklus, total waktu hijau saja, atau sekadar anggaran waktu hijau; apakah waktu kuning/all-red, batas minimum/maksimum, dan waktu transisi ikut dihitung. Jangan menyamakan total waktu hijau dengan total siklus tanpa bukti.

## B4. Formula yang perlu dicari di kode/paper, belum terkonfirmasi

- Penghalusan jumlah kendaraan antar-frame (moving average/EMA).
- Penghitungan antrean berdasarkan posisi kendaraan, bukan hanya jumlah deteksi.
- Prioritas kendaraan darurat dan aturan pre-emption.
- Batas minimum/maksimum waktu hijau.
- Waktu kuning, all-red, clearance interval, dan total cycle length.
- Penalti waktu tunggu atau fairness antar-lajur.
- Aturan pergiliran fase dan konflik gerakan.

Jangan masukkan formula standar berikut sebagai implementasi penelitian sebelum kode membuktikannya. Jika diperlukan untuk melengkapi metodologi, masukkan di bagian **usulan** dan jelaskan konsekuensi implementasinya.

# C. DETEKSI OBJEK YOLO26n

## C1. Keluarga formula yang harus diaudit

Paper dan kode perlu ditelusuri untuk memastikan definisi yang benar-benar digunakan pada versi YOLO26n yang dipakai. Periksa khusus:
- bounding box dan koordinat prediksi;
- IoU;
- objectness/class confidence sesuai arsitektur dan implementasi versi model;
- precision, recall;
- AP dan mAP;
- komponen loss box, classification, dan distribution focal loss (DFL), bila memang digunakan oleh implementasi tersebut;
- threshold confidence dan NMS/strategi post-processing.

Jangan menyalin formula YOLO versi lama begitu saja. Versi arsitektur dan implementasi harus sama dengan model yang dilatih. Pastikan apakah istilah/objectness dan komponen loss tertentu masih relevan untuk versi yang dipakai.

## C2. IoU — formula evaluasi kandidat

\[
IoU = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}
\]

\(B_p\) adalah bounding box prediksi dan \(B_{gt}\) adalah ground-truth. Formula ini adalah definisi umum IoU; audit harus mencocokkannya dengan sumber evaluasi yang dikutip dan implementasi evaluator.

## C3. Precision dan Recall — formula evaluasi kandidat

\[
Precision = \frac{TP}{TP+FP}
\]

\[
Recall = \frac{TP}{TP+FN}
\]

Definisikan TP, FP, FN berdasarkan aturan matching deteksi, kelas, dan IoU threshold. Laporkan threshold IoU dan confidence yang digunakan.

## C4. AP dan mAP — formula evaluasi kandidat

AP merupakan luas di bawah kurva precision-recall menurut metode interpolasi/integrasi yang dipakai evaluator:

\[
AP = \int_0^1 P_{\mathrm{interp}}(r)\,dr
\]

Bentuk diskret dan interpolasi dapat berbeda antar evaluator. Jangan mengasumsikan formula sederhana ini identik dengan implementasi Ultralytics yang dipakai. mAP rata-rata AP pada kelas dan/atau IoU threshold tertentu:

\[
mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c
\]

Untuk mAP@[.5:.95], pastikan definisi rata-rata pada beberapa IoU threshold sesuai evaluator yang digunakan. Tuliskan rentang threshold dan langkahnya secara eksplisit.

## C5. Loss YOLO — parameter yang tercatat, bukan formula lengkap

Konfigurasi training yang pernah ditemukan mencatat:
- `box = 7.5`
- `cls = 0.5`
- `dfl = 1.5`

Secara konseptual, total loss dapat berbentuk kombinasi berbobot:

\[
L_{\mathrm{total}} =
\lambda_{\mathrm{box}}L_{\mathrm{box}}+
\lambda_{\mathrm{cls}}L_{\mathrm{cls}}+
\lambda_{\mathrm{dfl}}L_{\mathrm{dfl}}
\]

dengan nilai koefisien yang tercatat di konfigurasi di atas. **Ini hanya representasi konseptual dari bobot konfigurasi**, bukan klaim bahwa formula internal loss persis demikian tanpa memeriksa source code versi Ultralytics/YOLO26 yang digunakan. Audit harus memastikan nama komponen, normalisasi, reduction, assignment, dan detail loss pada versi tepat.

## C6. Konfigurasi eksperimen yang pernah tercatat

Salah satu konfigurasi `runs/detect/train/args.yaml`:
- model: `yolo26n.pt`
- epochs: 50
- patience: 100
- batch: 8
- imgsz: 640
- workers: 8
- pretrained: true
- `cls_remap`: true
- optimizer: auto
- seed: 0
- deterministic: true
- fraction: 1.0
- lr0: 0.01
- lrf: 0.01
- momentum: 0.937
- weight_decay: 0.0005
- warmup_epochs: 3

Konfigurasi pilot 10 epoch yang juga tercatat berbeda: batch 4, device CPU, workers 0, fraction 0.1. Jangan gabungkan dua konfigurasi ini menjadi satu eksperimen. Identifikasi run, tanggal, dataset, dan checkpoint masing-masing.

## C7. Kelas dataset — perlu rekonsiliasi versi

Repo/catatan menunjukkan skema kelas yang pernah berbeda:
- versi 5 kelas: Ambulance, Fire_truck, Normal_vehicle, Police_car, Towing_truck;
- versi 7 kelas: Ambulance, Fire_Truck, Car, Motorcycle, Police_Car, Truck, Bus.

Pastikan paper menyebut skema kelas yang sesuai dengan eksperimen final. Jangan menggabungkan hasil dari dua skema kelas berbeda.

# D. PENGENALAN SIRENE: YAMNET + EMBEDDING + CLASSIFIER

## D1. Representasi embedding

Catatan eksperimen menunjukkan embedding YAMNet berdimensi 1024 dan jumlah sampel 1547. Embedding frame-level dirata-ratakan menjadi satu vektor per rekaman (verifikasi pooling persis di notebook):

\[
\mathbf{e} = \frac{1}{N}\sum_{t=1}^{N}\mathbf{e}_t
\]

- \(\mathbf{e}_t\): embedding pada frame/patch ke-\(t\).
- \(N\): jumlah embedding frame/patch.
- \(\mathbf{e}\): vektor embedding hasil pooling.

Jangan mengasumsikan mean pooling jika kode final menggunakan pooling lain. Periksa preprocessing audio, panjang klip, sample rate, channel, resampling, normalisasi, dan apakah embedding diproses per frame atau per klip.

## D2. SVM dengan kernel RBF — kandidat formula

\[
K(\mathbf{x}_i,\mathbf{x}_j)
=\exp\left(-\gamma\|\mathbf{x}_i-\mathbf{x}_j\|^2\right)
\]

Periksa nilai `C`, `gamma`, class weights, standardisasi fitur, probability setting, dan strategi validasi. Jangan mengklaim formula ini sebagai classifier final sebelum cocok dengan notebook/model tersimpan.

## D3. Logistic Regression — kandidat formula

Catatan eksperimen yang tersedia menunjukkan model terpilih versi V2 adalah `LogisticRegression_C100`. Bentuk probabilitas multinomial standar:

\[
P(y=k\mid\mathbf{x})=
\frac{\exp(\boldsymbol{\beta}_k^\top\mathbf{x}+b_k)}
{\sum_{j=1}^{K}\exp(\boldsymbol{\beta}_j^\top\mathbf{x}+b_j)}
\]

Loss cross-entropy multinomial standar:

\[
L=-\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\log \hat{p}_{ik}
\]

Keduanya merupakan bentuk teori umum. Audit kode harus memeriksa solver, penalty, regularization, intercept, multiclass behavior versi scikit-learn, scaling, dan parameter `C`. Jangan menyatakan formula ini sudah digunakan persis tanpa pemeriksaan.

## D4. Pembagian dataset dan hasil eksperimen yang tercatat

Catatan audio:
- total embedding: 1547 × 1024;
- train: 1079;
- validation: 235;
- test: 233;
- pemisahan disebut group-aware split v2, tetapi grup dan aturan pembagiannya harus diperiksa dari notebook.

Pada catatan V2:
- model terpilih: `LogisticRegression_C100`;
- validation macro F1: 0.921727;
- test samples: 233;
- test accuracy: 0.866953;
- balanced accuracy: 0.861590;
- macro precision: 0.862215;
- macro recall: 0.861590;
- macro F1: 0.860076.

**Peringatan konflik versi:** catatan lain yang lebih lama menyebut SVM RBF sebagai model terpilih dan metrik test berbeda. Jangan campur hasil V1 dan V2. Gemini harus menemukan notebook/checkpoint/CSV yang menjadi hasil final dan memastikan angka di paper cocok.

## D5. Metrik klasifikasi audio

Untuk kelas \(k\):

\[
Precision_k=\frac{TP_k}{TP_k+FP_k}
\]

\[
Recall_k=\frac{TP_k}{TP_k+FN_k}
\]

\[
F1_k=2\frac{Precision_k\cdot Recall_k}{Precision_k+Recall_k}
\]

Macro precision/recall/F1 adalah rata-rata aritmetika metrik tiap kelas:

\[
MacroF1=\frac{1}{K}\sum_{k=1}^{K}F1_k
\]

Balanced accuracy multiclass umumnya merupakan rata-rata recall tiap kelas:

\[
BalancedAccuracy=\frac{1}{K}\sum_{k=1}^{K}Recall_k
\]

Accuracy:

\[
Accuracy=\frac{\sum_k TP_k}{N}
\]

Formula perlu dicocokkan dengan fungsi metric yang benar-benar dipanggil dan `average` parameter. Untuk kasus multiclass, TP/FP/FN dihitung one-vs-rest per kelas.

## D6. Confusion matrix

Buat confusion matrix dengan urutan kelas yang dinyatakan eksplisit. Catatan eksperimen lama dan baru mungkin memiliki urutan/angka berbeda. Jangan salin matriks lama ke hasil V2 tanpa verifikasi file `test_confusion_matrix_v2.csv`.

# E. AUDIO FEATURE DIAGNOSTICS

Catatan diagnostik menampilkan RMS, peak, clipping ratio, silence ratio, spectral centroid, dan zero-crossing rate (ZCR). Pastikan formula, frame length, hop length, threshold, dan skala amplitudo di kode.

## E1. Root Mean Square (RMS)

\[
RMS=\sqrt{\frac{1}{N}\sum_{n=1}^{N}x[n]^2}
\]

## E2. Peak amplitude

\[
Peak=\max_n |x[n]|
\]

## E3. Clipping ratio — definisi operasional harus diverifikasi

Bentuk kandidat:

\[
ClippingRatio=\frac{1}{N}\sum_{n=1}^{N}
\mathbb{1}(|x[n]|\geq \tau)
\]

\(\tau\) adalah ambang clipping. Jangan mengasumsikan \(\tau=1\) atau nilai lain sebelum memeriksa kode dan format audio.

## E4. Silence ratio — definisi operasional harus diverifikasi

Bentuk kandidat berbasis frame:

\[
SilenceRatio=\frac{N_{\mathrm{silent\ frames}}}{N_{\mathrm{all\ frames}}}
\]

Frame disebut silent jika RMS frame berada di bawah threshold yang ditentukan. Nilai threshold dan ukuran frame harus dicatat.

## E5. Spectral centroid

Untuk spektrum magnitudo \( |X[k]| \) pada frekuensi \(f_k\):

\[
C=\frac{\sum_k f_k |X[k]|}{\sum_k |X[k]|}
\]

Nyatakan apakah dihitung per frame lalu dirata-ratakan, atau dari keseluruhan sinyal. Periksa penggunaan magnitude atau power spectrum.

## E6. Zero-crossing rate

Salah satu bentuk diskret:

\[
ZCR=\frac{1}{2N}\sum_{n=1}^{N-1}
|\operatorname{sgn}(x[n])-\operatorname{sgn}(x[n-1])|
\]

Ada variasi definisi (misalnya pembagi \(N-1\), perlakuan nilai nol, dan perhitungan per-frame). Ikuti implementasi aktual, bukan memilih rumus secara arbitrer.

# F. FORMULA TAMBAHAN YANG MUNGKIN DIPERLUKAN — USULAN, BUKAN IMPLEMENTASI TERVERIFIKASI

Tambahkan hanya bila metode/tujuan penelitian membutuhkannya dan sumber mendukungnya:
1. **Latency**: waktu dari frame/audio diterima hingga prediksi tersedia. Definisikan titik awal/akhir, unit, dan statistik (mean/median/p95).
2. **FPS**: jumlah frame yang diproses per detik; jelaskan apakah termasuk decoding, preprocessing, inference, dan rendering.
3. **False Positive Rate (FPR)**: \(FP/(FP+TN)\), khususnya penting untuk alarm sirene palsu.
4. **Specificity**: \(TN/(TN+FP)\).
5. **Confusion matrix normalized**: jelaskan normalisasi per baris atau kolom.
6. **End-to-end emergency response**: hanya jika ada definisi sistem dan pengukuran yang benar; jangan menciptakan metrik tanpa data.
7. **Fairness/maximum waiting time**: bila klaim penelitian mencakup pengurangan waktu tunggu, harus ada formula dan pengukuran waktu tunggu aktual per lajur.

# G. SUMBER YANG HARUS DIPERIKSA GEMINI

1. Paper `FORMAT FULL PAPER .docx` di repo `Varoo0jpg/Lampu-tes-`.
2. Daftar pustaka paper dan setiap PDF/jurnal/artikel yang dirujuk.
3. Repo `Varoo0jpg/Lampiu-Project` (terutama folder GUI, konfigurasi, dan kode audio).
4. Repo `Varoo0jpg/Lampu-tes-` (paper, catatan, konfigurasi, hasil eksperimen).
5. Notebook audio YAMNet yang menghasilkan embedding, split, model selection, dan test metrics V2.
6. File CSV hasil evaluasi, confusion matrix, model/checkpoint, dan konfigurasi run.
7. Dokumentasi resmi versi library/arsitektur yang benar-benar digunakan (Ultralytics/YOLO26, TensorFlow Hub/YAMNet, scikit-learn), sebagai pelengkap sumber primer.

Jika Gemini tidak dapat mengakses tautan privat atau file biner, minta file diunggah. Jangan menyatakan sudah membaca file yang tidak berhasil diakses.

# H. FORMAT TABEL AUDIT YANG WAJIB DIHASILKAN

Untuk setiap persamaan, isi kolom berikut:

| ID | Topik | Formula original dari sumber | Referensi + halaman/persamaan | Formula di paper | Formula di kode | Status | Jenis perubahan | Dampak | Koreksi/rekomendasi |
|---|---|---|---|---|---|---|---|---|---|

Status yang diperbolehkan:
- Sesuai sumber
- Berbeda notasi saja
- Dimodifikasi dan dijustifikasi
- Dimodifikasi tanpa justifikasi
- Salah implementasi
- Tidak ditemukan di paper
- Tidak ditemukan di kode
- Formula tambahan yang diperlukan
- Belum dapat diverifikasi

Untuk setiap modifikasi, Gemini wajib menampilkan:
1. bentuk original;
2. bentuk yang digunakan saat ini;
3. perbedaan simbol/variabel/operasi;
4. alasan modifikasi yang diklaim;
5. apakah alasan tersebut didukung sumber;
6. dampak terhadap hasil;
7. formula yang direkomendasikan;
8. file/baris kode yang perlu diperbarui (jangan mengubah repo tanpa persetujuan).

# I. PROMPT SIAP PAKAI UNTUK GEMINI

Anda bertindak sebagai auditor metodologi penelitian, auditor matematis, dan auditor implementasi perangkat lunak untuk penelitian Lampiu (sistem pengaturan lampu lalu lintas adaptif, deteksi kendaraan darurat menggunakan YOLO26n, dan pengenalan sirene menggunakan YAMNet dengan classifier).

Gunakan dokumen paper, jurnal/artikel referensi, repo, notebook, konfigurasi, dan hasil eksperimen yang saya berikan. Lakukan audit menyeluruh terhadap seluruh persamaan/formula/rumus.

Ketentuan wajib:
- Mulai dari daftar pustaka paper. Temukan sumber asli setiap formula.
- Jangan mengarang referensi/DOI/halaman/persamaan.
- Bandingkan formula sumber original dengan formula dalam paper dan formula yang benar-benar dijalankan kode.
- Tandai setiap perubahan, modifikasi, penambahan, pengurangan, penyederhanaan, atau perubahan notasi.
- Bedakan perubahan notasi yang ekuivalen dari perubahan matematis yang mengubah metode.
- Verifikasi bahwa formula cocok dengan versi algoritma/library yang digunakan.
- Jika formula salah, jelaskan letak kesalahan dan berikan formula koreksi beserta alasan dan sumber.
- Jika formula penting belum ada, usulkan formula tersebut dan tandai sebagai usulan; jangan mengklaim telah diimplementasikan.
- Verifikasi angka parameter, threshold, satuan, indeks, normalisasi, asumsi, dan batasan.
- Rekonsiliasi eksperimen yang berbeda versi; jangan gabungkan metrik YOLO/audio dari run atau split berbeda.
- Periksa konsistensi paper ↔ referensi ↔ kode ↔ hasil eksperimen.
- Jangan mengubah repository atau file. Berikan patch/rekomendasi saja sampai saya memberi izin.

Output:
1. Ringkasan temuan kritis.
2. Inventaris semua formula dalam tabel audit.
3. Formula original dari sumber beserta sitasi/halaman.
4. Formula yang tertulis di paper.
5. Formula yang benar-benar dijalankan kode (sertakan file dan baris).
6. Daftar perubahan/modifikasi dan dampaknya.
7. Formula yang hilang dan perlu ditambahkan, dengan justifikasi sumber.
8. Daftar koreksi prioritas tinggi/sedang/rendah.
9. Versi formula final yang konsisten untuk paper.
10. Daftar pertanyaan/berkas yang masih dibutuhkan untuk menutup gap audit.

Jangan menganggap daftar formula dalam dokumen handoff ini sudah final. Dokumen ini adalah peta awal untuk audit dan harus dikoreksi berdasarkan bukti primer.
