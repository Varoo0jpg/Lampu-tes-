# Fitur Pengenalan Suara Sirene

Fitur ini menambahkan deteksi suara **sirene** ke Smart Traffic Control System.
Detektor menggunakan model audio YAMNet melalui MediaPipe dan mikrofon komputer.

## Instalasi

Di lingkungan Python yang sama dengan aplikasi:

```bash
python -m pip install -r requirements-sirine.txt
```

Jalankan aplikasi seperti biasa. Tekan **Aktifkan Deteksi Sirine** untuk mulai
menggunakan mikrofon. Saat pertama kali dijalankan, aplikasi mengunduh model
YAMNet ke file `yamnet.tflite`; koneksi internet diperlukan hanya untuk
pengunduhan awal.

## Cara kerja dan batasan

- Audio dianalisis pada potongan pendek secara berulang di thread terpisah.
- Sirene dilaporkan setelah dua potongan berturut-turut melewati ambang skor.
- Model mendeteksi kelas suara "Siren"; model ini **tidak membedakan ambulans,
  polisi, dan pemadam kebakaran secara andal**.
- Mikrofon tunggal tidak memberi informasi arah yang cukup untuk menentukan
  lajur asal sirene. Setelah peringatan muncul, operator harus mengonfirmasi
  arah/lajur secara visual dan menekan tombol **Prioritas Darurat** yang sesuai.
- Fitur ini adalah prototipe penelitian, bukan sistem kendali lalu lintas siap
  pakai. Uji dengan data rekaman berlabel dan lakukan pengawasan manusia.
- Audio tidak disimpan oleh kode ini.

## Evaluasi penelitian

Uji pada kumpulan rekaman sirene dan suara non-sirene yang mencakup kebisingan
lalu lintas. Catat precision, recall, F1-score, false alarm per menit, dan waktu
deteksi. Pisahkan data latih/validasi/uji jika model dikembangkan lebih lanjut.
