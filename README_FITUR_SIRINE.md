# Pengenalan Suara Sirene — GUI Terintegrasi

Fitur pengenalan audio telah digabungkan langsung ke `Coba_kamera.py`, bersama
GUI dan deteksi kendaraan YOLO. Tidak diperlukan file Python sirene terpisah.

## Input yang didukung

- Kamera: sumber YouTube Live CCTV dikonfigurasi pada `SUMBER_KAMERA` di
  `Coba_kamera.py` dan diakses melalui `yt-dlp`.
- Audio mikrofon: pilih **Aktifkan Deteksi Sirene** di GUI.
- Audio berkas: pilih **Pilih MP3/WAV untuk Uji** untuk menganalisis berkas
  audio secara offline. Berkas audio diproses dalam potongan dan skor sirene
  ditampilkan pada GUI.

## Instalasi dependensi

Jalankan di lingkungan Python proyek:

```bash
python -m pip install -r requirements-sirine.txt
```

Pastikan dependensi aplikasi utama juga terpasang. Model YAMNet akan diunduh
otomatis saat pertama kali deteksi audio dimulai; koneksi internet diperlukan
untuk pengunduhan awal.

## Batasan untuk evaluasi

YAMNet mengklasifikasikan kelas suara seperti "Siren"; hasilnya bukan identifikasi
terverifikasi bahwa suara berasal dari ambulans, polisi, atau pemadam kebakaran.
Gunakan berkas positif dan negatif berlabel untuk menghitung precision, recall,
F1-score, false positive rate, dan latensi. Jangan menyimpulkan akurasi hanya
dari satu atau dua contoh. Skor model bukan probabilitas terkalibrasi kecuali
dikalibrasi melalui evaluasi tersendiri.
