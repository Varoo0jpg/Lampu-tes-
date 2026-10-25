# Coba_kamera.py
# GUI CustomTkinter (Coba.py) + deteksi kendaraan 4 kamera (YOLO)
#
# Instalasi:
#   pip install customtkinter pillow opencv-python ultralytics numpy
#
# Penanda perubahan terhadap Coba.py asli:
#   [BARU]       = bagian tambahan (deteksi kamera, tampilan feed)
#   [PERBAIKAN]  = bug pada kode asli yang diperbaiki
import collections
import os
import random
import sys
import threading
import time

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

# ===============================================================
# KONFIGURASI SISTEM
# ===============================================================
DAFTAR_LAJUR = ["Utara", "Timur", "Selatan", "Barat"]

BOBOT_KENDARAAN = {
    "Motor": 0.25,
    "Mobil": 1.0,
    "Truk": 2.5,
    "Bus": 2.0
}

FASE_HIJAU_MIN = 10     # Batas minimal waktu hijau (detik)
FASE_HIJAU_MAX = 40     # Batas maksimal waktu hijau (detik)
FASE_KUNING = 3         # Durasi lampu kuning (detik)

# Kode Warna UI Hexadecimal
HEX_RED = "#e74c3c"
HEX_YELLOW = "#f1c40f"
HEX_GREEN = "#2ecc71"
HEX_DARK_BG = "#1e2530"

# ---------------- [BARU] Konfigurasi kamera & deteksi ----------------
# Sumber tiap lajur: indeks kamera (0,1,2,..), path file video, atau None (input manual).
# Default: hanya kamera laptop untuk lajur Utara.
SUMBER_KAMERA = {"Utara": "https://www.youtube.com/watch?v=Ck45cxJAh-s", "Timur": "https://www.youtube.com/watch?v=u-8iSRWgi_c", "Selatan": "https://www.youtube.com/live/BdPvBNLy5kc?si=tm8jdiBL6TW7jkJy", "Barat": "https://www.youtube.com/watch?v=D_R5EpqQIpo"}
# Contoh 4 kamera USB:
# SUMBER_KAMERA = {"Utara": 0, "Timur": 1, "Selatan": 2, "Barat": 3}
# Contoh 4 file video:
# SUMBER_KAMERA = {"Utara": "utara.mp4", "Timur": "timur.mp4",
#                  "Selatan": "selatan.mp4", "Barat": "barat.mp4"}


def resource_path(nama):
    """Lokasi file bawaan; juga bekerja saat dibungkus PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, nama)


MODEL_PATH = resource_path("yolo26n.pt")   # YOLO26 nano; alternatif: yolo26s.pt (lebih akurat, lebih berat)
CONF = 0.35
IMGSZ = 416                      # kecil = lebih cepat di CPU
UKURAN_FRAME = (640, 480)        # ukuran proses deteksi
UKURAN_TAMPIL = (330, 248)       # ukuran feed di kotak kamera GUI

# Kelas COCO -> jenis kendaraan pada panel input
KELAS_YOLO = {2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}

# ROI per lajur (pecahan frame): area antrean di depan kamera. SESUAIKAN per kamera.
ROI_DEFAULT = [(0.15, 0.30), (0.85, 0.30), (0.98, 1.00), (0.02, 1.00)]
ROI = {l: list(ROI_DEFAULT) for l in DAFTAR_LAJUR}
# False = hitung kendaraan di SELURUH frame kamera (ROI di atas diabaikan)
# True  = hanya hitung kendaraan di dalam poligon ROI (garis kuning)
GUNAKAN_ROI = False

# Pemilihan lajur hijau berikutnya pada mode OTOMATIS:
#   "PADAT"    = lajur dengan skor tertinggi didahulukan (dengan batas tunggu maksimum)
#   "BERGILIR" = urutan tetap Utara -> Timur -> Selatan -> Barat
PILIH_LAJUR_BERIKUT = "PADAT"
BATAS_TUNGGU_MAKS = 180          # detik; lajur yang menunggu lebih lama dipaksa mendapat giliran


# ===============================================================
# [BARU] SUBSISTEM: DETEKSI KENDARAAN DARI KAMERA (THREAD)
# ===============================================================
def resolve_sumber(sumber):
    """[BARU] Ubah link YouTube (live/video) menjadi URL stream langsung via yt-dlp.
    Sumber lain (indeks kamera, file video, None) dikembalikan apa adanya."""
    if isinstance(sumber, str) and ("youtube.com" in sumber or "youtu.be" in sumber):
        import yt_dlp
        # Audio tidak dipakai, jadi stream video-saja (bv*) juga boleh
        opsi = {"quiet": True, "noplaylist": True,
                "format": "bv*[height<=480]/b[height<=480]/bv*/b"}
        with yt_dlp.YoutubeDL(opsi) as ydl:
            info = ydl.extract_info(sumber, download=False)
        return info["url"]
    return sumber


class DetektorKamera(threading.Thread):
    """Membaca semua kamera, deteksi YOLO (batch), hitung kendaraan per jenis dalam ROI."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.running = True
        self.siap = False          # True setelah model dimuat & kamera dicoba dibuka
        self.error = None
        self.caps = {}
        self.frames = {l: None for l in DAFTAR_LAJUR}
        self.jumlah = {l: {j: 0 for j in BOBOT_KENDARAAN} for l in DAFTAR_LAJUR}
        self._riwayat = {l: collections.deque(maxlen=5) for l in DAFTAR_LAJUR}

    def _buka_sumber(self):
        for lajur, sumber in SUMBER_KAMERA.items():
            if sumber is None:
                continue
            try:
                sumber_nyata = resolve_sumber(sumber)
            except Exception as e:
                print(f"[!] Gagal membaca link YouTube lajur {lajur}: {e}")
                continue
            if isinstance(sumber_nyata, int) and os.name == "nt":
                cap = cv2.VideoCapture(sumber_nyata, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(sumber_nyata)
            if cap.isOpened():
                self.caps[lajur] = cap
            else:
                print(f"[!] Sumber lajur {lajur} ({sumber}) tidak terbuka")

    def run(self):
        try:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
            self._buka_sumber()
        except Exception as e:
            self.error = repr(e)
            self.siap = True
            return
        self.siap = True

        w, h = UKURAN_FRAME
        while self.running and self.caps:
            imgs = {}
            for lajur, cap in self.caps.items():
                ok, img = cap.read()
                if not ok:
                    if isinstance(SUMBER_KAMERA[lajur], str):     # file video: ulang
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                imgs[lajur] = cv2.resize(img, UKURAN_FRAME)
            if not imgs:
                time.sleep(0.02)
                continue

            daftar = list(imgs)
            hasil = model([imgs[l] for l in daftar], conf=CONF, imgsz=IMGSZ,
                          classes=list(KELAS_YOLO), verbose=False)

            for lajur, res in zip(daftar, hasil):
                img = imgs[lajur]
                poly = np.array([(int(x * w), int(y * h)) for x, y in ROI[lajur]], np.int32)
                hitung = {j: 0 for j in BOBOT_KENDARAAN}
                for box in res.boxes:
                    jenis = KELAS_YOLO[int(box.cls[0])]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dalam = (not GUNAKAN_ROI) or cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
                    warna = (0, 255, 0) if dalam else (120, 120, 120)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), warna, 2)
                    if dalam:
                        hitung[jenis] += 1
                if GUNAKAN_ROI:
                    cv2.polylines(img, [poly], True, (0, 255, 255), 2)

                # median 5 frame terakhir agar angka tidak berkedip
                rw = self._riwayat[lajur]
                rw.append(hitung)
                median = {j: int(round(float(np.median([r[j] for r in rw]))))
                          for j in BOBOT_KENDARAAN}
                teks = " ".join(f"{j[:2]}:{n}" for j, n in median.items())
                cv2.putText(img, f"{lajur}  {teks}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                with self.lock:
                    self.jumlah[lajur] = median
                    self.frames[lajur] = img

    def ambil(self):
        with self.lock:
            jumlah = {l: dict(v) for l, v in self.jumlah.items()}
            frames = {l: (None if f is None else f.copy()) for l, f in self.frames.items()}
        return jumlah, frames

    def stop(self):
        self.running = False
        time.sleep(0.1)
        for cap in self.caps.values():
            cap.release()


# ===============================================================
# SUBSISTEM: INPUT DATA KENDARAAN (LAJUR PANEL CLASS)
# ===============================================================
class PanelInputLajur(ctk.CTkFrame):
    def __init__(self, parent, nama_lajur):
        super().__init__(parent, fg_color="#2d2d2d")
        self.nama_lajur = nama_lajur
        self.dikunci = False      # [BARU] True jika angka diisi otomatis oleh kamera

        # Label Title Lajur
        ctk.CTkLabel(self, text=f"Data Kendaraan - {nama_lajur}", font=("Arial", 11, "bold"), text_color="#FFCC00").pack(pady=2)

        # Frame Kontainer Kotak Input
        grid_input = ctk.CTkFrame(self, fg_color="transparent")
        grid_input.pack(fill="x", padx=10, pady=2)
        grid_input.columnconfigure((0, 1, 2, 3), weight=1)

        # Simpan objek widget entry ke dalam dictionary
        self.kolom_input = {}

        for indeks, jenis in enumerate(BOBOT_KENDARAAN.keys()):
            ctk.CTkLabel(grid_input, text=jenis, font=("Arial", 10)).grid(row=0, column=indeks, padx=2, pady=1)

            kotak_teks = ctk.CTkEntry(grid_input, width=45, justify="center")
            kotak_teks.insert(0, str(random.randint(0, 10)))  # Nilai acak awal
            kotak_teks.grid(row=1, column=indeks, padx=2, pady=2)
            self.kolom_input[jenis] = kotak_teks

    def ambil_skor_bobot(self):
        """Menghitung total beban PCU lajur dari angka input user."""
        total_skor = 0.0
        for jenis, bobot in BOBOT_KENDARAAN.items():
            try:
                jumlah = int(self.kolom_input[jenis].get())
                total_skor += jumlah * bobot
            except ValueError:
                total_skor += 0.0
        return round(total_skor, 2)

    def acak_jumlah_kendaraan(self):
        """Mengisi kolom input dengan angka tiruan acak baru."""
        if self.dikunci:      # [BARU] lajur berkamera tidak diacak
            return
        for kotak_teks in self.kolom_input.values():
            kotak_teks.delete(0, "end")
            kotak_teks.insert(0, str(random.randint(0, 20)))

    # ---------------- [BARU] ----------------
    def set_jumlah(self, data):
        """Mengisi kolom dari hasil deteksi, mis. {'Motor': 3, 'Mobil': 2, ...}."""
        for jenis, kotak in self.kolom_input.items():
            kotak.configure(state="normal")
            kotak.delete(0, "end")
            kotak.insert(0, str(data.get(jenis, 0)))
            if self.dikunci:
                kotak.configure(state="disabled")

    def kunci(self, status=True):
        self.dikunci = status
        for kotak in self.kolom_input.values():
            kotak.configure(state="disabled" if status else "normal")


# ===============================================================
# APLIKASI UTAMA: ENGINE GUI & CORE LOGIC SIMULATION
# ===============================================================
class TrafficGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup Window Utama Aplikasi
        self.title("Smart Traffic Control System - Deteksi Kamera")
        self.geometry("1450x850")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # --- STATE ENGINE SIMULASI INTERNAL ---
        self.simulasi_aktif = False
        self.fase_saat_ini = "HIJAU"     # Status: "HIJAU" atau "KUNING"
        self.lajur_aktif = "Utara"       # Menunjukkan lajur mana yang sedang hijau
        self.hitung_mundur = 10          # Durasi detik yang tersisa
        self.lajur_terpaksa = None       # Menyimpan antrean paksa lajur dari mode manual

        # [BARU] State kamera & penjadwalan
        self._id_ticker = None
        self.waktu_hijau_terakhir = {l: time.time() for l in DAFTAR_LAJUR}
        self.detektor = None
        self.kotak_kamera = {}
        self.label_kamera = {}
        self.gambar_kamera = {}
        self.jumlah_terakhir = {}
        self._lajur_gagal = set()

        # Konfigurasi Master Grid Layout
        self.grid_columnconfigure(0, weight=0, minsize=360)  # Kolom Kontrol Kiri
        self.grid_columnconfigure(1, weight=1)               # Kolom Grafik Tengah
        self.grid_columnconfigure(2, weight=1)               # Kolom Kamera & Data
        self.grid_columnconfigure(3, weight=1)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0, minsize=120)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0, minsize=120)

        # Bangun Komponen Tampilan
        self.bangun_panel_kontrol_kiri()
        self.bangun_grid_simulasi_kanan()

        # [BARU] Jalankan deteksi kamera
        self.mulai_deteksi()
        self.protocol("WM_DELETE_WINDOW", self.tutup_aplikasi)

        # Visualisasi awal status lampu
        self.sinkronisasi_tampilan_ui()

    # -----------------------------------------------------------
    # TATA LETAK UI & WIDGET GENERATOR
    # -----------------------------------------------------------
    def bangun_panel_kontrol_kiri(self):
        """Membangun bilah kendali simulasi di area sebelah kiri."""
        panel_kiri = ctk.CTkFrame(self, width=340, corner_radius=0, fg_color="#1e1e1e")
        panel_kiri.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(panel_kiri, text="PANEL KENDALI UTAMA", font=("Arial", 16, "bold"), text_color="#FFCC00").pack(pady=15)

        # Wadah Indikator Fisik Lampu
        bingkai_lampu = ctk.CTkFrame(panel_kiri, fg_color="#2b2b2b", border_width=1)
        bingkai_lampu.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(bingkai_lampu, text="Status Lampu Lalu Lintas", font=("Arial", 12, "bold")).pack(pady=5)

        self.visual_lampu = {}
        for lajur in DAFTAR_LAJUR:
            baris = ctk.CTkFrame(bingkai_lampu, fg_color="transparent")
            baris.pack(fill="x", padx=20, pady=4)

            ctk.CTkLabel(baris, text=f"Lajur {lajur}: ", width=100, anchor="w").pack(side="left")

            bulatan = ctk.CTkFrame(baris, width=20, height=20, corner_radius=10, fg_color=HEX_RED)
            bulatan.pack(side="left", padx=10)
            self.visual_lampu[lajur] = bulatan

        # Papan Informasi Countdown Status
        bingkai_info = ctk.CTkFrame(panel_kiri, fg_color="#151515", border_width=1, border_color="#333333")
        bingkai_info.pack(fill="x", padx=15, pady=15)

        self.txt_status = ctk.CTkLabel(bingkai_info, text="Status: Non-Aktif", font=("Arial", 14, "bold"), text_color="yellow")
        self.txt_status.pack(pady=5)

        self.txt_timer = ctk.CTkLabel(bingkai_info, text="-- dtk", font=("Arial", 28, "bold"), text_color="#ffffff")
        self.txt_timer.pack(pady=5)

        # Pilihan Mode Operasional
        self.var_mode = ctk.StringVar(value="OTOMATIS")
        bingkai_mode = ctk.CTkFrame(panel_kiri, fg_color="transparent")
        bingkai_mode.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(bingkai_mode, text="Mode Kendali:", font=("Arial", 11, "bold")).pack(anchor="w", padx=5)
        ctk.CTkRadioButton(bingkai_mode, text="Otomatis (Adaptif PCU)", variable=self.var_mode, value="OTOMATIS").pack(anchor="w", padx=10, pady=2)
        ctk.CTkRadioButton(bingkai_mode, text="Manual (Paksa Klik)", variable=self.var_mode, value="MANUAL").pack(anchor="w", padx=10, pady=2)

        # Tombol Aksi Kendali
        self.btn_mulai = ctk.CTkButton(panel_kiri, text="Mulai Simulasi", fg_color="#27ae60", hover_color="#1e8449", command=self.action_mulai_simulasi)
        self.btn_mulai.pack(fill="x", padx=15, pady=5)

        self.btn_stop = ctk.CTkButton(panel_kiri, text="Hentikan Simulasi", fg_color="#c0392b", hover_color="#962d22", command=self.action_hentikan_simulasi, state="disabled")
        self.btn_stop.pack(fill="x", padx=15, pady=5)

        self.btn_acak = ctk.CTkButton(panel_kiri, text="Acak Lajur Tanpa Kamera", fg_color="#2980b9", hover_color="#1f618d", command=self.action_acak_input_kendaraan)
        self.btn_acak.pack(fill="x", padx=15, pady=5)

    def bangun_grid_simulasi_kanan(self):
        """Membangun penempatan layout grid 4 kamera dan grafik suara."""
        # --- KOLOM 1: WADAH GRAFIK ANALISIS (FRAME POLOS) ---
        f_suara1 = ctk.CTkFrame(self, fg_color=HEX_DARK_BG, border_width=1)
        f_suara1.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(f_suara1, text="[ Frame Data Suara 1 ]\n(Spectrogram Audio)", text_color="#5dade2").pack(expand=True)

        f_akustik1 = ctk.CTkFrame(self, fg_color="#151a21")
        f_akustik1.grid(row=1, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(f_akustik1, text="Data Akustik 1", text_color="gray").pack(expand=True)

        f_suara2 = ctk.CTkFrame(self, fg_color=HEX_DARK_BG, border_width=1)
        f_suara2.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(f_suara2, text="[ Frame Data Suara 2 ]\n(Spectrogram Audio)", text_color="#5dade2").pack(expand=True)

        f_akustik2 = ctk.CTkFrame(self, fg_color="#151a21")
        f_akustik2.grid(row=3, column=1, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(f_akustik2, text="Data Akustik 2", text_color="gray").pack(expand=True)

        # --- KOLOM 2 & 3: BLOK MONITOR KAMERA & DATA INPUT LAJUR ---
        self.kumpulan_panel = {}

        # Kamera 1 - Utara
        self.buat_box_kamera_simulasi(row=0, col=2, nomor=1, nama="Utara")
        self.kumpulan_panel["Utara"] = PanelInputLajur(self, "Utara")
        self.kumpulan_panel["Utara"].grid(row=1, column=2, sticky="nsew", padx=4, pady=4)

        # Kamera 2 - Timur
        self.buat_box_kamera_simulasi(row=0, col=3, nomor=2, nama="Timur")
        self.kumpulan_panel["Timur"] = PanelInputLajur(self, "Timur")
        self.kumpulan_panel["Timur"].grid(row=1, column=3, sticky="nsew", padx=4, pady=4)

        # Kamera 3 - Selatan
        self.buat_box_kamera_simulasi(row=2, col=2, nomor=3, nama="Selatan")
        self.kumpulan_panel["Selatan"] = PanelInputLajur(self, "Selatan")
        self.kumpulan_panel["Selatan"].grid(row=3, column=2, sticky="nsew", padx=4, pady=4)

        # Kamera 4 - Barat
        self.buat_box_kamera_simulasi(row=2, col=3, nomor=4, nama="Barat")
        self.kumpulan_panel["Barat"] = PanelInputLajur(self, "Barat")
        self.kumpulan_panel["Barat"].grid(row=3, column=3, sticky="nsew", padx=4, pady=4)

        # Pasang Tombol Paksa Manual di bagian bawah tiap-tiap blok data lajur
        for lajur in DAFTAR_LAJUR:
            btn_paksa = ctk.CTkButton(
                self.kumpulan_panel[lajur],
                text=f"Paksa Hijau {lajur}",
                width=120,
                height=22,
                fg_color="#34495e",
                command=lambda l=lajur: self.action_pemicu_manual_lajur(l)
            )
            btn_paksa.pack(pady=4)

    def buat_box_kamera_simulasi(self, row, col, nomor, nama):
        f = ctk.CTkFrame(self, fg_color="#242424", border_width=2, border_color="#555555")
        f.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        lbl = ctk.CTkLabel(f, text=f"CAMERA {nomor}\n(Lajur {nama})", font=("Arial", 14, "bold"))
        lbl.pack(expand=True)
        self.kotak_kamera[nama] = f      # [BARU]
        self.label_kamera[nama] = lbl    # [BARU]

    # -----------------------------------------------------------
    # [BARU] INTEGRASI KAMERA
    # -----------------------------------------------------------
    def _teks_kamera(self, lajur, pesan):
        nomor = DAFTAR_LAJUR.index(lajur) + 1
        return f"CAMERA {nomor}\n(Lajur {lajur})\n{pesan}"

    def mulai_deteksi(self):
        self.lajur_berkamera = [l for l in DAFTAR_LAJUR if SUMBER_KAMERA[l] is not None]
        if not self.lajur_berkamera:
            return
        for l in self.lajur_berkamera:
            panel = self.kumpulan_panel[l]
            panel.set_jumlah({})            # mulai dari nol
            panel.kunci(True)               # isian diatur kamera
            self.label_kamera[l].configure(text=self._teks_kamera(l, "Memuat model YOLO..."))
        self.detektor = DetektorKamera()
        self.detektor.start()
        self.after(200, self.poll_kamera)

    def poll_kamera(self):
        """Dipanggil berkala di thread GUI: ambil hasil deteksi, perbarui feed dan angka."""
        det = self.detektor
        if det is None:
            return
        if det.siap:
            if det.error:
                for l in self.lajur_berkamera:
                    self.label_kamera[l].configure(image=None, text=self._teks_kamera(l, "Gagal memuat deteksi\n(input manual)"))
                    self.kumpulan_panel[l].kunci(False)
                print("[!] Detektor gagal:", det.error)
                return
            jumlah, frames = det.ambil()
            for l in self.lajur_berkamera:
                panel = self.kumpulan_panel[l]
                if l not in det.caps:
                    if l not in self._lajur_gagal:
                        self._lajur_gagal.add(l)
                        self.label_kamera[l].configure(text=self._teks_kamera(l, "Kamera tidak terbuka\n(input manual)"))
                        panel.kunci(False)
                    continue
                frame = frames[l]
                if frame is not None:
                    rgb = cv2.cvtColor(cv2.resize(frame, UKURAN_TAMPIL), cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    if l not in self.gambar_kamera:
                        self.gambar_kamera[l] = ctk.CTkImage(light_image=pil, dark_image=pil, size=UKURAN_TAMPIL)
                        self.label_kamera[l].configure(image=self.gambar_kamera[l], text="")
                    else:
                        self.gambar_kamera[l].configure(light_image=pil, dark_image=pil)
                if jumlah[l] != self.jumlah_terakhir.get(l):
                    self.jumlah_terakhir[l] = jumlah[l]
                    panel.set_jumlah(jumlah[l])
        self.after(120, self.poll_kamera)

    def tutup_aplikasi(self):
        if self._id_ticker is not None:
            self.after_cancel(self._id_ticker)
        if self.detektor is not None:
            self.detektor.stop()
        self.destroy()

    # -----------------------------------------------------------
    # ENGINE CORE SIMULATION LOGIC
    # -----------------------------------------------------------
    def tentukan_lajur_paling_padat(self, lajur_kecuali):
        """Mencari lajur dengan antrean paling padat untuk giliran berikutnya.
        [PERBAIKAN] Jika skor sama, pilih yang paling lama menunggu (versi asli selalu
        memilih kandidat pertama sehingga Selatan/Barat bisa tidak pernah dapat giliran)."""
        kandidat = [l for l in DAFTAR_LAJUR if l != lajur_kecuali]
        now = time.time()
        return max(kandidat, key=lambda l: (self.kumpulan_panel[l].ambil_skor_bobot(),
                                            now - self.waktu_hijau_terakhir[l]))

    def hitung_durasi_fase_hijau(self, lajur_target):
        """
        Menghitung durasi lampu hijau secara proporsional adaptif.
        Catatan:
        1. Waktu total dari keempat simpang pasti selalu 120 detik.
        2. Waktu masing-masing simpang ditentukan dari perhitungan value (skor) setiap simpang.
        ([PERBAIKAN] versi duplikat pertama yang tidak terpakai dihapus.)
        """
        skor_simpang = {l: self.kumpulan_panel[l].ambil_skor_bobot() for l in DAFTAR_LAJUR}

        total_skor = sum(skor_simpang.values())
        TOTAL_WAKTU_SIKLUS = 120

        # KONDISI 1: Jika semua simpang memiliki value sama atau kosong total (0)
        if len(set(skor_simpang.values())) == 1 or total_skor == 0:
            return 30  # 30 dtk x 4 simpang = 120 detik bersih

        # KONDISI 2: Jika value bervariasi (dihitung proporsional dari value masing-masing)
        alokasi_waktu = {}
        waktu_terdistribusi = 0

        for i, lajur in enumerate(DAFTAR_LAJUR):
            if i == len(DAFTAR_LAJUR) - 1:
                # Simpang terakhir mengambil sisa waktu agar total siklus tepat 120 detik
                # [PERBAIKAN] diberi batas minimal 5 dtk; versi asli bisa menghasilkan 0 atau negatif
                alokasi_waktu[lajur] = max(TOTAL_WAKTU_SIKLUS - waktu_terdistribusi, 5)
            else:
                porsi_waktu = int(round((skor_simpang[lajur] / total_skor) * TOTAL_WAKTU_SIKLUS))
                porsi_waktu = max(porsi_waktu, 5)   # proteksi: minimal 5 detik
                alokasi_waktu[lajur] = porsi_waktu
                waktu_terdistribusi += porsi_waktu

        return alokasi_waktu[lajur_target]

    def pilih_lajur_berikut(self):
        """[BARU] Menentukan lajur hijau berikutnya setelah fase kuning selesai."""
        if self.var_mode.get() == "MANUAL" or PILIH_LAJUR_BERIKUT == "BERGILIR":
            i = DAFTAR_LAJUR.index(self.lajur_aktif)
            return DAFTAR_LAJUR[(i + 1) % len(DAFTAR_LAJUR)]

        # OTOMATIS: anti-starvation dulu, lalu yang terpadat
        now = time.time()
        kandidat = [l for l in DAFTAR_LAJUR if l != self.lajur_aktif]
        terlama = max(kandidat, key=lambda l: now - self.waktu_hijau_terakhir[l])
        if now - self.waktu_hijau_terakhir[terlama] > BATAS_TUNGGU_MAKS:
            return terlama
        return self.tentukan_lajur_paling_padat(self.lajur_aktif)

    def siklus_transisi_fase(self):
        """[PERBAIKAN] Method ini dipanggil ticker tetapi tidak ada di kode asli
        (akan menyebabkan AttributeError saat hitung mundur habis)."""
        if self.fase_saat_ini == "HIJAU":
            self.fase_saat_ini = "KUNING"
            self.hitung_mundur = FASE_KUNING
        else:
            # kuning selesai -> pindah ke lajur berikutnya
            self.waktu_hijau_terakhir[self.lajur_aktif] = time.time()
            if self.lajur_terpaksa:
                lajur_baru = self.lajur_terpaksa
                self.lajur_terpaksa = None
            else:
                lajur_baru = self.pilih_lajur_berikut()
            self.lajur_aktif = lajur_baru
            self.fase_saat_ini = "HIJAU"
            self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)

    def sinkronisasi_tampilan_ui(self):
        """Memperbarui warna bulatan lampu, countdown, dan bingkai kamera."""
        for lajur, widget in self.visual_lampu.items():
            if lajur == self.lajur_aktif:
                warna = HEX_GREEN if self.fase_saat_ini == "HIJAU" else HEX_YELLOW
            else:
                warna = HEX_RED
            widget.configure(fg_color=warna)
            # [BARU] bingkai kamera mengikuti warna lampu
            self.kotak_kamera[lajur].configure(border_color=warna if self.simulasi_aktif else "#555555")

        if self.simulasi_aktif:
            warna_teks = HEX_GREEN if self.fase_saat_ini == "HIJAU" else HEX_YELLOW
            self.txt_status.configure(text=f"Status: {self.fase_saat_ini} @ {self.lajur_aktif}", text_color=warna_teks)
            self.txt_timer.configure(text=f"{max(self.hitung_mundur, 0)} dtk", text_color="#ffffff")
        else:
            self.txt_status.configure(text="Simulasi Dihentikan", text_color="orange")
            self.txt_timer.configure(text="-- dtk", text_color="gray")

    # -----------------------------------------------------------
    # INTERACTION & TIMING EVENTS
    # -----------------------------------------------------------
    def engine_simulation_ticker(self):
        """Ticker mandiri yang berjalan mundur 1 detik secara periodik."""
        if not self.simulasi_aktif:
            return

        self.hitung_mundur -= 1

        if self.hitung_mundur <= 0:
            self.siklus_transisi_fase()

        self.sinkronisasi_tampilan_ui()

        # [PERBAIKAN] id disimpan agar bisa dibatalkan (hindari ticker ganda saat stop lalu start cepat)
        self._id_ticker = self.after(1000, self.engine_simulation_ticker)

    def action_mulai_simulasi(self):
        if self.simulasi_aktif:
            return
        self.simulasi_aktif = True
        self.btn_mulai.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_acak.configure(state="disabled")

        # Reset fase awal dan hitung durasi hijau lajur aktif dari data saat ini
        self.fase_saat_ini = "HIJAU"
        self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)

        self.sinkronisasi_tampilan_ui()
        self._id_ticker = self.after(1000, self.engine_simulation_ticker)

    def action_hentikan_simulasi(self):
        self.simulasi_aktif = False
        if self._id_ticker is not None:
            self.after_cancel(self._id_ticker)
            self._id_ticker = None
        self.btn_mulai.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_acak.configure(state="normal")
        self.sinkronisasi_tampilan_ui()

    def action_acak_input_kendaraan(self):
        for panel in self.kumpulan_panel.values():
            panel.acak_jumlah_kendaraan()      # lajur berkamera dilewati
        if self.fase_saat_ini == "HIJAU" and self.simulasi_aktif:
            self.hitung_mundur = self.hitung_durasi_fase_hijau(self.lajur_aktif)
            self.txt_timer.configure(text=f"{self.hitung_mundur} dtk")

    def action_pemicu_manual_lajur(self, lajur_tujuan):
        """Memaksa sistem memindahkan lampu hijau ke lajur pilihan (hanya Mode Manual)."""
        if self.var_mode.get() != "MANUAL" or not self.simulasi_aktif:
            return

        # Jika lajur yang diklik sudah sedang hijau, abaikan
        if lajur_tujuan == self.lajur_aktif and self.fase_saat_ini == "HIJAU":
            return

        # Pemicu transisi kuning terlebih dahulu sebelum pindah
        self.fase_saat_ini = "KUNING"
        self.hitung_mundur = FASE_KUNING
        self.lajur_terpaksa = lajur_tujuan
        self.sinkronisasi_tampilan_ui()


# ===============================================================
# APLIKASI RUNNER (MAIN LAUNCHER)
# ===============================================================
if __name__ == "__main__":
    app = TrafficGUI()
    app.mainloop()
