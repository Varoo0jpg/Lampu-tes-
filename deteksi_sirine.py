"""Deteksi suara sirene untuk Smart Traffic Control System.

Menggunakan YAMNet melalui MediaPipe Audio Classifier. Model mengenali kelas
suara "Siren", bukan jenis kendaraan atau arah datangnya suara. Karena itu,
sistem meminta operator memilih lajur setelah sirene terdeteksi.
"""
from __future__ import annotations

import os
import threading
import time
import urllib.request

SAMPLE_RATE = 16_000
DURASI_AUDIO = 1.25
AMBANG_SIRENE = 0.25
DETEKSI_BERTURUT = 2
JEDA_ALARM = 8.0
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "audio_classifier/yamnet/float32/1/yamnet.tflite"
)


def _model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamnet.tflite")


class DetektorSirine(threading.Thread):
    """Mendengarkan mikrofon dalam thread terpisah agar GUI tidak macet."""

    def __init__(self, kirim_event):
        super().__init__(daemon=True, name="DetektorSirine")
        self.kirim_event = kirim_event
        self.running = True
        self._stop_event = threading.Event()

    def stop(self):
        self.running = False
        self._stop_event.set()

    def _kirim(self, jenis, **data):
        try:
            self.kirim_event({"jenis": jenis, **data})
        except Exception:
            pass

    @staticmethod
    def _siapkan_model():
        path = _model_path()
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".download"
            request = urllib.request.Request(
                MODEL_URL, headers={"User-Agent": "LampuTes-SirenDetector/1.0"}
            )
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    with open(tmp_path, "wb") as output:
                        output.write(response.read())
                os.replace(tmp_path, path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        return path

    def run(self):
        try:
            import numpy as np
            import sounddevice as sd
            import mediapipe as mp

            self._kirim("status", pesan="Menyiapkan model pengenalan suara...")
            model_path = self._siapkan_model()
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.audio.AudioClassifierOptions(
                base_options=base_options,
                max_results=10,
                score_threshold=0.0,
            )

            with mp.tasks.audio.AudioClassifier.create_from_options(options) as classifier:
                self._kirim("status", pesan="Mendengarkan mikrofon — deteksi sirene aktif")
                positif_berturut = 0
                alarm_terakhir = 0.0

                while self.running and not self._stop_event.is_set():
                    try:
                        rekaman = sd.rec(
                            int(DURASI_AUDIO * SAMPLE_RATE),
                            samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype="float32",
                            blocking=True,
                        )
                        if self._stop_event.is_set():
                            break

                        gelombang = np.asarray(rekaman, dtype=np.float32).reshape(-1)
                        audio_data = mp.tasks.audio.AudioData.create_from_array(
                            gelombang, SAMPLE_RATE
                        )
                        hasil = classifier.classify(audio_data)
                        skor_sirene = 0.0
                        for klasifikasi in hasil.classifications:
                            for kategori in klasifikasi.categories:
                                nama = (kategori.category_name or kategori.display_name or "").lower()
                                if "siren" in nama or "emergency vehicle" in nama:
                                    skor_sirene = max(skor_sirene, float(kategori.score))

                        if skor_sirene >= AMBANG_SIRENE:
                            positif_berturut += 1
                        else:
                            positif_berturut = 0

                        sekarang = time.monotonic()
                        if (
                            positif_berturut >= DETEKSI_BERTURUT
                            and sekarang - alarm_terakhir >= JEDA_ALARM
                        ):
                            alarm_terakhir = sekarang
                            positif_berturut = 0
                            self._kirim(
                                "sirene",
                                skor=round(skor_sirene, 3),
                                pesan="Sirene terdeteksi. Konfirmasi arah/lajur secara visual.",
                            )
                    except Exception as exc:
                        self._kirim("error", pesan=f"Kesalahan input audio: {exc}")
                        time.sleep(1.0)

            self._kirim("status", pesan="Deteksi sirene berhenti")
        except Exception as exc:
            self._kirim(
                "error",
                pesan=(
                    f"Deteksi sirene gagal: {exc}. Pastikan paket mediapipe dan "
                    "sounddevice terpasang serta mikrofon tersedia."
                ),
            )
