import cv2
import time
import serial
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from ultralytics import YOLO # Asumsi menggunakan YOLOv8

# --- 1. INISIALISASI ---
# Inisialisasi Model YOLO
# Ganti dengan path model Anda yang sudah dilatih untuk Merah, Kuning, Hijau
MODEL_PATH = 'yolov8n_colors.pt' 
model = YOLO(MODEL_PATH)
CLASS_NAMES = ['merah', 'kuning', 'hijau'] # Pastikan urutan kelas sesuai dengan model

# Inisialisasi Serial
SERIAL_PORT = '/dev/ttyACM0' # Ganti dengan port serial Arduino Anda (misalnya COM3 di Windows)
BAUD_RATE = 9600
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2) # Tunggu koneksi serial stabil
    print(f"Serial port {SERIAL_PORT} terhubung.")
except Exception as e:
    print(f"Gagal terhubung ke serial port: {e}")
    arduino = None # Nonaktifkan pengiriman jika gagal

# Inisialisasi Kamera
cap = cv2.VideoCapture(0) # Ganti 0 jika menggunakan kamera lain
if not cap.isOpened():
    print("Gagal membuka kamera")
    exit()

# --- 2. PERANCANGAN SISTEM KONTROL FUZZY (Sesuai dengan Soal) ---

# Input Variables (Antecedents)
# Laju Frame (FPS): U1=[0, 15] FPS
fps = ctrl.Antecedent(np.arange(0, 15.1, 0.1), 'fps')
# Kepercayaan Deteksi Rata-rata: U2=[0, 1]
confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')

# Output Variables (Consequents)
# Kecepatan Konveyor (PWM): U3=[0, 255]
speed = ctrl.Consequent(np.arange(0, 256, 1), 'speed', defuzzify_method='centroid') 

# Fuzzifikasi - Definisi Fungsi Keanggotaan (Membership Functions)

# FPS - Menggunakan trimf untuk "CUKUP" (sesuai permintaan soal)
# Ulangi untuk LAMBAT dan IDEAL
fps['LAMBAT'] = fuzz.trimf(fps.universe, [0, 0, 8])
# Formula untuk FPS CUKUP: trimf(x; 8, 10, 12) (Contoh, sesuaikan)
fps['CUKUP'] = fuzz.trimf(fps.universe, [8, 10, 12]) 
fps['IDEAL'] = fuzz.trimf(fps.universe, [10, 15, 15])

# Kepercayaan Deteksi (Confidence)
confidence['RENDAH'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
confidence['SEDANG'] = fuzz.trimf(confidence.universe, [0.3, 0.6, 0.9])
confidence['TINGGI'] = fuzz.trimf(confidence.universe, [0.7, 1, 1])

# Kecepatan Konveyor (PWM)
speed['MINIMAL'] = fuzz.trimf(speed.universe, [0, 0, 127])
speed['NORMAL'] = fuzz.trimf(speed.universe, [50, 150, 255])
speed['MAKSIMAL'] = fuzz.trimf(speed.universe, [127, 255, 255])

# Aturan Fuzzy (Contoh, 5 Aturan Adaptif - Sesuai permintaan soal)
# Fokus: FPS rendah/Kepercayaan rendah -> Kecepatan Rendah
rule1 = ctrl.Rule(fps['LAMBAT'] & confidence['RENDAH'], speed['MINIMAL']) # Paling Lambat
rule2 = ctrl.Rule(fps['CUKUP'] & confidence['SEDANG'], speed['NORMAL'])
rule3 = ctrl.Rule(fps['IDEAL'] & confidence['TINGGI'], speed['MAKSIMAL']) # Paling Cepat

# Aturan Adaptif yang Memperhitungkan Keyakinan Tinggi (Override)
rule4 = ctrl.Rule(confidence['TINGGI'], speed['MAKSIMAL']) # Jika yakin TINGGI, percepat konveyor!

# Aturan Adaptif: FPS Ideal tapi Keyakinan Rendah -> Tetap perlambat
rule5 = ctrl.Rule(fps['IDEAL'] & confidence['RENDAH'], speed['NORMAL']) 

# Pembentukan Sistem Kontrol dan Simulasi
speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
speed_simulation = ctrl.ControlSystemSimulation(speed_ctrl)