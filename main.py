from dl_vision_module import DLVisionModule
from fuzzy_control_module import FuzzyControlModule
import cv2
import time
import serial
import numpy as np
import os # Import modul os untuk penanganan path yang lebih baik

# --- Konfigurasi Serial Port ---
SERIAL_PORT = 'COM4' 
BAUD_RATE = 9600

# --- Konfigurasi Zona Pemicu (Dinyatakan dalam Persentase Lebar Frame) ---
TRIGGER_ZONE_MIN = 0.45 
TRIGGER_ZONE_MAX = 0.55

# 📢 PERBAIKAN UTAMA: Tentukan path model Anda di sini
# Gunakan path lengkap (absolut) ke file best.pt Anda.
# Perhatikan penggunaan 'r' di depan string untuk memastikan backslash (\) ditangani dengan benar.
MODEL_FILE_PATH = r'E:\TEORI KONTROL CERDAS\UTS\best.pt'
# -----------------------------------------------------------------------


class SmartConveyorSystem:
    def __init__(self, model_path=MODEL_FILE_PATH, serial_port=None): 
        
        # Tambahkan pemeriksaan untuk memastikan file model ada
        if not os.path.exists(model_path):
            print(f"ERROR: File model tidak ditemukan di path: {model_path}")
            # Anda bisa raise Exception atau keluar dari program di sini
            raise FileNotFoundError(f"File model '{model_path}' tidak ditemukan.")

        print("Menginisialisasi Modul Visi (Deep Learning)...")
        # ✅ PERBAIKAN: Objek dari kelas DLVisionModule telah dibuat
        self.vision_module = DLVisionModule(model_path=model_path)
# ... [Bagian lain dari kelas SmartConveyorSystem tidak berubah] ...
        print("Menginisialisasi Modul Fuzzy Control...")
        self.fuzzy_module = FuzzyControlModule()
        
        self.serial_conn = None
        self.serial_port = serial_port

        if self.serial_port:
            try:
                self.serial_conn = serial.Serial(self.serial_port, BAUD_RATE, timeout=1)
                time.sleep(2) # Beri waktu agar Arduino siap
                print(f"Koneksi serial ke {self.serial_port} berhasil.")
            except serial.SerialException as e:
                print(f"Gagal membuka port serial {self.serial_port}: {e}")
                print("Melanjutkan tanpa koneksi serial (SIMULASI).")
                self.serial_conn = None

        self.prev_time = time.time()
        self.fps = 0
        self.pwm_output = 0
        self.delay_output = 0
        self.sort_command = 0

    def send_command_to_arduino(self, pwm, delay, command):
        # Memastikan nilai berada dalam batas aman
        pwm_int = int(np.clip(pwm, 0, 255))
        delay_int = int(np.clip(delay, 50, 200)) # Batas asumsi
        command_int = int(command)
        
        # Format perintah: PWM, Delay, Command
        command_string = f"{pwm_int},{delay_int},{command_int}\n"
        
        # ✅ PERBAIKAN: Memastikan koneksi serial valid sebelum mengirim
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(command_string.encode('utf-8'))
                # print(f"Mengirim ke Arduino: {command_string.strip()}") # Nonaktifkan cetak untuk performa, aktifkan untuk debug
            except Exception as e:
                print(f"ERROR saat mengirim serial: {e}")
        else:
            # Tampilkan nilai kontrol yang dihitung
            print(f"[SIMULASI] Sent: PWM={pwm_int} (Speed), Delay={delay_int} ms, CMD={command_int} (Sort)")

    def get_sort_command(self, class_name):
        # 🟢 LOGIKA SORTIR
        # Sesuaikan dengan nama kelas yang digunakan model YOLO Anda (misalnya, 'box_merah', 'box_kuning', 'box_hijau')
        if class_name in ['merah', 'kuning']: 
            return 1  # CMD 1 untuk sortir ke sisi A
        elif class_name in ['biru', 'hijau']:
            return 2  # CMD 2 untuk sortir ke sisi B
        else:
            return 0  # CMD 0: Tidak ada sortir

    def run(self):
        print("Memulai Sistem Konveyor Cerdas...")
        cap = cv2.VideoCapture(1) # Ganti 0 jika menggunakan kamera USB eksternal

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Tidak dapat membaca frame dari kamera")
                    break
                
                # Mendapatkan dimensi frame
                height, width, _ = frame.shape
                
                # Perhitungan FPS (U1)
                current_time = time.time()
                dt = current_time - self.prev_time
                self.fps = 1.0 / dt if dt > 0 else 0
                self.prev_time = current_time

                # 1. Deteksi Objek
                results = self.vision_module.run_inference(frame)
                
                # 2. Ekstraksi Informasi Deteksi
                avg_confidence, trigger_objects, annotated_frame = self.vision_module.extract_detection_info(
                    results, frame, TRIGGER_ZONE_MIN, TRIGGER_ZONE_MAX
                )
                
                # Reset command setiap iterasi
                self.sort_command = 0

                # 3. Fuzzy Control (Kecepatan Konveyor)
                self.pwm_output, self.delay_output = self.fuzzy_module.calculate_control(self.fps, avg_confidence)
                
                # 4. Penentuan Perintah Sortir (Hanya jika ada objek di Zona Pemicu)
                if trigger_objects:
                    for obj in trigger_objects:
                        class_name_from_model = obj['class_name'] 
                        self.sort_command = self.get_sort_command(class_name_from_model)
                        print(f"Objek terdeteksi di zona pemicu: {class_name_from_model} -> CMD {self.sort_command}")
                        break 
                
                # 5. Kirim Perintah
                self.send_command_to_arduino(self.pwm_output, self.delay_output, self.sort_command)
                
                # Tampilkan informasi kontrol di frame
                cv2.putText(annotated_frame, f'FPS (U1): {self.fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Avg Conf (U2): {avg_confidence:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'PWM (Output): {int(self.pwm_output)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(annotated_frame, f'Sort CMD: {self.sort_command}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Smart Conveyor System', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            # Tutup koneksi serial jika terbuka
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                print("Port serial ditutup.")
            cv2.destroyAllWindows()
            print("Sistem Konveyor Cerdas Berhenti.")

if __name__ == "__main__":
    # ✅ PERBAIKAN: Menggunakan variabel MODEL_FILE_PATH yang telah disetel di atas
    # Jika Anda ingin mengujinya tanpa model di E:\, Anda bisa mengganti argumen di bawah
    try:
        system = SmartConveyorSystem(model_path=MODEL_FILE_PATH, serial_port=SERIAL_PORT) 
        system.run()
    except FileNotFoundError as e:
        print(f"Aplikasi gagal dimulai karena: {e}")