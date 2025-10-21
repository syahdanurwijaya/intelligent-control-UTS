import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyControlModule:
    """Modul Kontrol Fuzzy Adaptif (Mamdani) untuk Konveyor Cerdas.
    Input: FPS (U1), Confidence Rata-rata (U2)
    Output: PWM Konveyor (U3), Jeda Waktu Sortir/Jarak Aman (U4)
    """
    def __init__(self): # <-- PERBAIKAN: Ganti _init_ menjadi __init__
        print("Membangun Sistem Fuzzy...")
        
        # --- Definisikan Universe of Discourse ---
        self.fps_range = np.arange(0, 15.1, 0.1)      # U1: [0, 15] FPS
        self.conf_range = np.arange(0, 1.01, 0.01)    # U2: [0, 1]
        self.pwm_range = np.arange(0, 256, 1)         # U3: [0, 255] PWM
        self.delay_range = np.arange(50, 201, 1)      # U4: [50, 200] ms (Jarak Aman)

        # --- Definisikan Variabel Fuzzy (Linguistik) ---
        self.fps = ctrl.Antecedent(self.fps_range, 'fps')
        self.conf = ctrl.Antecedent(self.conf_range, 'conf')
        
        # Consequent (Output) menggunakan Centroid Defuzzification (default skfuzzy)
        self.pwm = ctrl.Consequent(self.pwm_range, 'pwm', defuzzify_method='centroid')
        self.delay = ctrl.Consequent(self.delay_range, 'delay', defuzzify_method='centroid')

        # --- Definisikan Himpunan Keanggotaan (Membership Functions) ---
        
        # FPS (U1): Lambat, Normal, Ideal
        self.fps['lambat'] = fuzz.trapmf(self.fps_range, [0, 0, 4, 6])
        # PERBAIKAN: Ganti 'cukup' menjadi 'normal' agar sesuai dengan definisi
        self.fps['normal'] = fuzz.trimf(self.fps_range, [4, 8, 12])
        self.fps['ideal'] = fuzz.trapmf(self.fps_range, [10, 14, 15, 15])

        # Kepercayaan (U2): Rendah, Sedang, Tinggi
        self.conf['rendah'] = fuzz.trapmf(self.conf_range, [0, 0, 0.4, 0.6])
        self.conf['sedang'] = fuzz.trimf(self.conf_range, [0.5, 0.7, 0.9])
        self.conf['tinggi'] = fuzz.trapmf(self.conf_range, [0.8, 1.0, 1.0, 1.0])

        # PWM (U3) - Output: Minimal, Normal, Maksimal
        self.pwm['minimal'] = fuzz.trapmf(self.pwm_range, [0, 0, 80, 120])
        self.pwm['normal'] = fuzz.trimf(self.pwm_range, [100, 170, 220])
        self.pwm['maksimal'] = fuzz.trapmf(self.pwm_range, [200, 255, 255, 255])

        # Delay (U4) - Output: Pendek, Sedang, Panjang
        self.delay['pendek'] = fuzz.trapmf(self.delay_range, [50, 50, 80, 120])
        self.delay['sedang'] = fuzz.trimf(self.delay_range, [100, 140, 180])
        self.delay['panjang'] = fuzz.trapmf(self.delay_range, [160, 200, 200, 200])

        # --- Definisikan Aturan Fuzzy (9 Aturan) ---
        
        rules = [
            # R1: Kinerja DL sangat buruk -> Utamakan akurasi, minimalisir kecepatan
            ctrl.Rule(self.fps['lambat'] & self.conf['rendah'], (self.pwm['minimal'], self.delay['panjang'])),
            # R2: FPS Lambat & Conf Sedang
            ctrl.Rule(self.fps['lambat'] & self.conf['sedang'], (self.pwm['minimal'], self.delay['sedang'])),
            # R3: Deteksi Akurat tapi Proses Lambat -> Kecepatan Normal, Jeda Panjang
            ctrl.Rule(self.fps['lambat'] & self.conf['tinggi'], (self.pwm['normal'], self.delay['panjang'])),
            
            # R4: FPS Normal & Conf Tinggi (PERBAIKAN: Ganti 'cukup' menjadi 'normal')
            ctrl.Rule(self.fps['normal'] & self.conf['tinggi'], (self.pwm['normal'], self.delay['pendek'])),
            # R5: Kinerja DL Optimal -> Kecepatan Maksimal
            ctrl.Rule(self.fps['ideal'] & self.conf['tinggi'], (self.pwm['maksimal'], self.delay['pendek'])),
            # R6: FPS Ideal tapi Conf Rendah -> Hati-hati, turunkan kecepatan sedikit
            ctrl.Rule(self.fps['ideal'] & self.conf['rendah'], (self.pwm['normal'], self.delay['sedang'])),
            
            # R7: FPS Normal & Conf Rendah (PERBAIKAN: Ganti 'cukup' menjadi 'normal')
            ctrl.Rule(self.fps['normal'] & self.conf['rendah'], (self.pwm['minimal'], self.delay['panjang'])),
            # R8: Kondisi Stabil (Normal)
            ctrl.Rule(self.fps['normal'] & self.conf['sedang'], (self.pwm['normal'], self.delay['sedang'])),
            # R9: FPS Ideal & Conf Sedang
            ctrl.Rule(self.fps['ideal'] & self.conf['sedang'], (self.pwm['maksimal'], self.delay['sedang'])),
        ]

        # --- Buat Sistem Kontrol ---
        self.control_system = ctrl.ControlSystem(rules)
        self.control_simulation = ctrl.ControlSystemSimulation(self.control_system)

        # --- Atribut untuk menyimpan nilai PWM terakhir ---
        self.pwm_output = 0 # Inisialisasi nilai default

        print("Sistem Fuzzy Siap.")

    def calculate_control(self, fps_value, conf_value):
        """Menghitung output PWM (U3) dan Delay (U4) berdasarkan input FPS (U1) dan Confidence (U2)."""
        try:
            # Set input dan klip nilai agar sesuai dengan semesta pembicaraan
            self.control_simulation.input['fps'] = np.clip(fps_value, 0, 15)
            self.control_simulation.input['conf'] = np.clip(conf_value, 0, 1.0)

            # Proses inferensi fuzzy 
            self.control_simulation.compute()

            # Ambil output crisp dan klip agar sesuai dengan batasan fisik Arduino
            pwm_output = np.clip(self.control_simulation.output['pwm'], 0, 255)
            delay_output = np.clip(self.control_simulation.output['delay'], 50, 200)

            # Simpan nilai PWM ke dalam atribut instance
            self.pwm_output = pwm_output

            return pwm_output, delay_output
        except Exception as e:
            # Kembalikan nilai default aman (Stop Konveyor, Delay Moderat) jika ada error
            print(f"Error dalam perhitungan fuzzy: {e}. Menggunakan nilai default aman (PWM=0, Delay=150ms).")
            self.pwm_output = 0 # Juga update atribut jika error
            return 0, 150
