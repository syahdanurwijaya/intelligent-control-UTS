import cv2
import numpy as np
# Asumsi Anda menggunakan library Ultralytics (YOLOv8)
from ultralytics import YOLO 

class DLVisionModule:
    """Mengelola inferensi Deep Learning (YOLOv8) dan ekstraksi data objek."""
    def __init__(self, model_path='best.pt'): # <-- PERBAIKAN: Ganti _init_ menjadi __init__
        print("Memuat Model DL...")
        try:
            # Memuat model Deep Learning 
            self.model = YOLO(model_path)
            # Mengambil daftar nama kelas dari model
            self.class_names = self.model.names 
            print(f"Model DL Siap: {model_path}")
        except Exception as e:
            print(f"Gagal memuat model DL: {e}. Pastikan file model ('{model_path}') ada.")
            self.model = None

    def run_inference(self, frame):
        """Melakukan inferensi pada frame."""
        if self.model is None:
            return None
        
        # Mode inferensi cepat (non-verbose)
        results = self.model(frame, verbose=False, stream=False)
        return results

    def extract_detection_info(self, results, frame, trigger_min_ratio, trigger_max_ratio):
        """
        Mengekstrak Kepercayaan Deteksi Rata-rata (U2) dan objek yang 
        memicu penyortiran, serta menghasilkan frame yang dianotasi.
        
        Mengembalikan: (avg_confidence, trigger_objects, annotated_frame)
        """
        height, width, _ = frame.shape
        annotated_frame = frame.copy() 
        
        # Inisialisasi default
        avg_confidence = 0.0
        trigger_objects = []

        if results is None or not results or results[0].boxes is None:
            return avg_confidence, trigger_objects, annotated_frame

        res = results[0]
        confidence_scores = []
        
        # Batas Trigger Zone (dalam piksel)
        trigger_min_x = int(width * trigger_min_ratio)
        trigger_max_x = int(width * trigger_max_ratio)

        # Proses setiap deteksi
        for i, box in enumerate(res.boxes):
            conf = box.conf.item() # Skor Kepercayaan
            class_id = int(box.cls.item())
            class_name = self.class_names.get(class_id, 'unknown')
            
            # Koordinat Bounding Box [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            
            # Hitung Centroid (Pusat Objek)
            centroid_x = (xmin + xmax) / 2
            
            confidence_scores.append(conf)
            
            # Cek apakah Centroid Objek berada di Zona Pemicu (Trigger Zone)
            if trigger_min_x <= centroid_x <= trigger_max_x:
                trigger_objects.append({
                    'class_name': class_name,
                    'centroid_x': centroid_x,
                    'confidence': conf
                })
                # Anotasi: Tandai objek pemicu (Magenta)
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 3)
                cv2.putText(annotated_frame, f'{class_name} ({conf:.2f})', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                 # Anotasi: Objek yang tidak memicu (Hijau)
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{class_name} ({conf:.2f})', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Hitung Rata-rata Kepercayaan Deteksi (U2)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Kembalikan rata-rata kepercayaan, daftar objek pemicu, dan frame yang sudah diwarnai
        return avg_confidence, trigger_objects, annotated_frame