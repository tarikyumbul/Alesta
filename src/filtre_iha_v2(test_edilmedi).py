import numpy as np
import easyocr
import time
import warnings
from picamera2 import Picamera2
import cv2

print(cv2.__version__)
# FutureWarning uyarısını bastır
warnings.filterwarnings("ignore", category=FutureWarning)

reader = easyocr.Reader(['en'], gpu=False)

# Picamera2 ile kamera başlatma
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})  # Orta çözünürlük
picam2.configure(config)
picam2.start()

# Kernel tanımlaması (morfolojik işlemler için)
kernel = np.ones((5, 5), np.uint8)  # Kernel boyutunu büyüttüm

# A4'e uygun dikdörtgen sınırları
A4_min_width = 50  
A4_min_height = 70  
A4_max_width = 400  # Resimdeki rakamlar büyük olduğu için sınırı biraz arttırdım
A4_max_height = 500  

last_checked_time = time.time()
last_frame_time = time.time()  
fps = 0  

while True:
    current_time = time.time()
    
    # FPS hesapla
    frame_time_diff = current_time - last_frame_time
    last_frame_time = current_time
    
    if frame_time_diff > 0:  
        fps = 1.0 / frame_time_diff

    frame = picam2.capture_array()
    if frame is None:
        break

    print(f"Anlık FPS: {fps:.2f}")
    
    if time.time() - last_checked_time >= 3:  # 3 saniyede bir işle
        last_checked_time = time.time()

        try:
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Kontrast artırma (CLAHE kullanımı)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Gürültü azaltma (GaussianBlur)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Biraz daha güçlü blur işlemi
            
            # Thresholding (Otsu metodu yerine adaptive threshold)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Morfolojik işlemler
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Kenar bulma
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"{len(contours)} kontur bulundu.")

            detected_numbers = []
            positions = {}

            for contour in contours:
                try:
                    # Dikdörtgen oluştur
                    x, y, w, h = cv2.boundingRect(contour)

                    # Sınır kontrolü
                    if w < A4_min_width or h < A4_min_height or w > A4_max_width or h > A4_max_height:
                        continue

                    # ROI oluştur
                    roi = frame[y:y+h, x:x+w]

                    # Küçük dikdörtgenleri büyüt
                    scale_factor = max(1.5, int(300 / max(w, h)))  # Ölçek faktörünü düşürdüm çünkü rakamlar yeterince büyük
                    roi_resized = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

                    # Dilate işlemi
                    roi_dilated = cv2.dilate(roi_resized, kernel, iterations=1)

                    # OCR işlemi
                    results = reader.readtext(roi_dilated, detail=0)

                    # Yalnızca 1, 2, 3 rakamlarını kontrol et
                    for text in results:
                        text = text.replace(" ", "")
                        if len(text) == 1 and text in ['1', '2', '3']:  # Sadece 1, 2, 3 rakamlarını kontrol et
                            detected_numbers.append(text)
                            positions[text] = (x, y)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                except Exception as e:
                    print(f"Contour işlemi sırasında hata: {e}")
                    continue

            # Eğer 1, 2, 3 varsa "OK"
            if len(set(detected_numbers)) >= 3 and all(num in detected_numbers for num in ['1', '2', '3']):
                print("OK")
            else:
                if detected_numbers:
                    avg_x_position = np.mean([positions[num][0] for num in detected_numbers])
                    avg_y_position = np.mean([positions[num][1] for num in detected_numbers])

                    # Yukarıda mı aşağıda mı olduğunu belirle
                    direction = "U" if avg_y_position < frame.shape[0] // 2 else "D"
                    
                    # Sağa veya sola yakınlık belirleme
                    if avg_x_position > frame.shape[1] // 2:
                        print(f"NOK {direction} SG")
                    else:
                        print(f"NOK {direction} SL")
                else:
                    print("NOK")

        except Exception as e:
            print(f"OCR işlemi sırasında hata: {e}")

    # Görüntü kaydı
    cv2.imwrite('detected_numbers_output.jpg', frame)
    print("Resim dosyaya kaydedildi.")
