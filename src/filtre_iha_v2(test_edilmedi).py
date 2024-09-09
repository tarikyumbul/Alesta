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
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
picam2.configure(config)
picam2.start()

# Kernel tanımlaması
kernel = np.ones((3, 3), np.uint8)

last_checked_time = time.time()
last_frame_time = time.time()  # İlk kare zamanını tanımla
fps = 0  # Başlangıç FPS değeri

while True:

    current_time = time.time()
    
    # Her kare için geçen zamanı hesapla
    frame_time_diff = current_time - last_frame_time
    last_frame_time = current_time
    
    if frame_time_diff > 0:  # Sıfırdan büyükse FPS'yi hesapla
        fps = 1.0 / frame_time_diff

    frame = picam2.capture_array()
    if frame is None:
        break

    # Anlık FPS değerini ekrana yazdır
    print(f"Anlık FPS: {fps:.2f}")
    
    # Her 7 saniyede bir işlemi gerçekleştirmek için kontrol
    if time.time() - last_checked_time >= 7:
        last_checked_time = time.time()

        try:
            # Gri tonlamalıya çevir
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Histogram eşitleme: kontrast artırma
            gray = cv2.equalizeHist(gray)

            # Gürültüyü azaltmak için Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Thresholding (Otsu metodu ile)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morfolojik işlemler: küçük gürültüleri temizleyip, rakamları daha belirgin hale getirme
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Kenarları bul
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"{len(contours)} kontur bulundu.")

            detected_numbers = []
            positions = {}

            for contour in contours:
                try:
                    # Dikdörtgen oluştur
                    x, y, w, h = cv2.boundingRect(contour)
                    print(f"Dikdörtgen bulundu: x={x}, y={y}, w={w}, h={h}")

                    # Çok küçük veya çok büyük konturları atla
                    if w < 5 or h < 5 or w > 800 or h > 800:
                        print("Dikdörtgen boyutu uygun değil, atlanıyor...")
                        continue

                    # ROI (Region of Interest - İlgi Alanı) oluştur
                    roi = frame[y:y+h, x:x+w]

                    # Dinamik ölçekleme: küçük dikdörtgenleri büyüt
                    scale_factor = max(2, int(400 / max(w, h)))
                    roi_resized = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

                    # Dilate işlemi: rakamları daha belirgin yapmak için genişletme
                    roi_dilated = cv2.dilate(roi_resized, kernel, iterations=1)

                    # OCR işlemi (EasyOCR kullanarak)
                    results = reader.readtext(roi_dilated, detail=0)

                    # Rakamları kontrol et
                    for text in results:
                        text = text.replace(" ", "")
                        if len(text) == 1 and text in ['1', '2', '3']:  # Tek karakterli ve istenilen rakamsa
                            detected_numbers.append(text)
                            positions[text] = (x, y)
                            print(f"Rakam tespit edildi: {text}")
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                except Exception as e:
                    print(f"Contour işlemi sırasında hata: {e}")
                    continue

            # 1, 2 ve 3'ün tümü tespit edilirse "OK" yazdır
            if len(set(detected_numbers)) == 3 and all(num in detected_numbers for num in ['1', '2', '3']):
                print("OK")
            else:
                # Eğer tespit edilen rakamlar varsa pozisyonlarını analiz et
                if detected_numbers:
                    avg_x_position = np.mean([positions[num][0] for num in detected_numbers])
                    avg_y_position = np.mean([positions[num][1] for num in detected_numbers])

                    # Yüksekliğe göre yukarıda mı aşağıda mı olduğunu belirle
                    direction = "U" if avg_y_position < frame.shape[0] // 2 else "D"

                    # X pozisyonuna göre sola veya sağa yakınlığını belirle
                    if avg_x_position > frame.shape[1] // 2:
                        print(f"NOK {direction} SG")
                    else:
                        print(f"NOK {direction} SL")
                else:
                    print("NOK")

        except Exception as e:
            print(f"OCR işlemi sırasında hata: {e}")

    cv2.imwrite('detected_numbers_output.jpg', frame)
    print("Resim dosyaya kaydedildi.")
