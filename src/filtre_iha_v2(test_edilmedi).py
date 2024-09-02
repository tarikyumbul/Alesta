import numpy as np
import easyocr
import time
import warnings
from picamera2 import Picamera2
import cv2
print(cv2.__version__)
# FutureWarning uyarısını bastır
warnings.filterwarnings("ignore", category=FutureWarning)

# EasyOCR okuyucuyu başlat
print("EasyOCR okuyucu başlatılıyor...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR okuyucu başlatıldı.")

# Picamera2 ile kamera başlatma
print("Kamera başlatılıyor...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (320, 240)})
picam2.configure(config)
picam2.start()
print("Kamera başlatıldı.")

# Kernel tanımlaması
kernel = np.ones((3, 3), np.uint8)

last_checked_time = time.time()

while True:
    print("Kamera görüntüsü alınıyor...")
    frame = picam2.capture_array()
    if frame is None:
        print("Görüntü alınamadı, döngüden çıkılıyor...")
        break
    print("Görüntü alındı.")

    # Her 7 saniyede bir işlemi gerçekleştirmek için kontrol
    if time.time() - last_checked_time >= 7:
        last_checked_time = time.time()

        try:
            print("Görüntü gri tonlamalıya çevriliyor...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Görüntü gri tonlamalıya çevrildi.")

            print("GaussianBlur uygulanıyor...")
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            print("GaussianBlur uygulandı.")

            print("Threshold uygulanıyor...")
            _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
            print("Threshold uygulandı.")

            print("Konturlar bulunuyor...")
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"{len(contours)} kontur bulundu.")

            detected_numbers = []
            positions = {}

            for contour in contours:
                try:
                    print("Konturdan dikdörtgen oluşturuluyor...")
                    x, y, w, h = cv2.boundingRect(contour)
                    print(f"Dikdörtgen bulundu: x={x}, y={y}, w={w}, h={h}")

                    if w < 5 or h < 5 or w > 800 or h > 800:
                        print("Dikdörtgen boyutu uygun değil, atlanıyor...")
                        continue

                    roi = frame[y:y+h, x:x+w]
                    print("ROI oluşturuldu.")

                    scale_factor = max(2, int(400 / max(w, h)))  # Küçük dikdörtgenleri büyütmek için dinamik ölçek
                    roi_resized = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                    roi_dilated = cv2.dilate(roi_resized, kernel, iterations=1)
                    print("ROI yeniden boyutlandırıldı ve dilate edildi.")

                    print("OCR işlemi başlatılıyor...")
                    results = reader.readtext(roi_dilated, detail=0)
                    print("OCR işlemi tamamlandı.")

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

            if len(set(detected_numbers)) == 3 and all(num in detected_numbers for num in ['1', '2', '3']):
                print("OK")
            else:
                if detected_numbers:
                    avg_x_position = np.mean([positions[num][0] for num in detected_numbers])
                    avg_y_position = np.mean([positions[num][1] for num in detected_numbers])

                    direction = "U" if avg_y_position < frame.shape[0] // 2 else "D"

                    if avg_x_position > frame.shape[1] // 2:
                        print(f"NOK {direction} SG")
                    else:
                        print(f"NOK {direction} SL")
                else:
                    print("NOK")

        except Exception as e:
            print(f"OCR işlemi sırasında hata: {e}")

    cv2.imshow('Detected Numbers', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
