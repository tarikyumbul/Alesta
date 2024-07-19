import cv2
import easyocr
import numpy as np

# OCR okuyucu oluştur
reader = easyocr.Reader(['en'])

# Kullanıcıdan iki farklı rakam girmesini iste
target_number = input("Gidilecek koordinatı temsil eden rakamı girin: ")
moving_number = input("Hareket eden nesneyi temsil eden rakamı girin: ")

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Videonun kare hızını öğren
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# Video karelerini işleyin
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görseli gri tonlamaya çevir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Görüntüyü büyüt (interpolasyon ile yeniden boyutlandırma)
    scale_factor = 2  # Büyütme oranı
    upscaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Bulanıklaştırma
    blur = cv2.blur(upscaled, (5, 5))

    # Kontrast artırma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(blur)

    # Gürültü azaltma
    denoised = cv2.fastNlMeansDenoising(contrast, None, 30, 7, 21)

    # Morfolojik işlemler
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # Uyarlamalı thresholding uygulama
    adaptive_thresh = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Görselden metinleri oku (sadece 1, 2, 3 ve 4 rakamlarını tanıyacak şekilde)
    results = reader.readtext(adaptive_thresh, allowlist='1234', text_threshold=0.3)

    target_coords = None
    moving_coords = None

    # Tespit edilen metni ve koordinatları yazdır, dikdörtgen içine al
    for (bbox, text, prob) in results:
        # Güven eşiğini burada belirliyoruz
        if text in {'1', '2', '3', '4'} and prob == 1.00:  # Güven eşiği belirleme
            top_left = tuple([int(val / scale_factor) for val in bbox[0]])
            bottom_right = tuple([int(val / scale_factor) for val in bbox[2]])
            
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            
            if width > 10 and height > 10:  # Boyut filtreleme
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Hedef ve hareket eden nesnenin koordinatlarını belirle
                if text == target_number:
                    target_coords = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
                elif text == moving_number:
                    moving_coords = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)

    # Eğer hem hedef hem de hareket eden nesne bulunduysa, çizgiyi çiz
    if target_coords and moving_coords:
        cv2.line(frame, moving_coords, target_coords, (255, 0, 0), 2)

    # İşlenmiş kareyi göster
    cv2.imshow('Adaptive Threshold', frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()