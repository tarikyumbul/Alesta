import cv2
import numpy as np
import easyocr
import time

# EasyOCR okuyucuyu başlat
reader = easyocr.Reader(['en'], gpu=False)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

# Kernel tanımlaması
kernel = np.ones((3, 3), np.uint8)

last_checked_time = time.time()

# Görüntü çözünürlüğünü düşürerek işlem yükünü azaltma
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Her 5 saniyede bir işlemi gerçekleştirmek için kontrol
    if time.time() - last_checked_time >= 5:
        last_checked_time = time.time()

        # Görüntüyü gri tonlamalıya çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Gürültüyü azaltmak için GaussianBlur uygulayalım
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Beyaz dikdörtgenleri bulmak için threshold uygulayın
        _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

        # Konturları bulun
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_numbers = []
        positions = {}

        for contour in contours:
            try:
                x, y, w, h = cv2.boundingRect(contour)

                # Boyut filtrelerini optimize edelim, daha küçük dikdörtgenleri de dahil edelim
                if w < 5 or h < 5 or w > 800 or h > 800:
                    continue

                # Dikdörtgenin içindeki alanı al
                roi = frame[y:y+h, x:x+w]

                # Küçük dikdörtgenlerdeki detayları belirginleştirmek için yeniden boyutlandırma
                scale_factor = max(2, int(400 / max(w, h)))  # Küçük dikdörtgenleri büyütmek için dinamik ölçek
                roi_resized = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                roi_dilated = cv2.dilate(roi_resized, kernel, iterations=1)

                # EasyOCR ile metni oku ve detayları al
                results = reader.readtext(roi_dilated, detail=0)  # 'detail=0' ile sadece metin çıktısını alıyoruz

                # Sonuçları işle
                for text in results:
                    text = text.replace(" ", "")
                    if len(text) == 1 and text in ['1', '2', '3']:  # Tek karakterli ve istenilen rakamsa
                        detected_numbers.append(text)
                        positions[text] = (x, y)
                        # Karakter için bir dikdörtgen çiz
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Rakamı dikdörtgenin üstüne yaz
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            except Exception as e:
                continue

        # Eğer 3 rakam birden tespit edilirse sadece OK çıktısı ver
        if len(set(detected_numbers)) == 3 and all(num in detected_numbers for num in ['1', '2', '3']):
            print("OK")
        else:
            # Eğer tespit edilen rakam sayısı 3'ten azsa NOK çıktısı ver ve SG/SL, U/D kontrolü yap
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

    # İşlenmiş kareyi göster
    cv2.imshow('Detected Numbers', frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()