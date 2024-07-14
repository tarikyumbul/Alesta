import cv2
import numpy as np

# Kamerayı aç. Seçilmek istenen kameraya göre değer "0", "1", "2" vs. olabilir
cap = cv2.VideoCapture(0)

color_to_detect = "all"  # Seçenekler: "red", "green", "yellow", "all"

# Alt ve üst renk aralıkları
color_ranges = {
    "red": {
        "lower1": np.array([0, 120, 100]),
        "upper1": np.array([5, 255, 255]),
        "lower2": np.array([175, 120, 100]),
        "upper2": np.array([180, 255, 255])
    },
    "green": {
        "lower1": np.array([36, 50, 50]),
        "upper1": np.array([86, 255, 255]),
        "lower2": np.array([36, 50, 50]),
        "upper2": np.array([86, 255, 255])
    },
    "yellow": {
        "lower1": np.array([20, 100, 100]),
        "upper1": np.array([30, 255, 255]),
        "lower2": np.array([20, 100, 100]),
        "upper2": np.array([30, 255, 255])
    }
}

# Seçili renklerin tespit edildiği pikselleri beyaz, geri kalanları siyah olarak görüntüleyen bir maske oluştur
def create_combined_mask(frame_hsv, colors):
    # Boş (siyah) bir maskeyi temsil edecek, 0'lardan oluşan bir array oluştur. frame_hsv.shape[:2] = karenin çözünürlüğü
    # uint8 = unsigned 8-bit integer => array, renk değer aralığı olan 0-255 arası değerleri tutabilir
    combined_mask = np.zeros(frame_hsv.shape[:2], dtype="uint8")
    for color in colors:
        if color not in color_ranges:
            raise ValueError(f"Yanlış renk girdisi. Desteklenen girdiler: {list(color_ranges.keys())}")
        lower1 = color_ranges[color]["lower1"]
        upper1 = color_ranges[color]["upper1"]
        lower2 = color_ranges[color]["lower2"]
        upper2 = color_ranges[color]["upper2"]
        
        # Alt ve üst renk aralıklarında olan piksellerin renk değerlerini 255'e ayarla
        lower_mask = cv2.inRange(frame_hsv, lower1, upper1)
        upper_mask = cv2.inRange(frame_hsv, lower2, upper2)
        full_mask = lower_mask + upper_mask
        
        # Yukarıdaki değerleri en baştaki boş (siyah) maskeye ekle. İstenen renklerdeki pikseller beyaz olarak görüntülenecek
        combined_mask = cv2.bitwise_or(combined_mask, full_mask)

    return combined_mask

while True:
    # Görüntüyü karelere ayır
    ret, frame = cap.read()
    
    if not ret:
        break

    # Kareyi yeniden boyutlandır
    frame = cv2.resize(frame, (640, 480))

    # Karenin kopyasını oluştur
    result = frame.copy()

    # Kareyi HSV renk formatına çevir
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color_to_detect == "all":
        colors_to_detect = ["red", "green", "yellow"]
    else:
        colors_to_detect = [color_to_detect]

    combined_mask = create_combined_mask(frame_hsv, colors_to_detect)

    # Maskeyi ve kareyi birleştir
    masked_result = cv2.bitwise_and(result, result, mask=combined_mask)

    # Maskelenmiş kareyi siyah-beyaza çevir, objelerin dış hatlarını belirle
    gray_mask = cv2.cvtColor(masked_result, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Objeleri dış hatları üzerinden ovalliklerine göre filtrele ve görüntüle
    buoys = []
    for contour in contours:
        # Objenin çevresini bul
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            # Objenin ovalliğini kontrol et
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)
            
            # Sadece yeterince oval olan objeleri kabul et
            if 0.7 < circularity < 1.3 and radius > 10:  # Bu ayarları kalibre edebiliriz
                # Dubanın rengini kontrol et
                mask = np.zeros(frame_hsv.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_val = cv2.mean(frame_hsv, mask=mask)[:3]
                for color in colors_to_detect:
                    lower1 = color_ranges[color]["lower1"]
                    upper1 = color_ranges[color]["upper1"]
                    lower2 = color_ranges[color]["lower2"]
                    upper2 = color_ranges[color]["upper2"]
                    if (lower1 <= mean_val).all() and (mean_val <= upper1).all() or (lower2 <= mean_val).all() and (mean_val <= upper2).all():
                        buoys.append((int(x), int(y), int(radius), color))
                        break
    
    # Dubaları boyutları üzerinden filtrele
    if buoys:
        # En büyüğünü bul
        max_radius = max(buoy[2] for buoy in buoys)
        
        # Sadece en büyük dubanın en az %50'si kadar büyük olan dubaları dikkate al (bu sayı değişebilir)
        filtered_buoys = [buoy for buoy in buoys if buoy[2] >= 0.5 * max_radius]
        
        # Filtrelenmiş dubaları görüntüle
        for (x, y, radius, color) in filtered_buoys:
            cv2.circle(result, (x, y), radius, (0, 255, 0), 2)

        # Her 2 ardışık farklı renkteki dubanın arasındaki mesafeyi hesapla
        if len(filtered_buoys) > 1:
            # Dubaları x düzlemine göre sırala
            filtered_buoys.sort(key=lambda buoy: buoy[0])
            
            max_distance = 0
            max_pair = None
            for (x1, y1, r1, color1), (x2, y2, r2, color2) in zip(filtered_buoys, filtered_buoys[1:]):
                if color1 != color2:
                    # Dubaların birbirine bakan kenarları arasındaki mesafeyi hesapla
                    edge_distance = np.sqrt(((x1 + r1) - (x2 - r2)) ** 2 + (y1 - y2) ** 2)
                    # Arasındaki mesafe en uzun olan duba çiftini bul
                    if edge_distance > max_distance:
                        max_distance = edge_distance
                        max_pair = ((x1 + r1, y1), (x2 - r2, y2))
            # Eğer bulunmuş bir en uzun mesafe varsa, görüntüye işle ve konsolda yazdır
            if max_pair:
                (x1, y1), (x2, y2) = max_pair
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)
                print(f"Longest distance center: ({center_x}, {center_y})")

    # Maskeyi ve işlenmiş görüntüyü girdi görüntünün boyutuna getir
    full_mask_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    full_mask_resized = cv2.resize(full_mask_colored, (frame.shape[1], frame.shape[0]))
    result_resized = cv2.resize(result, (frame.shape[1], frame.shape[0]))

    # 3 görüntüyü bir araya getirip birleştir
    stacked_images = np.hstack((frame, full_mask_resized, result_resized))

    # Birleştirilmiş görüntüleri görüntüle
    cv2.imshow('Buoy Detection', cv2.resize(stacked_images, None, fx=0.8, fy=0.8))

    # Eğer "q" tuşuna basılırsa işlemi durdur
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı kapat
cap.release()
cv2.destroyAllWindows()
