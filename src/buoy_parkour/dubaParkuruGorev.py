import cv2
import numpy as np
import itertools

# Renk aralıklarını HSV cinsinden tanımla (kalibre et)
color_ranges = {
    # Kırmızı duba rengi: https://rgb.to/ral/3000
    'kırmızı': [(0, 120, 70), (10, 255, 255)],  # Alt
    'kırmızı2': [(170, 120, 70), (180, 255, 255)],  # Üst

    # Yeşil duba rengi: https://rgb.to/ral/1021
    'yeşil': [(40, 200, 10), (70, 255, 255)],

    # Sarı duba rengi: https://rgb.to/ral/6001
    'sarı': [(20, 200, 100), (30, 255, 255)]
}

# Renkleri tanımla
neon_yesil = (0, 255, 0)
mavi = (255, 0, 0)
kirmizi = (0, 0, 255)
sari = (0, 255, 255)
beyaz = (255, 255, 255)

# Minimum genişlik oranı (kalibre edilebilir)
width_ratio = 0.7  # Diğer objelerin en büyük objenin en az yüzde 70'i kadar geniş olması gerekir

# Gölge toleransı (kalibre edilebilir)
shadow_tolerance = 30  # Gölgeler için tolere edilecek piksel sayısı

# Dikkate alınacak minimum obje boyutu (kalibre edilebilir)
min_size = 30

# Sarı-yeşil arası alanı tercih etme toleransı, parkur sağa dönüş içerdiği için (kalibre edilebilir)
preference_tolerance = 5

# Son dubanın tespiti (kalibre edilebilir)
min_height_for_parkour_over = 100  # Parkurun bittiğini belirleyecek piksel boy limiti

cap = cv2.VideoCapture(0)

def draw_line(frame, pt1, pt2, color, draw_dot=True):
    # pt1 ve pt2 arasında çizgi çiz
    cv2.line(frame, pt1, pt2, color, 2)
    if draw_dot:
        # Çizginin ortasına nokta çiz
        middle = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.circle(frame, middle, 5, kirmizi, -1)
        return middle
    return None

def draw_center_lines(frame, center_point, target_point, width, height):
    # Dikey sarı çizgi çiz
    if target_point:
        vertical_x = target_point[0]
    else:
        vertical_x = width // 2
    cv2.line(frame, (vertical_x, 0), (vertical_x, height), sari, 2)

    # Merkez noktasına yeşil nokta çiz
    cv2.circle(frame, center_point, 5, neon_yesil, -1)

    # Dikey ve yatay çizgilerin kesiştiği noktaya sarı nokta çiz
    cv2.circle(frame, (vertical_x, center_point[1]), 5, sari, -1)

    return (vertical_x, center_point[1])

# Ekran çözünürlüğünü al ve kareyi ölçekle
screen_width = 1280
screen_height = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Kareyi HSV renkuzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tüm renk maskeleri için birleştirilmiş bir boş maske başlat
    combined_mask = np.zeros(hsv.shape[:2], dtype="uint8")

    all_contours = []
    object_bboxes = []
    object_colors = []

    max_width = 0

    # Her renk aralığını ayrı ayrı işle
    for color, (lower, upper) in color_ranges.items():
        if color == 'kırmızı2':  # İkinci kırmızı aralığını ayrı işle
            continue
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if color == 'kırmızı':  # İki kırmızı maskesini birleştir
            mask2 = cv2.inRange(hsv, np.array(color_ranges['kırmızı2'][0]), np.array(color_ranges['kırmızı2'][1]))
            mask = cv2.add(mask, mask2)
        
        # Bağlantılı parçaları doldurmak için morfolojik kapama uygula
        kernel = np.ones((shadow_tolerance, shadow_tolerance), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # En az 30 piksel genişlik ve yüksekliğe sahip objeleri filtrele
            if w >= min_size and h >= min_size:
                object_bboxes.append((x, y, w, h))
                object_colors.append(color)
                max_width = max(max_width, w)

    # Genişlik oranı ve boyuta göre objeleri filtrele
    filtered_bboxes = []
    filtered_colors = []

    for (x, y, w, h), color in zip(object_bboxes, object_colors):
        if w >= width_ratio * max_width:
            filtered_bboxes.append((x, y, w, h))
            filtered_colors.append(color)
            color_label = f'{color.capitalize()} obje'
            color_value = kirmizi if color == 'kırmızı' else (neon_yesil if color == 'yeşil' else sari)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_value, 2)
            cv2.putText(frame, color_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_value, 2)

    # En soldaki kırmızı obje, en sağdaki yeşil obje ve en büyük sarı objeyi bul
    leftmost_red = None
    rightmost_green = None
    largest_yellow = None
    largest_yellow_size = 0

    for bbox, color in zip(filtered_bboxes, filtered_colors):
        x, y, w, h = bbox
        if color == 'kırmızı' and (leftmost_red is None or x < leftmost_red[0]):
            leftmost_red = bbox
        if color == 'yeşil' and (rightmost_green is None or x + w > rightmost_green[0] + rightmost_green[2]):
            rightmost_green = bbox
        if color == 'sarı' and w * h > largest_yellow_size:
            largest_yellow = bbox
            largest_yellow_size = w * h

    # Sadece ilgili objeleri içeren listeyi oluştur
    relevant_objects = []
    relevant_colors = []

    if leftmost_red:
        relevant_objects.append(leftmost_red)
        relevant_colors.append('kırmızı')
    if rightmost_green:
        relevant_objects.append(rightmost_green)
        relevant_colors.append('yeşil')
    if largest_yellow:
        relevant_objects.append(largest_yellow)
        relevant_colors.append('sarı')

    # Tespit edilen objelere göre sonraki hareketi belirle
    middle_point = None
    if len(relevant_objects) > 0:
        if all(color == 'kırmızı' for color in relevant_colors):
            red_bbox = relevant_objects[0]
            red_height = red_bbox[3]

            if red_height < min_height_for_parkour_over:
                next_movement = "Parkur bitti, son şamandıra tespit edildi"
            elif red_height >= min_height_for_parkour_over:
                next_movement = "Sağa dön"
        elif all(color == 'yeşil' for color in relevant_colors):
            next_movement = "Sola dön" # DO,100
        elif all(color == 'sarı' for color in relevant_colors):
            next_movement = "İleri git" # IL,100
        else:
            # Geçerli bir yeşil obje yoksa
            if 'yeşil' not in relevant_colors:
                next_movement = "Sağa dön"
            else:
                # Herhangi bir objeyi kesmeden geçerli olan en geniş çizgiyi bul
                max_distance = 0
                best_pair = None
                yellow_green_distance = None
                red_yellow_distance = None
                yellow_green_pair = None
                red_yellow_pair = None

                for (i, j) in itertools.combinations(range(len(relevant_objects)), 2):
                    # Sınırlayıcı kutucukların kenarlarını al
                    x1, y1, w1, h1 = relevant_objects[i]
                    x2, y2, w2, h2 = relevant_objects[j]

                    # Kenar noktalarını hesapla (en yakın kenarları birleştirmek istiyoruz)
                    if x1 < x2:
                        pt1 = (x1 + w1, y1 + h1 // 2)  # Birinci objenin sağ kenarı
                        pt2 = (x2, y2 + h2 // 2)       # İkinci objenin sol kenarı
                    else:
                        pt1 = (x2 + w2, y2 + h2 // 2)  # İkinci objenin sağ kenarı
                        pt2 = (x1, y1 + h1 // 2)       # Birinci objenin sol kenarı

                    # Çizginin diğer objeleri kesmeyeceğini kontrol et
                    cross_objects = False
                    for k in range(len(relevant_objects)):
                        if k != i and k != j:
                            xk, yk, wk, hk = relevant_objects[k]
                            if pt1[0] < xk < pt2[0] or pt1[0] < xk + wk < pt2[0]:
                                cross_objects = True
                                break

                    if not cross_objects:
                        # Kenar noktaları arasındaki mesafeyi hesapla
                        distance = np.linalg.norm(np.array(pt1) - np.array(pt2))

                        # Sarı-yeşil ve kırmızı-sarı çiftlerini belirle ve mesafelerini sakla
                        if (relevant_colors[i] == 'sarı' and relevant_colors[j] == 'yeşil') or \
                        (relevant_colors[i] == 'yeşil' and relevant_colors[j] == 'sarı'):
                            yellow_green_distance = distance
                            yellow_green_pair = (pt1, pt2)
                        elif (relevant_colors[i] == 'kırmızı' and relevant_colors[j] == 'sarı') or \
                            (relevant_colors[i] == 'sarı' and relevant_colors[j] == 'kırmızı'):
                            red_yellow_distance = distance
                            red_yellow_pair = (pt1, pt2)

                        # Bu mesafe şimdiye kadarki en büyükse, bunu en iyi çift olarak kaydet
                        if distance > max_distance:
                            max_distance = distance
                            best_pair = (pt1, pt2)

                # Tolerans dahilinde ise, sarı-yeşil mesafesini tercih et
                if yellow_green_distance is not None and red_yellow_distance is not None:
                    # Sarı-yeşil daha küçükse ama tolerans dahilindeyse
                    if yellow_green_distance + preference_tolerance < red_yellow_distance:
                        # Toleransın dışındaysa kırmızı-sarıyı seç
                        best_pair = yellow_green_pair
                        max_distance = yellow_green_distance
                    elif red_yellow_distance > yellow_green_distance:
                        # Aksi takdirde daha geniş olan kırmızı-sarıyı seç
                        best_pair = red_yellow_pair
                        max_distance = red_yellow_distance

                if best_pair:
                    pt1, pt2 = best_pair
                    middle_point = draw_line(frame, pt1, pt2, kirmizi)

                    # Videonun merkez noktasını tanımla
                    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

                    # Merkez noktası ile hedef nokta arasındaki yatay mesafeyi hesapla
                    horizontal_distance_to_center = abs(frame_center[0] - middle_point[0])

                    # Yatay mesafenin yüzdesini hesapla
                    horizontal_percentage = (horizontal_distance_to_center / frame.shape[1]) * 2 * 100

                    # Sonraki hareketi belirle
                    if horizontal_distance_to_center <= 30:
                        next_movement = "Sonraki hareket: İleri git" # IL,100
                    else:
                        if middle_point[0] < frame_center[0]:
                            next_movement = f"Sonraki hareket: %{horizontal_percentage:.0f} sola dön" # SO,0-100
                        else:
                            next_movement = f"Sonraki hareket: %{horizontal_percentage:.0f} sağa dön" # SA,0-100

                    # Hedef noktayı, mesafeyi ve sonraki hareketi yazdır
                    print(f"Hedef nokta: {middle_point}")
                    print(f"Hedef ile merkez noktaları arası yatay mesafe: {int(horizontal_distance_to_center)} piksel")
                else:
                    next_movement = "Geçerli obje tespit edilmedi" # ER,100

        print(next_movement)
    else:
        print("Geçerli obje tespit edilmedi") # ER,100

    # Kare merkezini tanımla
    center_point = (width // 2, height // 2)

    # Merkez çizgileri ve noktaları çiz
    intersection_point = draw_center_lines(frame, center_point, middle_point if middle_point else None, width, height)

    if center_point and intersection_point:
        # Merkez noktasından kesişim noktasına yeşil yatay çizgi çiz, ortasına nokta ekleme
        draw_line(frame, (center_point[0], center_point[1]), (intersection_point[0], center_point[1]), neon_yesil, draw_dot=False)

    # Siyah-beyaz tespit edilen renkler görüntüsü oluştur
    bw_detected_colors = np.zeros_like(frame)
    bw_detected_colors[combined_mask > 0] = [255, 255, 255]

    # İki görüntüyü yatay olarak birleştir
    combined_frame = np.hstack((bw_detected_colors, frame))

    # Birleştirilmiş kareyi ekran boyutuna ölçekle
    frame_height, frame_width = combined_frame.shape[:2]
    scale_ratio = min(screen_width / frame_width, screen_height / frame_height)
    scaled_width = int(frame_width * scale_ratio)
    scaled_height = int(frame_height * scale_ratio)
    scaled_combined_frame = cv2.resize(combined_frame, (scaled_width, scaled_height))

    # Çıktıyı göster
    cv2.imshow('Duba Tanıma', scaled_combined_frame)

    # 'q' tuşuna basılırsa döngüyü kes
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Yakalama işlemini sonlandır ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()