import cv2
import numpy as np
import itertools
import socket

HOST = '127.0.0.1'
PORT = 65432

# HSV formatında renk aralıklarını tanımla
color_ranges = {
    # https://rgb.to/ral/3000
    'red': [(0, 120, 70), (10, 255, 255)],  # Lower
    'red2': [(170, 120, 70), (180, 255, 255)],  # Upper

    # https://rgb.to/ral/1021
    'green': [(40, 200, 40), (70, 255, 255)],

    # https://rgb.to/ral/6001
    'yellow': [(20, 200, 100), (30, 255, 255)]
}

neon_green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
white = (255, 255, 255)

width_ratio = 0.5  # Bir nesneyi dikkate almak için minimum genişlik oranı (kalibre edilebilir)

shadow_tolerance = 30  # Gölgeler için tolere edilecek piksel sayısı (kalibre edilebilir)

min_size = 30   # Bir nesneyi dikkate almak için her iki boyutta minimum boyut (kalibre edilebilir)

camNumber = 4
while True:
    cap = cv2.VideoCapture(camNumber)
    if not cap.isOpened():
        camNumber += 1
    else:
        break

def draw_line(frame, pt1, pt2, color, draw_dot=True):
    cv2.line(frame, pt1, pt2, color, 2)
    if draw_dot:
        middle = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.circle(frame, middle, 5, red, -1)
        return middle
    return None

def draw_center_lines(frame, center_point, target_point, width, height):
    if target_point:
        vertical_x = target_point[0]
    else:
        vertical_x = width // 2
    cv2.line(frame, (vertical_x, 0), (vertical_x, height), yellow, 2)

    cv2.circle(frame, center_point, 5, neon_green, -1)

    cv2.circle(frame, (vertical_x, center_point[1]), 5, yellow, -1)

    return (vertical_x, center_point[1])

screen_width = 1280
screen_height = 720

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))  # Sunucuyu belirtilen IP ve port ile ilişkilendir
    s.listen()  # Gelen bağlantıları dinlemeye başla
    conn, addr = s.accept()  # Bağlantı kabul edildiğinde, yeni bir soket nesnesi ve adres bilgisi al
    with conn:
        print('Bağlantı kuruldu:', addr)
        while True:
            data = ""

            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]

            # Çerçeveyi HSV formatına çevir
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Tüm renk maskelerini birleştirmek için boş bir maske başlat
            combined_mask = np.zeros(hsv.shape[:2], dtype="uint8")

            all_contours = []
            object_bboxes = []
            object_colors = []

            max_width = 0

            # Her renk aralığını ayrı ayrı işle
            for color, (lower, upper) in color_ranges.items():
                if color == 'red2':
                    continue
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                if color == 'red':
                    mask2 = cv2.inRange(hsv, np.array(color_ranges['red2'][0]), np.array(color_ranges['red2'][1]))
                    mask = cv2.add(mask, mask2)
                
                # Küçük boşlukları doldurmak ve parçaları birleştirmek için morfolojik kapatma uygula
                kernel = np.ones((shadow_tolerance, shadow_tolerance), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                combined_mask = cv2.bitwise_or(combined_mask, mask)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Genişlik ve yüksekliği en az 30 piksel olan küçük nesneleri filtrele
                    if w >= min_size and h >= min_size:
                        object_bboxes.append((x, y, w, h))
                        object_colors.append(color)
                        max_width = max(max_width, w)

            # Nesneleri genişlik oranına ve boyutuna göre filtrele
            filtered_bboxes = []
            filtered_colors = []

            for (x, y, w, h), color in zip(object_bboxes, object_colors):
                if w >= width_ratio * max_width:
                    filtered_bboxes.append((x, y, w, h))
                    filtered_colors.append(color)
                    color_label = f'{color.capitalize()} Object'
                    color_value = red if color == 'red' else (neon_green if color == 'green' else yellow)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color_value, 2)
                    cv2.putText(frame, color_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_value, 2)

            leftmost_red = None
            rightmost_green = None
            largest_yellow = None
            largest_yellow_size = 0

            for bbox, color in zip(filtered_bboxes, filtered_colors):
                x, y, w, h = bbox
                if color == 'red' and (leftmost_red is None or x < leftmost_red[0]):
                    leftmost_red = bbox
                if color == 'green' and (rightmost_green is None or x + w > rightmost_green[0] + rightmost_green[2]):
                    rightmost_green = bbox
                if color == 'yellow' and w * h > largest_yellow_size:
                    largest_yellow = bbox
                    largest_yellow_size = w * h

            relevant_objects = []
            relevant_colors = []

            if leftmost_red:
                relevant_objects.append(leftmost_red)
                relevant_colors.append('red')
            if rightmost_green:
                relevant_objects.append(rightmost_green)
                relevant_colors.append('green')
            if largest_yellow:
                relevant_objects.append(largest_yellow)
                relevant_colors.append('yellow')

            # Tespit edilen nesnelere göre bir sonraki hareketi belirle
            middle_point = None
            if len(relevant_objects) > 0:
                if all(color == 'red' for color in relevant_colors):
                    next_movement = "Saga don" # DA,100
                    data = "DA,100"
                elif all(color == 'green' for color in relevant_colors):
                    next_movement = "Sola don" # DO,100
                    data = "DO,100"
                elif all(color == 'yellow' for color in relevant_colors):
                    next_movement = "Düz ilerle" # IL,100
                    data = "IL,100"
                else:
                    # Herhangi bir nesnenin üstünden geçmeden en geniş alanı bul
                    max_distance = 0
                    best_pair = None

                    for (i, j) in itertools.combinations(range(len(relevant_objects)), 2):
                        # Sınır kutularının kenarlarını al
                        x1, y1, w1, h1 = relevant_objects[i]
                        x2, y2, w2, h2 = relevant_objects[j]

                        # 2 objenin yakın kenar noktalarını bul
                        if x1 < x2:
                            pt1 = (x1 + w1, y1 + h1 // 2)  # İlk kutunun sağ kenarı
                            pt2 = (x2, y2 + h2 // 2)       # İkinci kutunun sol kenarı
                        else:
                            pt1 = (x2 + w2, y2 + h2 // 2)  # İkinci kutunun sağ kenarı
                            pt2 = (x1, y1 + h1 // 2)       # İlk kutunun sol kenarı

                        # Çizginin nesnelerin üstünden geçip geçmediğini kontrol et
                        cross_objects = False
                        for k in range(len(relevant_objects)):
                            if k != i and k != j:
                                xk, yk, wk, hk = relevant_objects[k]
                                if pt1[0] < xk < pt2[0] or pt1[0] < xk + wk < pt2[0]:
                                    cross_objects = True
                                    break

                        if not cross_objects:
                            # İki nokta arasındaki mesafeyi hesapla
                            distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
                            if distance > max_distance:
                                max_distance = distance
                                best_pair = (pt1, pt2)

                    if best_pair:
                        pt1, pt2 = best_pair
                        middle_point = draw_line(frame, pt1, pt2, red)

                        # Kameranın merkez noktasını tanımla
                        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

                        # Hedef nokta ile merkez nokta arasındaki yatay mesafeyi hesapla
                        horizontal_distance_to_center = abs(frame_center[0] - middle_point[0])

                        # Hedef nokta ile merkez nokta arasındaki yatay mesafeyi yüzde olarak hesapla
                        horizontal_percentage = (horizontal_distance_to_center / frame.shape[1]) * 2 * 100

                        # Sonraki hareketi belirle
                        if horizontal_distance_to_center <= 30:
                            next_movement = "Sonraki manevra: Düz ilerle" # IL,100
                            data = "IL,100"
                        else:
                            if middle_point[0] < frame_center[0]:
                                next_movement = f"Sonraki manevra: %{horizontal_percentage:.0f} sola dön" # SO,0-100
                                data = f"SO,{horizontal_percentage:.0f}"
                            else:
                                next_movement = f"Sonraki manevra: %{horizontal_percentage:.0f} sağa dön" # SA,0-100
                                data = f"SA,{horizontal_percentage:.0f}"

                        print(f"Hedef nokta: {middle_point}")
                        print(f"Hedef ve merkez noktalar arasındaki yatay mesafe: {int(horizontal_distance_to_center)} piksel")
                    else:
                        next_movement = "Obje tespit edilemedi" # ER,100
                        data = "ER,100"

                print(next_movement)
            else:
                print("Obje tespit edilemedi") # ER,100
                data = "ER,100"

            conn.sendall(data.encode())  # Veriyi istemciye gönder
            if data == "exit":  # Eğer kullanıcı "exit" girerse döngüden çık
                break

            # Kameranın merkez noktasını tanımla
            center_point = (width // 2, height // 2)

            # Merkez noktayı ve hedef noktayı birleştir
            intersection_point = draw_center_lines(frame, center_point, middle_point if middle_point else None, width, height)

            if center_point and intersection_point:
                # Merkez noktadan kesişim noktasına çizgi çiz
                draw_line(frame, (center_point[0], center_point[1]), (intersection_point[0], center_point[1]), neon_green, draw_dot=False)

            bw_detected_colors = np.zeros_like(frame)
            bw_detected_colors[combined_mask > 0] = [255, 255, 255]

            combined_frame = np.hstack((bw_detected_colors, frame))

            frame_height, frame_width = combined_frame.shape[:2]
            scale_ratio = min(screen_width / frame_width, screen_height / frame_height)
            scaled_width = int(frame_width * scale_ratio)
            scaled_height = int(frame_height * scale_ratio)
            scaled_combined_frame = cv2.resize(combined_frame, (scaled_width, scaled_height))

            cv2.imshow('Duba Tespiti', scaled_combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
