import cv2
import numpy as np
from ultralytics import YOLO
import logging
from math import atan2, cos, sin, sqrt, pi

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Названия классов
class_names = ['Ball', 'Base green', 'Base red', 'Basket', 'Button blue', 'Button box with red', 'Button green', 'Button red', 'Cube', 'Enemy']

# Правильные RGB цвета для классов
class_colors = [
    [255, 165, 0],    # оранжевый
    [0, 128, 0],      # зеленый
    [255, 0, 0],      # красный
    [128, 128, 128],  # серый (Basket)
    [135, 206, 235],  # голубой (Button blue)
    [245, 245, 220],  # бежевый
    [127, 255, 0],    # салатовый
    [255, 255, 0],    # желтый
    [128, 0, 0],      # бордовый
    [0, 0, 255]       # синий
]

# Приоритеты классов (серый и голубой классы с максимальными приоритетами)
class_priorities = [0, 1, 2, 10, 9, 3, 4, 5, 6, 7]

# Загружаем модель YOLO
model = YOLO('C:\\Users\\redmi\\Desktop\\work\\test\\best.pt')


def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
  ## [visualization1]
 
def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv2.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  label = str(-int(np.rad2deg(angle)) - 90)
  textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
 
  return angle
 


def update_image(img):
    h, w = img.shape[:2]
    # Параметры камеры и искажения
    mtx = np.array([[1.17937478e+03, 0.00000000e+00, 9.24866066e+02],
                    [0.00000000e+00, 1.17865941e+03, 5.47165399e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([-0.37825373,  0.16971861, -0.00140652,  0.00480215, -0.03276766])
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Коррекция искажений
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Обрезаем изображение на основе roi (регион интереса)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst

def convert_to_binary_dark(image_np, threshold=80):
    # Преобразуем изображение в градации серого
    gray_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Преобразуем изображение в бинарное: объекты черные, фон белый
    _, binary_image_np = cv2.threshold(gray_image_np, threshold, 255, cv2.THRESH_BINARY_INV)

    return binary_image_np


def remove_small_objects(binary_image_np, min_area):
    # Находим все контуры на бинарной маске
    contours, _ = cv2.findContours(binary_image_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Проходим по всем найденным контурам
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Если площадь контура меньше минимальной, заполняем этот контур черным цветом
        if area < min_area:
            cv2.drawContours(binary_image_np, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    
    return binary_image_np

def nothing(args):pass
cv2.namedWindow("setup")
cv2.createTrackbar("1", "setup", 0, 255, nothing)
cv2.createTrackbar("2", "setup", 0, 255, nothing)
cv2.createTrackbar("3", "setup", 0, 255, nothing)
cv2.createTrackbar("4", "setup", 0, 255, nothing)
cv2.createTrackbar("5", "setup", 0, 255, nothing)
cv2.createTrackbar("6", "setup", 0, 255, nothing)

# Открытие видео
# ip_camera_url_left = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"
# cap = cv2.VideoCapture(ip_camera_url_left)

pt = "C:\\Users\\redmi\\Downloads\\Telegram Desktop\\Left_1.avi"
cap = cv2.VideoCapture(pt)
if not cap.isOpened():
    print("Ошибка открытия видеофайла")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Конец видео")
        break

    # Обновляем изображение
    updated_frame = update_image(frame)

    # Применяем бинаризацию к обновлённому изображению (создаём черно-белую маску с белым фоном)
    binary_frame = convert_to_binary_dark(updated_frame)

    # Удаляем маленькие объекты (задаём минимальную площадь, например, 100 пикселей)
    binary_frame_cleaned = remove_small_objects(binary_frame, min_area=100)

    # Преобразуем черно-белую маску в трёхканальное изображение для совмещения с цветными слоями
    binary_frame_colored = cv2.cvtColor(binary_frame_cleaned, cv2.COLOR_GRAY2BGR)

    # Накладываем черно-белую маску на оригинальное изображение, делая её нижним слоем
    combined_frame = cv2.addWeighted(binary_frame_colored, 1, updated_frame, 0, 0)

    # Используем модель YOLO для предсказаний
    results = model.predict(source=updated_frame, imgsz=640, conf=0.60)  # Конфигурация размера и порога
    result = results[0].cpu()

    boxes = result.boxes

    # Сортируем боксы по приоритетам классов
    sorted_boxes = sorted(boxes, key=lambda box: class_priorities[int(box.cls[0])])
    maxi = 0
    for i, box in enumerate(sorted_boxes):
        # Получаем координаты бокса в формате x1, y1, x2, y2
        xyxy = box.xyxy[0].numpy()  # Извлекаем numpy массив для координат

        # Координаты
        x1, y1, x2, y2 = map(int, xyxy)

        # Получаем класс объекта
        class_id = int(box.cls[0])  # Получаем класс объекта (от 0 до 9)
        if class_id == 9:
            if (x2 - x1) * (y2 - y1) > maxi:
                maxi = (x2 - x1) * (y2 - y1)
                x1_max = x1
                y1_max = y1
                x2_max = x2
                y2_max = y2
                # print(x1, x2, y1, y2)
        # Выбираем цвет для этого класса и переводим его из RGB в BGR
        color = class_colors[class_id][::-1]  # Реверсируем порядок цветов с RGB на BGR

        # Закрашиваем прямоугольник на изображении поверх черно-белой маски
        cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, -1)
    p1 = cv2.getTrackbarPos('1', 'setup')
    p2 = cv2.getTrackbarPos('2', 'setup')
    p3 = cv2.getTrackbarPos('3', 'setup')
    p4 = cv2.getTrackbarPos('4', 'setup')
    p5 = cv2.getTrackbarPos('5', 'setup')
    p6 = cv2.getTrackbarPos('6', 'setup')
    # Трансляция видео
    if maxi != 0:
        # combined_frame[x1:x2, y1:y2]
        # print(x1_max, x2_max, y1_max, y2_max)
        to_show = updated_frame[y1_max - 20:y2_max + 20, x1_max - 20:x2_max + 20, :]
        print(updated_frame.shape, (y1_max - 20,y2_max + 20, x1_max - 20,x2_max + 20))
        # print(to_show.shape)
        # print(updated_frame.shape)
        # print(combined_frame.shape)
        # hsv_min = np.array((0, 54, 5), np.uint8)
        # hsv_max = np.array((187, 255, 253), np.uint8)
        hsv_min = np.array([p1, p2, p3])
        hsv_max = np.array([p4, p5, p6])
        hsv = cv2.cvtColor( to_show, cv2.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
        thresh = cv2.inRange( hsv, hsv_min, hsv_max )
        blurVal = 2
        mask = cv2.medianBlur(thresh, 1 + blurVal*2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)
        # gray = cv2.cvtColor(thresh, cv2.COLOR_HSV2BGR)
        # gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        
        # # _, bw = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # _, bw = cv2.threshold(gray, p1, p2, cv2.THRESH_BINARY)
    
        # Find all the contours in the thresholded image
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # print(contours)
        for i, c in enumerate(contours):
		# Calculate the area of each contour
            rect = cv2.minAreaRect(c) # пытаемся вписать прямоугольник
            box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
            box = np.int0(box) # округление координат
            area = int(rect[1][0]*rect[1][1]) # вычисление площади
            print(area)
            if area < 1000 or area > 10000:
                continue
            cv2.drawContours(to_show, contours, i, (0, 0, 255), 2)
		
		# Find the orientation of each shape
            getOrientation(c, to_show)
        to_show = cv2.resize(to_show, (640, 480))
        # bw = cv2.resize(bw, (640, 480))
        thresh = cv2.resize(thresh, (640, 480))
        
        cv2.imshow('Binary Mask with White Background and Colored Boxesss', to_show)
        cv2.imshow('Binary Mask with White Background and Colored Boxes', thresh)
        

    # cv2.imshow('Binary Mask with White Background and Colored Boxes', combined_frame)
    

    # Ожидание выхода по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрытие всех окон
cap.release()
cv2.destroyAllWindows()


 