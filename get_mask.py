import cv2
import numpy as np
import pickle
import torch

def update_mask(image):
    # Преобразуем изображение в формат uint8, если оно не в этом формате
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Найдем контуры, чтобы определить границы квадрата
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Найдем самый большой контур (предположительно квадрат)
    largest_contour = max(contours, key=cv2.contourArea)

    # Создадим маску с нулями, того же размера что и изображение
    mask = np.zeros_like(image)
    
    # Заполним маску контуром квадрата
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Применим расширение только внутри маски
    kernel = np.ones((7, 7), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    
    # Ограничиваем действие морфологической операции только в пределах маски
    result = np.where(mask == 255, dilated_image, image)
    
    return result

def convert_to_binary_dark(image_np, threshold=90):
    # Проверяем, если изображение не является одноканальным, преобразуем его в градации серого
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Преобразуем изображение в бинарное: объекты черные, фон белый
    _, binary_image_np = cv2.threshold(image_np, threshold, 255, cv2.THRESH_BINARY_INV)

    binary_image_np = (remove_small_objects(binary_image_np, 100) / 255).astype(int)

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

def update_cadr(frame, classes_coords):
    combined_frame = convert_to_binary_dark(frame)
    combined_frame = update_mask(combined_frame)

    # Используем модель YOLO для предсказаний
    classes_coords_new = {}

    for id_cls, xyxys in classes_coords.items():
        if id_cls in {0, 1, 2, 5, 8, 9}:
            for x1, y1, x2, y2 in xyxys:
                x1 = np.floor(x1).astype(int)  # Округляем влево вниз
                y1 = np.floor(y1).astype(int)  # Округляем вверх вниз
                x2 = np.ceil(x2).astype(int)   # Округляем вправо вверх
                y2 = np.ceil(y2).astype(int)   # Округляем вниз вверх
                combined_frame[y1:y2, x1:x2] = 0
        if id_cls in {3, 5, 9}:
            classes_coords_new[id_cls] = xyxys

    return combined_frame, classes_coords_new

def get_predictions(predictions):
    """
    Выполняет предсказания на одном кадре.
    :param frame: Изображение, на котором будет происходить предсказание.
    :return: Результаты предсказания (тензоры с боксами, классами и уверенностями).
    """
    result = predictions[0]  # Берем первый результат, если обрабатываем один кадр
    return result.boxes.xyxy, result.boxes.conf, result.boxes.cls

def create_mask(classes, confs, class_id, threshold):
    """
    Создает маску для фильтрации боксов по классу и порогу уверенности.
    :param classes: Тензор с классами.
    :param confs: Тензор с уверенностями.
    :param class_id: Интересующий класс.
    :param threshold: Порог уверенности.
    :return: Булевый тензор, который можно использовать для маскировки данных.
    """
    return (classes == class_id) & (confs >= threshold)

def filter_predictions(boxes, confs, classes, class_conf_thresholds, class_limits):
    """
    Фильтрует предсказания по порогам уверенности для каждого класса и ограничивает количество объектов.
    :param boxes: Тензор с координатами боксов.
    :param confs: Тензор с уверенностями для каждого бокса.
    :param classes: Тензор с классами для каждого бокса.
    :param class_conf_thresholds: Список порогов уверенности для каждого класса.
    :param class_limits: Список лимитов (максимальное количество объектов) для каждого класса.
    :return: Отфильтрованные боксы с учетом порогов уверенности и лимитов для каждого класса.
    """
    filtered_boxes = {}

    # Проходим по каждому классу и его порогу уверенности
    for class_id, conf_threshold in enumerate(class_conf_thresholds):
        # Создаем маску для текущего класса и его порога уверенности
        mask = create_mask(classes, confs, class_id, conf_threshold)

        # Проверяем, есть ли хотя бы один box, который прошел фильтр
        if torch.any(mask):
            class_boxes = boxes[mask]
            class_confs = confs[mask]

            # Сортируем боксы по уверенности (по убыванию)
            sorted_indices = torch.argsort(class_confs, descending=True)
            sorted_boxes = class_boxes[sorted_indices]
            sorted_confs = class_confs[sorted_indices]

            # Применяем лимит на количество объектов для текущего класса
            limit = class_limits[class_id]
            limited_boxes = sorted_boxes[:limit]
            limited_confs = sorted_confs[:limit]

            # Добавляем боксы для текущего класса
            filtered_boxes[int(class_id)] = limited_boxes.cpu().numpy()

    return filtered_boxes

def get_all_classes_coordinates(predictions, class_conf_thresholds, class_limits):
    """
    Метод для получения координат боксов для всех классов с учетом заданных порогов уверенности и лимитов для каждого класса.
    :param frame: Изображение, на котором выполняется предсказание.
    :param class_conf_thresholds: Список порогов уверенности для каждого класса.
    :param class_limits: Список лимитов (максимальное количество объектов) для каждого класса.
    :return: Словарь с координатами и уверенностями для всех классов, удовлетворяющих порогу уверенности и лимитам.
    """
    boxes, confs, classes = get_predictions(predictions)
    return filter_predictions(boxes, confs, classes, class_conf_thresholds, class_limits)


def save_cadr(frame, predictions):
    conf_class_for_top = [0.1,0.4,0.4,0.4,0.3,0.3,0.2,0.2,0.3,0.5]
    class_limits = [1,1,1,2,2,3,2,3,2,4]
    classes_coords = get_all_classes_coordinates(predictions, conf_class_for_top, class_limits)
    update_frame, classes = update_cadr(frame, classes_coords)
    np.save(f"numpy_data/frame.npy", update_frame)
    with open(f"numpy_data/boxes.pickle", 'wb') as handle:
        pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
