import cv2
import urllib.request
import numpy as np
import torch
from ultralytics import YOLO
import logging

def define_class(cls_id, xyxys, classes):
    mask = classes == cls_id
    # print(xyxys[mask][0])
    return xyxys[mask][0] if torch.any(mask) else -1


def find_obj(cls_id, results):
    boxes = results[0].boxes
    classif = boxes.cls
    xyxys = boxes.xyxy

    return define_class(cls_id, xyxys, classif)


logging.getLogger("ultralytics").setLevel(logging.ERROR)
model = YOLO("best (3).pt")
url = 'http://192.168.1.1:8080/?action=stream'
stream = urllib.request.urlopen(url)

current_pos = None

bts = bytes()
threads = []
cls_id = 8
def do():
    global bts, current_pos
    bts += stream.read(1024)
    a = bts.find(b'\xff\xd8')
    b = bts.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bts[a:b + 2]
        bts = bts[b + 2:]

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        predict = model.predict(frame, conf=0.7)
        frame = predict[0].plot()
        cv2.imshow('Live Stream', frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            return -1
        obj = find_obj(cls_id, predict)
        f = open('cur_inf.txt', 'w', encoding='UTF-8')
        if type(obj) != int and len(obj):
            current_pos = obj.tolist()
            f.write(' '.join(list(map(str, current_pos))) + f' {cls_id}')
        else:
            f.write('-1')

while True:
    do()
