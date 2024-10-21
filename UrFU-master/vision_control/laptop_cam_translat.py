import threading
import time
import cv2 as cv


def show_camera():
    ip_camera_url_left = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"
    cap = cv.VideoCapture(ip_camera_url_left)
    while 1:
        ret, frame = cap.read()
        if ret:
            cv.imshow('Output Image', frame)
            cv.waitKey(1)


def do_some_movement():
    while 1:
        print("go forw")
        time.sleep(1)
        print("go back")
        time.sleep(1)


threads = []  # 创建一个线程序列
t1 = threading.Thread(target=show_camera, args=())  # 摄像头数据收集处理线程
threads.append(t1)  # 将线程添加到线程队列中
t2 = threading.Thread(target=do_some_movement, args=())  # 新建蓝牙线程
threads.append(t2)

for t in threads:
    t.start()  # 启动线程
    time.sleep(0.05)