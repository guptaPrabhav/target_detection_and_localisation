import cv2
from ultralytics import YOLO

import time
frame_rate = 5
prev = 0

video_capture = cv2.VideoCapture("rtsp://192.168.0.72:8901/live")

model = YOLO('/home/prabhav/Desktop/IISc/yolov8m.pt')

while(True):
    time_elapsed = time.time() - prev
    ret, frame = video_capture.read()

    if time_elapsed>(1/frame_rate):
        prev = time.time()


        cv2.imshow('original_video', frame)

        results = model.predict(frame, conf=0.5)
        # print(results)
        annotaed_frame = results[0].plot()
        cv2.imshow('drone_video', annotaed_frame)
    
        cv2.waitKey(1)