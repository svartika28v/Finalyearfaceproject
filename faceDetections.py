# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:34:06 2020

@author: naveen
"""

import cv2

trained_model = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
openCam = cv2.VideoCapture(0)
while True:
    frame_read, frame = openCam.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_array_coordinates = trained_model.detectMultiScale(gray_image)
    for x, y, w, h in face_array_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
    cv2.imshow("face detection", frame)
    key = cv2.waitKey(1)
    if (key == 81 or key == 113):
        break
openCam.release()
