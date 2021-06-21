# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:34:06 2020

@author: naveen
"""


import numpy as np
import cv2
import pickle

#loading epenCV file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

#intiating webcam 
cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #converting image into gray
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #getting faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=3)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #pridicting faces 
        id_, conf = recognizer.predict(roi_gray)
        # confi sbesically accuracy of the pridiction 
        if conf>=70 and conf <= 80:
            font = cv2.FONT_HERSHEY_SIMPLEX
            print(labels[id_] , conf)
            #lining the face 
            cv2.putText(frame, labels[id_], (x,y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
              
    cv2.imshow('frame',frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()