#!/usr/bin/env python3

import numpy as np
import cv2



T = "PLEASE BE ALERT"
font = cv2.FONT_HERSHEY_SIMPLEX

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')   #Import Sorce file for huaman eye and face for detection
cam = cv2.VideoCapture(0)

while(1):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    if faces == ():
            cv2.putText(img,str(T),(100,500), font, 3,(30,30,255),6,cv2.LINE_AA)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,100,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye.detectMultiScale(roi_gray)
    
        if eyes == ():
            cv2.putText(img,str(T),(100,500), font, 3,(30,30,255),6,cv2.LINE_AA)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(50,255,255),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()  

