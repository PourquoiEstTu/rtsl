import cv2
import numpy as np
import json
import time

# replace this with where your video dataset is located
PATH = "/windows/Users/thats/Documents/archive/videos"

cap = cv2.VideoCapture(f"{PATH}/08671.mp4")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
while cap.isOpened():
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
 
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('frame', 600,600)
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imshow('frame', thresh)
    frame = cv2.resize(frame, (600,500))
    frame = cv2.GaussianBlur(frame, (7,7), 0)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(frame2, 30, 100)
    # frame = cv2.threshold(frame, 45, 255, cv2.THRESH_BINARY_INV)[1]
    # contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours[1])
    # for c in contours :
    # cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    time.sleep(0.2)
    if cv2.waitKey(40) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
