import cv2
import numpy as np
import json

PATH = "/windows/Users/thats/Documents/archive/videos"

cap = cv2.VideoCapture(f"{PATH}/00335.mp4")
 
while cap.isOpened():
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
    cv2.imshow('frame', frame)
    if cv2.waitKey(40) == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
