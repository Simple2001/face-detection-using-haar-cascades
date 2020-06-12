import cv2
import numpy as np
face_classifier = cv2.CascadeClassifier('C:/Users/saran/cascades/haarcascade_frontalface_default.xml')
path = r'C:/Users/saran/images.jpeg'
resized = cv2.imread(path)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
while True:
    faces = face_classifier.detectMultiScale(gray, 1.048525, 6)
    for (x,y,w,h) in faces:
        cv2.rectangle(resized, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()