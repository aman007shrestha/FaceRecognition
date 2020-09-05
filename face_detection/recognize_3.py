from cv2 import cv2
import pickle
import numpy as np
import os


face_cascade = cv2.CascadeClassifier("HaarCascade/haarcascade_frontalface_alt2.xml")
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print(BASE_DIR)
# a = os.path.join(BASE_DIR, "HaarCascade")
# print(a)
# print(os.path.exists("HaarCascade/haarcascade_frontalface_alt2.xml"))
# print(os.path.exists(os.path.join(a, "haarcascade_frontalface_alt2.xml")))
recognizer = cv2.face.LBPHFaceRecognizer_create()
# # print(type(recognizer))
# # print(help(recognizer.read))
recognizer.read("trainner.yml")
#
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}
# print(labels) # {0: 'emilia-clarke', 1: 'peter-dinklage'}
cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)
#
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)  # helps in mirror image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 80:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
    cv2.imshow('face recognition', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()