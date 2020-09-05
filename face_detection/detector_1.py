import os
import numpy as np
import cv2, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists("TrainingData"):
    os.makedirs('TrainingData')

def capture(name, id):
    sample = 0
    db_path = os.path.join(BASE_DIR, "TrainingData")
    # path including \TrainingData where we will store directory corresponding individual picture entry

    # loading classifier
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # to generate a unique directory for each entry
    name_id = name + " (" + str(id) + ")"
    print(f"System taking sample picture for {name_id}!!\n")
    # this id_path will be directory to store image
    id_path = db_path + os.sep + name_id
    # id_path represents db_path / name_id directory where we store sampled of images
    if not os.path.exists(id_path):
        os.makedirs(id_path)
    video = cv2.VideoCapture(cv2.CAP_DSHOW)
    time.sleep(2)

    if not video.isOpened():
        raise IOError("Cannot open webcam")
    #capture images
    while True:

        _, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25)
        if len(faces) > 0:
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                sample += 1
                cropped_gray = gray[y: y + h, x:x + w]
                resized_gray = cv2.resize(cropped_gray, (100, 100))
                print(resized_gray.shape)
                cv2.imshow("new", resized_gray)
                print(cropped_gray.shape)
                cv2.imwrite(id_path + os.sep + str(sample) + ".jpg", resized_gray)

        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
        if sample > 20:
            break
    video.release()
    cv2.destroyAllWindows()
    print("DOne\n")


def user_input():
    name = input("Enter full Name")
    while True:
        try:
            unique_id = int(input("Enter numeric unique id:\t"))
        except:
            print("Id must be numeric:\t")
            continue
        else:
            break
    print("System taking sample picture of you!!\n")
    capture(name, unique_id)
user_input()
