import cv2
import os
import numpy as np
import imutils


face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('modeloEigenFaces.xml')

dataPath= os.getcwd() + '/data'
imagePaths = os.listdir(dataPath)

cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION)
cap.set(3,640)
cap.set(4,480)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False: break
    #frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])
    #nFrame = frame
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = auxFrame[y:y+h,x:x+w]
        face = cv2.resize(face, (200, 200), interpolation=cv2.INTER_CUBIC)
        grayface = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = face_recognizer.predict(grayface)
        print(result)

        if result[1] < 4000:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y),(x + w, y + h),(0, 255, 0), 2)
            nFrame = frame
        else:
            cv2.putText(frame,'Unknown', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x + w, y + h), (0, 0, 255), 2)
            nFrame = cv2.hconcat([frame,np.zeros((480, 300, 3),dtype=np.uint8)])
    cv2.imshow('nFrame',frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
