import cv2
import os
import numpy as np
import time

def obtenerModelo(method, facesData, labels):
    emotion_recognizer = cv2.face.EigenFaceRecognizer_create()

    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time()-inicio
    print("Tiempo de entrenamiento ( " + method + " ): ", tiempoEntrenamiento)

    # Almacenando el modelo obtenido
    emotion_recognizer.write("modelo" + method + ".xml")

dataPath= os.getcwd() + '/data'
personList = os.listdir(dataPath)
print('Lista de personas: ', personList)

labels = []
facesData = []
label = 0

for nameDir in personList:
    personPath = dataPath + '/' + nameDir

    for fileName in os.listdir(personPath):
        #print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath + '/' + fileName, 0)
        cv2.imshow('image',image)
        cv2.waitKey(10)
    label = label + 1

obtenerModelo('EigenFaces',facesData,labels)
