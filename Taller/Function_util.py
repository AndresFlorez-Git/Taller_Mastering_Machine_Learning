from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np


def detect_and_predict_mask(frame, faceNet, emotionNet,Tamano_imagen):
    # Se modifican las dimensiones de la imagen original y se crea el blob como imagen
    # procesada.
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    # Se realizan las predicciones obtenidas por el modelo de deteccion de rostros
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # Se inicializan listas
    faces = []
    locs = []
    preds = []

    # Para cada una de las detecciones de rostros realizada, se debe predecir la probabilidad de
    # portar o no máscara.
    for i in range(0, detections.shape[2]):
        # Probabilidad de las detecciones de rostro
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Se obtienen las coordenadas en la imagen de la deteccion del rostro
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Se extrae la porcion de la imagen correspondiente al rostro
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, Tamano_imagen)
            face = img_to_array(face)
            face = preprocess_input(face)

            #Se agregan a la lista
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Si hay rostros en la imagen, entonces se realiza la deteccion de máscara
    if len(faces) > 0:
        # Se realizan las predicciones del modelo de detección de máscara
        faces = np.array(faces, dtype="float32")
        preds = emotionNet.predict(faces, batch_size=32)

    # Se retorna la ubicación de los rostros y la probabilidad asociada a la detección de máscaras
    return (locs, preds)


def plot_box(frame,locs, preds):
    
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (withoutMask, mask) = pred
        prediction = np.argmax((withoutMask, mask))
        print(pred)
        if prediction == 0: label = 'Sin Mascara: ' + str(round(withoutMask,2))
        if prediction == 1: label = 'Con Mascara: ' + str(round(mask,2))
        color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
        cv2.putText(frame, label, (startX - 60, endY + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame


def save_pictures(frame,locs, preds):
    
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (withoutMask, mask) = pred
        prediction = np.argmax(( mask,withoutMask))
        print(prediction)
        if prediction == 0: label = 'WithoutMask' + str(round(withoutMask,2))+str(np.random.randint(0,1000000))
        if prediction == 1: label = 'Mask' + str(round(mask,2))+str(np.random.randint(0,1000000))
        path = 'dataset/con_mascara/' if prediction == 1 else 'dataset/sin_mascara/'
        img = frame[startY-30:endY+30,startX-30:endX+30]
        cv2.imwrite(path+label+'.jpg', img)
        frame = img
    return frame