from tensorflow.keras.models import load_model
from Function_util import detect_and_predict_mask, plot_box
import cv2
import numpy as np


###############################################################
#                    Modelo a utilizar
###############################################################



nombre_modelo = "mask_detector_custom" #  <<<-------- Modifica aquí el nombre del modelo que quieres evaluar


Tamano_imagen = (64,64) # <<<-------- Pon el mismo tamaño de imagen que configuraste en el entrenamiento




###############################################################
#                    Resultados del modelo // NO TOCAR A PARTIR DE AQUI
###############################################################



# Se carga el modelo de detección de máscaras
model = load_model(nombre_modelo+'.model')   #  <<<---------- Modifica aquí el nombre del modelo que quieres utilizar


# Se carga el modelo de detección de rostros

print('Cargando redes neuronales...')

modelPath = "Modelo_deteccion_rostros/model.prototxt"
weightsPath = "Modelo_deteccion_rostros/weights.caffemodel"
face_detection_Net = cv2.dnn.readNet(modelPath, weightsPath)

# Se inicializa el video
print("Iniciando Video...")

# Configuración de la cámara
camera = cv2.VideoCapture(0)


while True:
    True_image, frame = camera.read()
    
    # Usando los modelos de detección de rostro concatenado al modelo de detección de emociones.
    (Ubicacion, Probabilidades) = detect_and_predict_mask(frame, face_detection_Net, model)
    
    # Se pinta sobre la imagen la ubicacion y la probabilidad.
    frame = plot_box(frame, Ubicacion, Probabilidades)
    
    
    # Se visualiza la imagen captada por la cámara
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF
    
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()



