from tensorflow.keras.models import load_model
from Function_util import detect_and_predict_mask, plot_box, save_pictures
import cv2
import numpy as np




########################################################

#                     NO TOCAR

########################################################



print('Cargando redes neuronales...')

# Se carga el modelo de detección de rostros
modelPath = "Modelo_deteccion_rostros/model.prototxt"
weightsPath = "Modelo_deteccion_rostros/weights.caffemodel"
face_detection_Net = cv2.dnn.readNet(modelPath, weightsPath)

# Se carga el modelo de detección de máscaras
model = load_model('mask_detector1.model')

Tamano_imagen = (224,224)

# Se inicializa el video
print("Iniciando Video...")

# Configuración de la cámara
camera = cv2.VideoCapture(0)


while True:
    True_image, frame = camera.read()
    
    # Usando los modelos de detección de rostro concatenado al modelo de detección de máscaras.
    (Ubicacion, Probabilidades) = detect_and_predict_mask(frame, face_detection_Net, model, Tamano_imagen)
    
    # Se guarda la imagen correctamente etiquetada
    frame = save_pictures(frame, Ubicacion, Probabilidades)
    
    # Se visualiza la imagen captada por la cámara
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(1) & 0xFF
    
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()



