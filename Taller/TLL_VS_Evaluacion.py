from tensorflow.keras.models import load_model
import tensorflow as tf



###############################################################
#                    Parámetros de evaluación
###############################################################


nombre_modelo = "mask_detector_custom" #  <<<-------- Modifica aquí el nombre del modelo que quieres evaluar


Tamano_imagen = (64,64) # <<<-------- Pon el mismo tamaño de imagen que configuraste en el entrenamiento


 





###############################################################
#                    Datos de prueba // NO TOCAR A PARTIR DE AQUI
###############################################################
Lote_imagenes_por_epoca = 8
test_dataset = tf.keras.preprocessing.image_dataset_from_directory('dataset_test',
                                                                shuffle=True,
                                                                batch_size=Lote_imagenes_por_epoca,
                                                                image_size = Tamano_imagen)


###############################################################
#                    Ponemos a prueba el modelo
###############################################################
model = load_model(nombre_modelo+'.model')  

test_results = model.evaluate(test_dataset)
print('\n')
print("test accuracy:", test_results[1])
print("El modelo que creaste clasifica bien ", str(round( test_results[1]*10,0)),' de cada 10 imagenes en el contexto adecuado')

