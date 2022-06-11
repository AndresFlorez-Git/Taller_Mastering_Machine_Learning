
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

###############################################################
#                    Parámetros de red
###############################################################
Tamano_imagen = (224,224)

Capas_convolucionales = 2
Tamano_filtro = (3,3)
No_filtros_por_capa = 32

Capas_densas = 0
neuronas_por_capa = 2


epocas_enetrenamiento = 5
Lote_imagenes_por_epoca = 2


###############################################################
#                    Datos de entrenamiento y test
###############################################################


train_dataset = tf.keras.preprocessing.image_dataset_from_directory('dataset_train',
                                                                shuffle=True,
                                                                batch_size=Lote_imagenes_por_epoca,
                                                                image_size = Tamano_imagen)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory('dataset_test',
                                                                shuffle=True,
                                                                batch_size=Lote_imagenes_por_epoca,
                                                                image_size = Tamano_imagen)


###############################################################
#                    Compilamos la red neuronal
###############################################################

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(Tamano_imagen[0], Tamano_imagen[1], 3 )))
for i in range(Capas_convolucionales):
    model.add(tf.keras.layers.Conv2D(No_filtros_por_capa, Tamano_filtro, activation='relu', kernel_initializer='he_normal', padding='same'))
model.add(tf.keras.layers.Flatten())
for i in range(Capas_densas):
    model.add(tf.keras.layers.Dense(neuronas_por_capa, activation= 'relu', kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(2, activation= 'softmax', kernel_initializer='he_normal'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.summary()


###############################################################
#                    Entrenamos el modelo
###############################################################
results = model.fit(train_dataset,  epochs=epocas_enetrenamiento)

plt.figure()
plt.subplot(2,1,1)
plt.plot( results.history["loss"], label="train_loss")
plt.subplot(2,1,2)
plt.plot( results.history["accuracy"], label="accuracy")
plt.show()




###############################################################
#                    Ponemos a prueba el modelo
###############################################################
test_results = model.evaluate(test_dataset)
print('\n')
print("test accuracy:", test_results[1])
print("El modelo que creaste clasifica bien ", str(round( test_results[1]*10,0)),' de cada 10 imagenes en el contexto adecuado')



###############################################################
#                    Guardamos el modelo creado
###############################################################
model.save("mask_detector_custom.model", save_format="h5")