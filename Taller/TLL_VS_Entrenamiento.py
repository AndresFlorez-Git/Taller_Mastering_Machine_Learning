import shutil
import os
from sklearn.utils import shuffle
import tensorflow as tf


###############################################################
#                    Parámetros de red
###############################################################
Tamano_imagen = (64,64)

Capas_convolucionales = 2
Tamano_filtro = (3,3)
No_filtros_por_capa = 3

Capas_densas = 0
neuronas_por_capa = 2



nombre_modelo = "mask_detector_custom"




###############################################################
#                    Datos de entrenamiento // NO TOCAR A PARTIR DE AQUI
###############################################################


epocas_enetrenamiento = 2
Lote_imagenes_por_epoca = 8


src_path_mask = 'dataset/con_mascara/'
src_path_without_mask = 'dataset/sin_mascara/'

dst_path_mask = 'dataset_train/con_mascara/'
dst_path_without_mask = 'dataset_train/sin_mascara/'

list_with_Mask = os.listdir(src_path_mask)
list_without_Mask = os.listdir(src_path_without_mask)


for i in list_with_Mask:
    shutil.move(src_path_mask+i,dst_path_mask+i)
for i in list_without_Mask:
    shutil.move(src_path_without_mask+i,dst_path_without_mask+i)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory('dataset_train',
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




###############################################################
#                    Guardamos el modelo creado
###############################################################
model.save(nombre_modelo+".model", save_format="h5")