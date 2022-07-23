# ======================================================================================================
# PROBLEM A3 
#
# Build a classifier for the Human or Horse Dataset with Transfer Learning. 
# The test will expect it to classify binary classes.
# Note that all the layers in the pre-trained model are non-trainable.
# Do not use lambda layers in your model.
#
# The horse-or-human dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
# Inception_v3, pre-trained model used in this problem is developed by Google.
#
# Desired accuracy and validation_accuracy > 93%.
# =======================================================================================================

import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3


def solution_A3():
    inceptionv3 = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    urllib.request.urlretrieve(inceptionv3, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
    local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=local_weights_file)
    last_layer = pre_trained_model.get_layer('mixed7')
    for layer in pre_trained_model.layers:
        layer.trainable = False


    data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
    urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/horse-or-human')

    data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
    urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    x = layers.Flatten()(last_layer.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    train_dir = 'data/horse-or-human'
    validation_dir = 'data/validation-horse-or-human'

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=10,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      zoom_range=0.1,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(directory=validation_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

    model.fit(train_generator, validation_data=val_generator, epochs=5)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A3()
    model.save("model_A3.h5")