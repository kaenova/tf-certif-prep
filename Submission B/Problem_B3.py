# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator


def solution_B3():
    data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
    urllib.request.urlretrieve(data_url, 'rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    TRAINING_DIR = "data/rps/"
    batch_size = 20
    val_split = 0.2
    seed = 1
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=val_split,
        fill_mode='nearest')

    validaiton = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=val_split,
        fill_mode='nearest')

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "categorical"
    train_generator = training_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                           shuffle=True,
                                                           batch_size=batch_size,
                                                           class_mode='categorical',
                                                           target_size=(150, 150),
                                                           seed=seed,
                                                           subset="training")

    validation_generator = training_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                                shuffle=True,
                                                                batch_size=batch_size,
                                                                class_mode='categorical',
                                                                target_size=(150, 150),
                                                                seed=seed,
                                                                subset="validation")

    model = tf.keras.models.Sequential([
        # YOUR CODE HERE, end with 3 Neuron Dense, activated by softmax
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    class CustomCallback(tf.keras.callbacks.Callback):
        total_above_target = 0

        def on_epoch_end(self, epoch, logs=None):
            if logs["val_accuracy"] > 0.83 and logs["accuracy"] > 0.83:
                self.total_above_target += 1
                if self.total_above_target > 3:
                    self.model.stop_training = True

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_generator, epochs=200,
              validation_data=validation_generator,
              callbacks=[CustomCallback()])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B3()
    model.save("model_B3.h5")