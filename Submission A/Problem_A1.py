# =================================================================================
# PROBLEM A1 
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1].
# Do not use lambda layers in your model.
# 
# Desired loss (MSE) < 1e-4
# =================================================================================


import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_model():
    input = tf.keras.layers.Input(shape=(1), dtype=float)
    x = tf.keras.layers.Dense(3, activation="linear")(input)
    output = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    return model


def solution_A1():
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)


    # YOUR CODE HERE
    model = create_model()
    prediction = model.predict([-2, 10])
    model.fit(X, Y, epochs=20000)
    print(model.predict([-2.0, 10.0]))
    prediction = model.predict([-2, 10])
    prediction = tf.reshape(prediction, [len(prediction)])
    print("MSE:", tf.keras.metrics.mean_squared_error([7.0, 19.0], prediction))
    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A1()
    model.save("model_A1.h5")