# =============================================================================
# PROBLEM B1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-3
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_model():
    input = tf.keras.layers.Input(shape=(1), dtype=float)
    x = tf.keras.layers.Dense(1)(input)
    output = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    return model

def solution_B1():
    # DO NOT CHANGE THIS CODE
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    Y = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0], dtype=float)

    # YOUR CODE HERE
    model = create_model()
    model.fit(X, Y, epochs=20000)
    prediction = model.predict([-2.0, 10.0])
    prediction = tf.reshape(prediction, [len(prediction)])
    print("MSE:", tf.keras.metrics.mean_squared_error([-1.0, 23.0], prediction))
    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B1()
    model.save("model_B1.h5")
