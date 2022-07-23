# =============================================================================
# PROBLEM C1
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
# Desired loss (MSE) < 1e-4
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras


class TargetCallbacks(keras.callbacks.Callback):
    MAX_CONSECUTIVE = 5
    consecutive_target = 0

    target = "mean_squared_error"
    val_target ="val_mean_squared_error"

    desired = 1e-8

    def on_epoch_end(self, epoch, logs=None):
        if (logs[self.target] < self.desired) and (logs[self.val_target] < self.desired):
            self.consecutive_target += 1
            if self.consecutive_target >= self.MAX_CONSECUTIVE:
                self.model.stop_training=True
        else:
            self.consecutive_target = 0

def solution_C1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # YOUR CODE HERE
    # Calculated manually
    prediction = [-2.0, 10.0]
    g_truth = [0, 6]

    input = tf.keras.layers.Input(shape=(1))
    x = tf.keras.layers.Dense(1)(input)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    model.fit(X, Y, epochs=20000, validation_data=(prediction, g_truth), callbacks=[TargetCallbacks()])

    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C1()
    model.save("model_C1.h5")
