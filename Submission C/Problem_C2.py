# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf

class TargetCallbacks(tf.keras.callbacks.Callback):
    MAX_CONSECUTIVE = 5
    consecutive_target = 0

    target = "accuracy"
    val_target ="val_accuracy"

    desired = 0.91

    def on_epoch_end(self, epoch, logs=None):
        if (logs[self.target] > self.desired) and (logs[self.val_target] > self.desired):
            self.consecutive_target += 1
            if self.consecutive_target >= self.MAX_CONSECUTIVE:
                self.model.stop_training=True
        else:
            self.consecutive_target = 0


def solution_C2():
    mnist = tf.keras.datasets.mnist

    # NORMALIZE YOUR IMAGE HERE
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255, x_test/255

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    input = tf.keras.layers.Input(shape=(28,28))
    x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    output = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    # COMPILE MODEL HERE
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # TRAIN YOUR MODEL HERE
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=200,
              callbacks=[TargetCallbacks()])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("model_C2.h5")