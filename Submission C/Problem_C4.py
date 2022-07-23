# =====================================================================================================
# PROBLEM C4
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
#
# Do not use lambda layers in your model.
#
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TargetCallbacks(tf.keras.callbacks.Callback):
    MAX_CONSECUTIVE = 5
    consecutive_target = 0

    target = "accuracy"
    val_target ="val_accuracy"

    desired = 0.75

    def on_epoch_end(self, epoch, logs=None):
        if (logs[self.target] > self.desired) and (logs[self.val_target] > self.desired):
            self.consecutive_target += 1
            if self.consecutive_target >= self.MAX_CONSECUTIVE:
                self.model.stop_training=True
        else:
            self.consecutive_target = 0

def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    file_object = open("sarcasm.json", "r")
    json_content = file_object.read()
    arr = json.loads(json_content)
    for i in arr:
        sentences.append(str(i["headline"]).lower())
        labels.append(int(i["is_sarcastic"]))
    labels = np.asarray(labels)

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token=oov_tok
    )
    tokenizer.fit_on_texts(sentences)

    all_seq = tokenizer.texts_to_sequences(sentences)
    all_seq = pad_sequences(
        all_seq,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type,
    )

    train_x = all_seq[:training_size]
    train_y = labels[:training_size]

    test_x = all_seq[training_size:]
    test_y = labels[training_size:]

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Input(shape=(max_length)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y), callbacks=[TargetCallbacks()])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C4()
    model.save("model_C4.h5")
