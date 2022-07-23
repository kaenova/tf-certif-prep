# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    # YOUR CODE HERE
    train_data = imdb["train"]
    training_sentences = []
    training_labels = []

    test_data = imdb["test"]
    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    # YOUR CODE HERE

    vocab_size = 10000
    embedding_dim = 16
    max_length = 256
    trunc_type = 'post'
    oov_tok = "<OOV>"

    # NOTE: Text to sequence ready
    tokenizer = Tokenizer(num_words=vocab_size,
                          oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    tokenizer.fit_on_texts(testing_sentences)
    training_seq = tokenizer.texts_to_sequences(training_sentences)
    training_seq = pad_sequences(training_seq, maxlen=max_length, truncating=trunc_type)
    test_seq = tokenizer.texts_to_sequences(testing_sentences)
    test_seq = pad_sequences(test_seq, maxlen=max_length, truncating=trunc_type)

    word_index = tokenizer.word_index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    model = tf.keras.Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs["accuracy"] > 0.83 and logs["val_accuracy"] > 0.83:
                self.model.stop_training = True

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(x=training_seq, y=np.asarray(training_labels),
              epochs=50, batch_size=32,
              validation_data=(test_seq, np.asarray(testing_labels)),
              callbacks=[early_stopping, CustomCallback()])
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A4()
    model.save("model_A4.h5")