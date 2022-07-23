# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd


# def remove_stopwords(text: str):
#   stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
#   sentence = text.lower()
#   sentence_split = sentence.split()
#   new_sentence = []
#   for i in sentence_split:
#       if i in stopwords:
#           continue
#       new_sentence.append(i)
        
#   sentence = " ".join(new_sentence)
#   return sentence


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')
    df = bbc

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"

    # Convert category to int
    # y_train = df["category"].astype("category").cat.codes.values
    # Hmm.. let's try using tokenizer too.
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(df["category"].values)
    y_train = label_tokenizer.texts_to_sequences(df["category"].values)
    y_train = tf.reshape(y_train, len(y_train))
    print(y_train)

    # # Preprocess text
    # final_text = []
    # for i in df["text"].values:
    #   final_text.append(i)

    final_text = df["text"].values

    # Fit your tokenizer with training data
    tokenizer =  Tokenizer(num_words=vocab_size,
                          oov_token=oov_tok)
    tokenizer.fit_on_texts(final_text)
    training_seq = tokenizer.texts_to_sequences(final_text)
    training_seq = pad_sequences(training_seq, maxlen=max_length, 
                                 truncating=trunc_type, padding=padding_type)
    x_train = training_seq

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        # With LSTM
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        # tf.keras.layers.LSTM(128),

        # Without LSTM
        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class CustomCallback(tf.keras.callbacks.Callback):
        total_above_target = 0
        def on_epoch_end(self, epoch, logs=None):
            if logs["accuracy"] > 0.91 and logs["val_accuracy"] > 0.91:
                self.total_above_target +=1
                if self.total_above_target > 3:
                  self.model.stop_training = True
            else:
                self.total_above_target = 0

    model.fit(x=x_train, y=y_train,
              epochs=1000, batch_size=23,
              validation_split = 1 - training_portion,
              callbacks=[CustomCallback()])

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")