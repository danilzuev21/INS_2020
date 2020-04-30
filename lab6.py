import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

strings = ["This film made a good impression on me. I'm glad I bought a ticket and watched this movie! "
           "perfect beautiful great",
           "This movie makes you think about the role of man on this planet. I recommend watching the movie. "
           "wonder exiting amazing",
           "Sarik Andreasyan once again confirmed that he is the king of the Comedy genre. "
           "This movie is a must-see for people with a sense of humor. impossible wonderful laugh",
           "This Quentin Tarantino movie is as usual full of senseless violence and inappropriate behavior. "
           "This movie is terrible and should be banned from cinemas. poor shoddy low grade",
           "The graphics in this movie are terrible. It is clear that this film was made with a slipshod hand. "
           "awful terrible substandart",
           "The characters in this film have absolutely no motivation, their actions are devoid of any meaning. "
           "I hope the Director of this film will not make any more films. worse horrible unfit"]
values = [1, 1, 1, 0, 0, 0]


def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def prepare_data(dimension):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, dimension)
    targets = np.array(targets).astype("float32")
    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]
    return (train_x, train_y), (test_x, test_y)


def build_model(dimension):
    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(dimension,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def model_fit(train_x, train_y, test_x, test_y, dimension):
    model = build_model(dimension)
    H = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
    return H


def test_dim(dimension):
    (train_x, train_y), (test_x, test_y) = prepare_data(dimension)
    H = model_fit(train_x, train_y, test_x, test_y, dimension)
    return H.history['val_accuracy'][-1]


def draw_plot(H):
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    print(len(loss))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test_dimensions():
    dimensions = [10, 100, 500, 1000, 5000, 10000]
    val_accuracies = []
    for dim in dimensions:
        val_accuracies.append(test_dim(dim))
    plt.plot(dimensions, val_accuracies, 'b', label='Validation acc')
    plt.title('Validation accuracy')
    plt.xlabel('Dimensions')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test_10000_dim():
    (train_x, train_y), (test_x, test_y) = prepare_data(10000)
    H = model_fit(train_x, train_y, test_x, test_y, 10000)
    draw_plot(H)


def text_load():
    dictionary = dict(imdb.get_word_index())
    test_x = []
    test_y = np.array(values).astype("float32")
    for string in strings:
        words = string.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('\n', ' ').split()
        num_words = []
        for word in words:
            word = dictionary.get(word)
            if word is not None and word < 10000:
                num_words.append(word)
        test_x.append(num_words)
    print(test_x)
    test_x = vectorize(test_x)
    # print(test_x)
    model = build_model(10000)
    (train_x, train_y), (s1, s2) = prepare_data(10000)
    model.fit(train_x, train_y, epochs=2, batch_size=500)
    val_loss, val_acc = model.evaluate(test_x, test_y)
    print("Validation accuracy is %f" % val_acc)
    predictions = model.predict(test_x)
    plt.title("Predictions")
    plt.plot(test_y, 'r', label='Real values')
    plt.plot(predictions, 'b', label='Predictions')
    plt.legend()
    plt.show()
    plt.clf()


test_dimensions()
test_10000_dim()
text_load()
