from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_epochs = 15
kernel_sizes = [2, 4, 6]
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512
dropout_flags = [True, False]
count = 1

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

num_train, depth, height, width = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_test /= np.max(X_train)

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)


def fit_model(kernel_size, dropout_enabled):
    global count
    inp = Input(shape=(depth, height, width))

    conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size), activation='relu', padding='same')(inp)
    conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), activation='relu', padding='same')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    if dropout_enabled:
        drop_1 = Dropout(drop_prob_1)(pool_1)
    else:
        drop_1 = pool_1

    conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size), activation='relu', padding='same')(drop_1)
    conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size), activation='relu', padding='same')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    if dropout_enabled:
        drop_2 = Dropout(drop_prob_2)(pool_2)
    else:
        drop_2 = pool_1

    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    if dropout_enabled:
        drop_3 = Dropout(drop_prob_2)(hidden)
    else:
        drop_3 = hidden
    out = Dense(num_classes, activation='softmax')(drop_3)

    model = Model(input=inp, output=out)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    H = model.fit(X_train, Y_train,
                  batch_size=batch_size, epochs=num_epochs,
                  verbose=1, validation_split=0.1)
    count = count + 1
    name = "model%i.h5" % count
    model.save(name)
    return H


def test_dropout_layer():
    for dropout_flag in dropout_flags:
        H = fit_model(3, dropout_flag)
        str = "enabled" if dropout_flag else "disabled"
        draw_test(H, "Test with the dropout layer "+str)


def test_kernel_size():
    for size in kernel_sizes:
        H = fit_model(size, True)
        draw_test(H, "Test with a %ix%i core size" % (size, size))


def draw_test(H, title):
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


test_dropout_layer()
test_kernel_size()
