from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow.keras.callbacks
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from var5 import gen_data


class MyCustomCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, x_train, y_train):
        super(MyCustomCallback, self).__init__()
        self.test_data = test_data
        self.test_labels = test_labels
        self.values = []

    def on_epoch_end(self, epoch, logs=None):
        count = 0
        predicted = self.model.predict(self.test_data)
        for i in range(len(predicted)):
            if abs(self.test_labels[i] - predicted[i]) >= 0.1:
                count += 1
        self.values.append(count)

    def on_train_end(self, logs=None):
        print(self.values)
        plt.plot(self.values)
        plt.ylabel('Number of observations')
        plt.xlabel('Epochs')
        plt.savefig('num_obs.png')


size = 3000
test_count = size // 5
data, labels = gen_data(size)
labels = np.asarray(labels).flatten()

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

temp = list(zip(data, labels))
np.random.shuffle(temp)
data, labels = zip(*temp)
data = np.asarray(data).reshape(size, 50, 50, 1)
labels = np.asarray(labels).flatten()

test_data = data[:test_count]
test_labels = labels[:test_count]
train_data = data[test_count:size]
train_labels = labels[test_count:size]
print(test_labels)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
H = model.fit(train_data, train_labels, epochs=12, batch_size=128,
              validation_data=(test_data, test_labels), callbacks=[MyCustomCallback(test_data, test_labels)])
