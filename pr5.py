import csv

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential


SIZE_TRAIN_DATA = 300
SIZE_TEST_DATA = 50

def dataGen(size):
    data = []
    targets = []
    for i in range(size):
        X = np.random.normal(3, 10)
        e = np.random.normal(0, 0.3)
        data.append((X**2+e, np.sin(X/2), np.cos(2*X)+e, X-3+e, np.fabs(X)+e, (X**3)/4+e))
        targets.append((-X+e))
    return data, targets


def writeToCSV(path, data):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',',
                        quoting=csv.QUOTE_MINIMAL)
        try:
            for item in data:
                writer.writerow(item)
        except Exception as ex:
            writer.writerows(map(lambda x: [x], data))


train_data, train_targets = dataGen(SIZE_TRAIN_DATA)
test_data, test_targets = dataGen(SIZE_TEST_DATA)
writeToCSV('train_data.csv', np.round(train_data, 4))
writeToCSV('train_targets.csv', np.round(train_targets, 4))
writeToCSV('test_data.csv', np.round(test_data, 4))
writeToCSV('test_targets.csv', np.round(test_targets, 4))

mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std

test_data -= mean
test_data /= std

# encode
main_input = Input(shape=(6,))

encoded = Dense(65, activation='relu')(main_input)
encoded = Dense(30, activation='relu')(encoded)
encoded = Dense(3, activation='relu')(encoded)

# decode
input_dec = Input(shape=(3,))

decoded = Dense(30, activation='relu', name='dec_layer_1')(encoded)
decoded = Dense(60, activation='relu', name='dec_layer_2')(decoded)
decoded = Dense(60, activation='relu', name='dec_layer_3')(decoded)
decoded = Dense(6, name="dec_layer_4")(decoded)
# regression
predicted = Dense(60, activation='relu')(encoded)
predicted = Dense(60, activation='relu')(predicted)
predicted = Dense(60, activation='relu')(predicted)
predicted = Dense(60, activation='relu')(predicted)
predicted = Dense(1)(predicted)

encoder = Model(main_input, encoded)
regr_model = Model(main_input, predicted)
autoencoder = Model(main_input, decoded)

autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])
autoencoder.fit(train_data, train_data, epochs=100, batch_size=5, shuffle=True, validation_data=(test_data, test_data))
encode_data = encoder.predict(test_data)

decoder = autoencoder.get_layer('dec_layer_1')(input_dec)
decoder = autoencoder.get_layer('dec_layer_2')(decoder)
decoder = autoencoder.get_layer('dec_layer_3')(decoder)
decoder = autoencoder.get_layer('dec_layer_4')(decoder)
decoder = Model(input_dec, decoder)
decode_data = decoder.predict(encode_data)

regr_model.compile(optimizer="adam", loss="mse", metrics=['mae'])
regr_model.fit(train_data, train_targets, epochs=100, batch_size=5, validation_data=(test_data, test_targets))
predict_data = regr_model.predict(test_data)

decoder.save('decoder.h5')
encoder.save('encoder.h5')
regr_model.save('regression.h5')


writeToCSV('encoded_data.csv', np.round(encode_data, 4))
writeToCSV('decoded_data.csv', np.round(decode_data, 4))
writeToCSV('predicted_data.csv', np.round(predict_data, 4))