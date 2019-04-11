from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten


def create_rectangular_model(input_window_width):
    model = Sequential()

    model.add(Conv1D(16, 4, activation='linear', input_shape=(input_window_width, 1), padding='valid', strides=1, name='conv_1'))
    model.add(Conv1D(16, 4, activation='linear', input_shape=(input_window_width, 1), padding='valid', strides=1, name='conv_2'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dense(3072, activation='relu', name='dense_2'))
    model.add(Dense(2048, activation='relu', name='dense_3'))
    model.add(Dense(512, activation='relu', name='dense_4'))

    model.add(Dense(3, activation='linear', name='dense_out'))

    model.compile(loss='mse', optimizer='adam')

    return model
