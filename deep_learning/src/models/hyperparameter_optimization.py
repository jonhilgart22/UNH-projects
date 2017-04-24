from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from hyperas import optim
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from hyperas.distributions import choice, uniform, conditional
__author__ = 'JOnathan Hilgart'


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    import numpy as np
    x = np.load('training_x.npy')
    y = np.load('training_y.npy')
    x_train = x[:15000,:]
    y_train = y[:15000,:]
    x_test = x[15000:,:]
    y_test = y[15000:,:]
    return x_train, y_train, x_test, y_test


def model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model_mlp = Sequential()
    model_mlp.add(Dense({{choice([126, 256, 512, 1024])}},
                        activation='relu', input_shape= (2,)))
    model_mlp.add(Dropout({{uniform(0, .5)}}))
    model_mlp.add(Dense({{choice([126, 256, 512, 1024])}}))
    model_mlp.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model_mlp.add(Dropout({{uniform(0, .5)}}))
    model_mlp.add(Dense({{choice([126, 256, 512, 1024])}}))
    model_mlp.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model_mlp.add(Dropout({{uniform(0, .5)}}))
    model_mlp.add(Dense({{choice([126, 256, 512, 1024])}}))
    model_mlp.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model_mlp.add(Dropout({{uniform(0, .5)}}))
    model_mlp.add(Dense(9))
    model_mlp.add(Activation('softmax'))
    model_mlp.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})



    model_mlp.fit(x_train, y_train,
              batch_size={{choice([16, 32, 64, 128])}},
              epochs=100,
              verbose=2,
              validation_data=(x_test, y_test))
    score, acc = model_mlp.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model_mlp}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
