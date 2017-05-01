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
from keras.layers import LSTM
from keras.optimizers import SGD , Adam
import tensorflow as tf
import keras.backend as K
from hyperas.distributions import choice, uniform, conditional
from keras.layers.normalization import BatchNormalization
__author__ = 'JOnathan Hilgart'


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    import numpy as np
    x = np.load('training_x.npy')
    x = x.reshape(1,15000,2)
    y = np.load('training_y.npy')
    y = y.reshape(1,15000, 9)
    x_train ,x_test  = np.split(x,2, axis=1)
    y_train , y_test= np.split(y,2, axis=1)

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
    model_lstm = Sequential()
    model_lstm .add(LSTM({{choice([64, 126, 256, 512, 1024])}}, dropout={{uniform(0, .5)}},
                         batch_input_shape=(1,x_train.shape[1], 2),
                     recurrent_dropout={{uniform(0, .5)}},return_sequences = True))
    model_lstm.add(BatchNormalization())
    condition = conditional({{choice(['one','two','three', 'four'])}})

    if condition == 'one':
        pass
    elif condition == 'two':
        model_lstm .add(LSTM({{choice([64, 126, 256, 512, 1024])}}, dropout={{uniform(0, .5)}},
                     recurrent_dropout={{uniform(0, .5)}},
                     return_sequences = True))
        model_lstm.add(BatchNormalization())
    elif condition  == 'three':
        model_lstm .add(LSTM({{choice([64, 126, 256, 512, 1024])}}, dropout={{uniform(0, .5)}},
                     recurrent_dropout={{uniform(0, .5)}},
                     return_sequences = True))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Dense({{choice([126, 256, 512, 1024])}}))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Activation({{choice(['relu','tanh','sigmoid'])}}))
    elif condition == 'four':
        model_lstm .add(LSTM({{choice([64, 126, 256, 512, 1024])}}, dropout={{uniform(0, .5)}},
                     recurrent_dropout={{uniform(0, .5)}},
                     return_sequences = True))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Dense({{choice([126, 256, 512, 1024])}}))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Activation({{choice(['relu','tanh','sigmoid'])}}))
        model_lstm.add(Dense({{choice([126, 256, 512, 1024])}}, activation='relu'))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Activation({{choice(['relu','tanh','sigmoid'])}}))


    model_lstm .add(Dense(9, activation='linear',name='dense_output'))
    adam = Adam(clipnorm=.5, clipvalue=.5)
    model_lstm .compile(loss='mean_squared_error', optimizer=adam,
                        metrics=['accuracy'])
    model_lstm.summary()



    model_lstm.fit(x_train, y_train,
              batch_size=1,
              epochs=1,
              verbose=1,
              validation_data=(x_test, y_test))
    loss, acc = model_lstm.evaluate(x_test, y_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model_lstm}


if __name__ == '__main__':
    import gc; gc.collect()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=40,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
