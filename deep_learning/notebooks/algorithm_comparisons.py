#! usr/bin/env python
import pandas as pd
import numpy as np
from py_geohash_any import geohash as gh
import datetime
import random
import numpy as np
from collections import deque
import time
from keras.layers.normalization import BatchNormalization
import json
from collections import defaultdict
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import InputLayer
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD , Adam
import tensorflow as tf
import pickle
from operator import itemgetter
import sys
sys.path.insert(0, '../data/') ## for running on local
import auxiliary_functions, make_dataset
from auxiliary_functions import convert_miles_to_minutes_nyc, list_of_output_predictions_to_direction
__author__ = ' Jonathan Hilgart'


class AlgorithmComparison(object):
    """A class used to compare DQN (mlp and lstm), Actor Critic, and a naive approach"""


    def __init__(self, args, ACTION_SPACE, OBSERVATION_SPACE,
                 list_of_unique_geohashes,list_of_time_index, list_of_geohash_index,
                             list_of_inverse_heohash_index, final_data_structure,
                             list_of_output_predictions_to_direction):
        """Store the data attributes needed for each algorithm"""

        self.ACTION_SPACE = ACTION_SPACE
        self.OBSERVATION_SPACE = OBSERVATION_SPACE
        self.args = args
        self.actor_model()
        self.critic_model()
        self.list_of_unique_geohashes = list_of_unique_geohashes
        self.list_of_time_index = list_of_time_index
        self.list_of_geohash_index = list_of_geohash_index
        self.list_of_inverse_heohash_index = list_of_inverse_heohash_index
        self.final_data_structure = final_data_structure
        self.list_of_output_predictions_to_direction = list_of_output_predictions_to_direction

        self.actor_model()
        self.build_mlp_dqn_model()

        self.first_run = True

    def actor_model(self):
        """Build an actor model with mlp.
         http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.htmlCode
         Input time followed by geohash index for predictions.
         When making predictions, you do not need the critic network.
         The critic network is solely used in training an actor critic model."""
        model_mlp = Sequential()
        model_mlp.add(Dense(100, input_shape=(self.OBSERVATION_SPACE,)))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Activation('relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(500))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Activation('relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(1000))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Activation('relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(self.ACTION_SPACE, activation='linear'))
        # predict which geohash to move to next
        model_mlp.load_weights(self.args['model_weights_load_actor_mlp'])
        adam = Adam(clipnorm=1.0)
        model_mlp.compile(loss='mse',optimizer=adam)
        self.actor_model = model_mlp

    def build_mlp_mdqn_odel(self):
        """Build a simple MLP model.
        Input  time follwoed by the  geohash index for predictions"""
        model_mlp = Sequential()
        model_mlp.add(Dense(100, input_shape=(self.OBSERVATION_SPACE,,)))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Activation('relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(500))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Activation('relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(1000))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Activation('relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(self.ACTION_SPACE, activation='linear')) ## predict which geohash to move to next
        adam = Adam(lr=LEARNING_RATE)
        model_mlp.load_weights(self.args['model_weights_load_dqn_mlp'])
        model_mlp.compile(loss='mse',optimizer=adam)

        self.model_mlp_dqn = model_mlp_dqn

    def build_lstm_dqn_model(self):
        """Build a simpleLSTM model choosen by hyperparameter selection from hyperas.
        Input  time follwoed by the  geohash index. """
        model_lstm = Sequential()
        model_lstm .add(LSTM( 512, dropout=.24,
                             batch_input_shape=(1,None, 2),
                         recurrent_dropout=.24,return_sequences = True))
        model_lstm.add(BatchNormalization())
        model_lstm .add(LSTM(1024, dropout=.18,
                 recurrent_dropout=.18,
                 return_sequences = True))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Dense(512))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Activation('sigmoid'))
        model_lstm .add(Dense(9, activation='linear',name='dense_output'))
        adam = Adam(clipnorm=.5, clipvalue=.5)
        model_lstm.load_weights(self.args['model_weights_load_dqn_lstm'])
        model_lstm .compile(loss='mean_squared_error', optimizer=adam)
        self.model_lstm_dqn = model_lstm

    def output_lat_long_predictions_given_input(self,geohash_start=None, time_start=None):
        """Give n the starting geohash , and time, see which direction each algorithm
        goes to. Return the latitude and longitude for each geohash start and time_start.
        If running for the first time, provide a geohash and time to start at.
        If running past the first time, the model will have kept the previous geohashah
        and time to create a prediction from.
        """

        if self.first_run = True:
            start_geohash_index = self.list_of_geohash_index[geohash_start]
            start_state = np.array([[time_start,geohash_start]])
            start_state_lstm = np.array([[[time_start, geohash_start]]])

            # predict for DQN MLP
            mlp_dqn_predictions = self.model_mlp_dqn.predict(start_state)
            # action to take for MLP DQN
            mlp_dqn_action = np.argmax(mlp_dqn_preidctions)

            # predict for DQN LSTM
            lstm_dqn_predictions = self.model_lstm_dqn.predict(start_state_lstm)
            # action to take for MLP DQN
            lstm_dqn_action = np.argmax(lstm_dqn_preidctions)

            # predict for actor critic
            mlp_ac_predictions = self.actor_model.predict(start_state)
            # action to take for MLP DQN
            mlp_ac_action = np.argmax(mlp_ac_preidctions)

            # predict for naive
            naive_action = np.random.choice([0,1,2,3,4,5,6,7,8])

            # Record the informationfor DQN MLP
            self.s_geohash1_dqn_mlp, self.s_time1_dqn_mlp, r_t_dqn_mlp, fare_t_dqn_mlp, \
                latitude_s1_dqn_mlp, longtitude_s1_dqn_mlp = \
                self.geohash_conversion_given_action_state(
                    mlp_dqn_action, geohash_start, time_start)
            # Record the information for DQN LSTM
            self.s_geohash1_dqn_lstm, self.s_time1_dqn_lstm, r_t_dqn_mlp, fare_t_dqn_lstm, \
                latitude_s1_dqn_lstm, longtitude_s1_dqn_lstm = \
                self.geohash_conversion_given_action_state(
                    lstm_dqn_action, geohash_start, time_start)
            # Record information for Actor-Critic MLP
            self.s_geohash1_ac_mlp, self.s_time1_ac_mlp, r_t_dqn_mlp, fare_t_ac_mlp, \
                latitude_s1_ac_mlp, longtitude_s1_ac_mlp = \
                self.geohash_conversion_given_action_state(
                    mlp_ac_action, geohash_start, time_start)
            # Record information for the Naive implementation
            self.s_geohash1_naive, self.s_time1_naive, r_t_dqn_mlp, fare_t_naive, \
                latitude_s1_naive, longtitude_s1_naive = \
                self.geohash_conversion_given_action_state(
                    naive_action, geohash_start, time_start)

            self.first_run = False

            return latitude_s1_dqn_mlp, longtitude_s1_dqn_mlp, \
                latitude_s1_dqn_lstm, longtitude_s1_dqn_lstm, \
                latitude_s1_ac_mlp, longtitude_s1_ac_mlp, \
                latitude_s1_naive, longtitude_s1_naive

        else:
            ## convert index geohash to string geohash
            geohash_dqn_mlp = self.list_of_inverse_heohash_index[self.s_geohash1_dqn_mlp]
            geohash_dqn_lstm = self.list_of_inverse_heohash_index[self.s_geohash1_dqn_lstm]
            geohash_ac_mlp = self.list_of_inverse_heohash_index[self.s_geohash1_dqn_lstm]
            geohash_naive = self.list_of_inverse_heohash_index[self.s_geohash1_naive]

            start_state_dqn_mlp = np.array([[self.s_time1_dqn_mlp, self.s_geohash1_dqn_mlp ]])
            start_state_ac_mlp = np.array([[self.s_time1_ac_mlp, self.s_geohash1_ac_mlp ]])
            start_state_lstm_dqn = np.array([[[self.s_time1_dqn_lstm, self.s_geohash1_dqn_lstm ]]])

            # predict for DQN MLP
            mlp_dqn_predictions = self.model_mlp_dqn.predict(start_state_dqn_mlp)
            # action to take for MLP DQN
            mlp_dqn_action = np.argmax(mlp_dqn_preidctions)

            # predict for DQN LSTM
            lstm_dqn_predictions = self.model_lstm_dqn.predict(start_state_lstm_dqn)
            # action to take for MLP DQN
            lstm_dqn_action = np.argmax(lstm_dqn_preidctions)

            # predict for actor critic
            mlp_ac_predictions = self.actor_model.predict(start_state_ac_mlp)
            # action to take for MLP DQN
            mlp_ac_action = np.argmax(mlp_ac_preidctions)

            # predict for naive
            naive_action = np.random.choice([0,1,2,3,4,5,6,7,8])

            # Record the informationfor DQN MLP
            self.s_geohash1_dqn_mlp, self.s_time1_dqn_mlp, r_t_dqn_mlp, fare_t_dqn_mlp, \
                latitude_s1_dqn_mlp, longtitude_s1_dqn_mlp = \
                self.geohash_conversion_given_action_state(
                    mlp_dqn_action, geohash_dqn_mlp, self.s_time1_dqn_mlp)
            # Record the information for DQN LSTM
            self.s_geohash1_dqn_lstm, self.s_time1_dqn_lstm, r_t_dqn_mlp, fare_t_dqn_lstm, \
                latitude_s1_dqn_lstm, longtitude_s1_dqn_lstm = \
                self.geohash_conversion_given_action_state(
                    lstm_dqn_action, geohash_dqn_lstm, self.s_time1_dqn_lstm,)
            # Record information for Actor-Critic MLP
            self.s_geohash1_ac_mlp, self.s_time1_ac_mlp, r_t_dqn_mlp, fare_t_ac_mlp, \
                latitude_s1_ac_mlp, longtitude_s1_ac_mlp = \
                self.geohash_conversion_given_action_state(
                    mlp_ac_action, geohash_ac_mlp, self.s_time1_ac_mlp)
            # Record information for the Naive implementation
            self.s_geohash1_naive, self.s_time1_naive, r_t_dqn_mlp, fare_t_naive, \
                latitude_s1_naive, longtitude_s1_naive = \
                self.geohash_conversion_given_action_state(
                    naive_action, geohash_naive , self.s_time1_naive)

            return latitude_s1_dqn_mlp, longtitude_s1_dqn_mlp, \
                latitude_s1_dqn_lstm, longtitude_s1_dqn_lstm, \
                latitude_s1_ac_mlp, longtitude_s1_ac_mlp, \
                latitude_s1_naive, longtitude_s1_naive


            self.first_run = False



    def geohash_conversion_given_action_state(self,action, start_geohash, start_time):
        """Go through the process of converting an actions from a state into a
        geohash and corresponding latitude and longtitude.
        Returns geohash, time, reward ratio (fare / time), fare, lat, and longtitude"""

        #Get the neighbors from the current geohash - convert back to string
        current_geohash_string = self.list_of_inverse_heohash_index[start_geohash]
        neighbors = gh.neighbors(current_geohash_string)
        # Get the direction we should go
        direction_to_move_to = list_of_output_predictions_to_direction[action]
        # Get the geohash of the direction we moved to
        if direction_to_move_to =='stay':
            new_geohash = starting_geohash  # stay in current geohash, get the index of said geohash
            possible_rewards = np.array(self.final_data_structure[start_time][new_geohash])
            # hash with the letters  of the geohash above
            new_geohash = self.list_of_geohash_index[starting_geohash]
        else:
            new_geohash = neighbors[direction_to_move_to]## give us the geohash to move to next

        # get the reward of the geohash we just moved to (this is the ratio of fare /time of trip)
        # time, geohash, list of tuple ( fare, time ,ratio)
        possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])

        if len (possible_rewards) ==0:
            r_t = -.1  # we do not have information for this time and geohash, don't go here. waste gass
            fare_t = 0  # no information so the fare = 0
            s_time1 = start_time+10  # assume this took ten minutes
        else:
            reward_option = np.random.randint(0,len(possible_rewards))
            r_t =  possible_rewards[reward_option][2]  # get the ratio of fare / trip time
            fare_t = possible_rewards[reward_option][0]
            # get the trip length
            s_time1 = start_time + possible_rewards[reward_option][1]
        s_geohash1 = self.list_of_geohash_index[new_geohash]
        # decode the geohash into latitude nad longtitude
        decoded_geohash_s1 = gh.decode(s_geohash1)
        latitude_s1 = decoded_geohash_s1['lat']
        longtitude_s1 = decoded_geohash_s1['lon']
        # return the latitude and longtitude, fare, geohash, and time
        return s_geohash1, s_time1, r_t, fare_t, latitude_s1, longtitude_s1






args = {'model_weights_load_actor_mlp':,
        'model_weights_load_dqn_mlp':'mlp_model_dqn/model_mlp_linear_2million.h5',
        'model_weights_load_dqn_lstm':}
