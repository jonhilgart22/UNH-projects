#! usr/bin/env python
import numpy as np
import datetime
import random
import numpy as np
from collections import deque
import json
from collections import defaultdict
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
import time
import sys
import pickle
import auxiliary_functions, make_dataset
from py_geohash_any import geohash as gh
from keras import backend as K
from auxiliary_functions import convert_miles_to_minutes_nyc, \
    list_of_output_predictions_to_direction
__author__ = 'Jonathan Hilgart'


 #parameters
ACTIONS = 9 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 10000. # timesteps to observe before training
EXPLORE = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
TRAINING_EPSILON = .1
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-1


class RLNYCTaxiCabLargeNetwork_LSTM(object):
    """Creates an lstm model with DQN to train on NYC taxi data from January 2016."""

    def __init__(self, list_of_unique_geohashes,list_of_time_index, list_of_geohash_index,
                list_of_inverse_heohash_index, final_data_structure, return_metrics=False):
        self.list_of_unique_geohashes = list_of_unique_geohashes
        self.list_of_time_index = list_of_time_index
        self. list_of_geohash_index =  list_of_geohash_index
        self.list_of_inverse_heohash_index = list_of_inverse_heohash_index
        self.final_data_structure = final_data_structure
        self.build_lstm_model()
        self.return_metrics = return_metrics


    def build_lstm_model(self):
        """Build a simple MLP model.
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
        model_lstm .compile(loss='mean_squared_error', optimizer=adam,
                            metrics=['accuracy'])
        self.model_lstm = model_lstm

    def NaiveApproach(self, s_time_, s_geohash_,starting_geo, input_fare_list = None, historic_current_fare = None):
        """Assign the same probability to every state and keep track of the total fare received, total fare over time,
        and geohashes visited"""

        ## parameters to track where we are and at what time
        starting_geohash = starting_geo
        s_time = s_time_
        s_geohash = s_geohash_
        list_of_geohashes_visited = []

        ## check and see if we have old fare to continue adding to
        if input_fare_list == None:
            total_fare = 0
            total_fare_over_time = []

        else:
            total_fare = historic_current_fare
            total_fare_over_time = input_fare_list


        while True:
            a_t = np.zeros([ACTIONS])
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
            #Get the neighbors from the current geohash - convert back to string
            current_geohash_string = self.list_of_inverse_heohash_index[s_geohash]
            neighbors = gh.neighbors(current_geohash_string)
            # Get the direction we should go
            direction_to_move_to = list_of_output_predictions_to_direction[action_index]
            # Get the geohash of the direction we moved to
            if direction_to_move_to =='stay':
                new_geohash = starting_geohash # stay in current geohash, get the index of said geohash
                possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])
                # hash with the letters  of the geohash above
                new_geohash = self.list_of_geohash_index[starting_geohash]
            else:
                new_geohash = neighbors[direction_to_move_to]## give us the geohash to move to next

            # get the reward of the geohash we just moved to (this is the ratio of fare /time of trip)
            # time, geohash, list of tuple ( fare, time ,ratio)
            possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])

            if len (possible_rewards) ==0:
                r_t = -.1 ## we do not have information for this time and geohash, don't go here. waste gass
                fare_t = 0 ## no information so the fare = 0
                s_time1 = s_time+10 ## assume this took ten minutes
            else:
                r_t =  possible_rewards[np.random.randint(0,len(possible_rewards))][2] # get the ratio of fare / trip time
                fare_t = possible_rewards[np.random.randint(0,len(possible_rewards))][0]
                # get the trip length
                s_time1 = s_time + possible_rewards[np.random.randint(0,len(possible_rewards))][1]
                #r_t = np.random.choice(possible_rewards)
            s_geohash1 = self.list_of_geohash_index[new_geohash]
            # store the transition in D
            if s_time1 <= 2350: # The last possible time for a trip
                terminal = 0
                 ### get the naive implementation per day
            else: # the day is over, pick a new starting geohash and time
                break # the day is over

#                 terminal = 1

#                 s_time1 = np.random.choice(self.list_of_time_index)
#                 s_geohash1 =   self.list_of_geohash_index[np.random.choice(self.list_of_unique_geohashes)]

            total_fare += fare_t
            total_fare_over_time.append(total_fare)
            list_of_geohashes_visited.append(starting_geohash)
            # increment the state and time information
            s_time = s_time1
            s_geohash = s_geohash1
            starting_geohash = new_geohash## update the starting geohash in case we stay here
        return total_fare, total_fare_over_time, list_of_geohashes_visited





    def trainNetworkNeuralNetworkTaxicab_LSTM(self, args, training_length=1000,
                                         return_training_data = False, save_model = False):
        # Code adapted from https://github.com/yanpanlau/Keras-FlappyBird/blob/master/qlearn.py
        """Train a DQN algorithm to learn how the best geohashes to go to throughout the day.
         Each geohash is about
        3803 x 3803 meters (~15 minutes of driving time to traverse in NYC).
        This algoirthm incorporates experience replay to stablize the training procedure
        for the DWN algorithm. Due to the large size of the input features,
        you need to train for a long time (1-2million iterations) """

        self.return_training_data = return_training_data
        # store the previous observations in replay memory
        D = deque()

        # get the first state by randomlly choosing a geohash to start at and random time to start at
        # Assume that the first state has no reward associated with it
        # Over multiple steps, starting geohash becomes the previous geohash we visited
        starting_geohash = np.random.choice(self.list_of_unique_geohashes)
        s_time = np.random.choice(self.list_of_time_index)
        s_geohash = self.list_of_geohash_index[starting_geohash]

        s_t = np.array([[[s_time,
                         s_geohash]]])

        if args['mode'] == 'Run':
            OBSERVE = 10000  #We keep observe, never train
            epsilon = TRAINING_EPSILON
            print ("Now we load weight")
            self.model_lstm.load_weights(args['model_weights_load'])
            adam = Adam(lr=LEARNING_RATE)
            self.model_lstm.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")
        else:                       #We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        #start your observations
        t = 0
        total_days_driven = 0
        loss_list = []
        total_fare_received = 0
        total_fare_received_over_time = []
        list_of_geohashes_visited = []
        total_naive_fare = 0
        total_naive_fare_over_time =[]
        list_of_naive_geohashes_visited = []
        if return_training_data == True:
            self.training_data_X = np.zeros((training_length+1,2))
            self.training_data_y = np.zeros((training_length+1,ACTIONS))

        if self.return_metrics == True: ##  Compare to a naive approach, only train / observe
            if t > OBSERVE:
                total_naive_fare, total_naive_fare_over_time, list_of_naive_geohashes_visited = \
                    self.NaiveApproach(s_time, s_geohash,starting_geohash)

        start_time = time.time()
        while (True):
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            a_t = np.zeros([ACTIONS])
            #choose a random action action epsilon greedy
            if t % FRAME_PER_ACTION == 0: ## will always choose this if frame per action is 1
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS) # Randomlly choose another geohash to go to
                    a_t[action_index] = 1
                else:
                    #print("------------Predicted Action___________")

                    q = self.model_lstm.predict(s_t)       #input the time followed by the geohash index
                    max_Q = np.argmax(q)  # find the position of the highest probability (which direction to go in)
                    action_index = max_Q
                    #print('Action {}'.format(action_index))
                    a_t[max_Q] = 1

            #We reduced the epsilon gradually to take more random actions
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            #run the selected action and observed next state and reward
            # We need to find the neighbors to the geohash that we started at

            #Get the neighbors from the current geohash - convert back to string
            current_geohash_string = self.list_of_inverse_heohash_index[s_geohash]
            neighbors = gh.neighbors(current_geohash_string)
            # Get the direction we should go
            direction_to_move_to = list_of_output_predictions_to_direction[action_index]
            # Get the geohash of the direction we moved to
            if direction_to_move_to =='stay':
                new_geohash = starting_geohash # stay in current geohash, get the index of said geohash
                possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])
                # hash with the letters  of the geohash above
                new_geohash = self.list_of_geohash_index[starting_geohash]
            else:
                new_geohash = neighbors[direction_to_move_to]## give us the geohash to move to next

            # get the reward of the geohash we just moved to (this is the ratio of fare /time of trip)
            # time, geohash, list of tuple ( fare, time ,ratio)
            possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])

            if len (possible_rewards) ==0:
                r_t = -.1 ## we do not have information for this time and geohash, don't go here. waste gass
                fare_t = 0 ## no information so the fare = 0
                s_time1 = s_time+10 ## assume this took ten minutes
            else:
                r_t =  possible_rewards[np.random.randint(0,len(possible_rewards))][2] # get the ratio of fare / trip time
                fare_t = possible_rewards[np.random.randint(0,len(possible_rewards))][0]
                # get the trip length
                s_time1 = s_time + possible_rewards[np.random.randint(0,len(possible_rewards))][1]
                #r_t = np.random.choice(possible_rewards)
            s_geohash1 = self.list_of_geohash_index[new_geohash]

            # store the transition in D
            if s_time1 <= 2350: # The last possible time for a trip
                terminal = 0
            else: # the day is over, pick a new starting geohash and time
                print('We finished a day!')
                terminal = 1
                total_days_driven +=1
                # Choose a new starting time and geohash
                s_time1 = np.random.choice(self.list_of_time_index)
                s_geohash1 =   self.list_of_geohash_index[np.random.choice(
                    self.list_of_unique_geohashes)]
                # Chech the naive approach to the new geohashes and time
                if self.return_metrics == False: ## don't benchmark to the naive approach
                    pass
                else:
                    if t > OBSERVE: # only record after observations
                        total_naive_fare, total_naive_fare_over_time, naive_geohashes_visited = \
                        self.NaiveApproach(s_time1, s_geohash1,
                            starting_geohash, total_naive_fare_over_time,total_naive_fare )
                        if return_training_data == True:
                            list_of_naive_geohashes_visited.extend(naive_geohashes_visited)
            # Terminal should be a one if the day is over or a zero otherwise
            # time, geohash, action index, reward, time1, geohash 1, terminal

            D.append((s_time,s_geohash, action_index, r_t, s_time1, s_geohash1, terminal))

            if return_training_data  == True: # append training data
                if r_t >0: ## normalize the values for hyperas
                    self.training_data_X[t,:] = np.array([s_time, s_geohash])
                    self.training_data_y[t,action_index] = np.array([r_t])
                else:
                    self.training_data_X[t,:] = np.array([s_time, s_geohash])
                    # action index for the reward
                    self.training_data_y[t,action_index] = r_t


            if len(D) > REPLAY_MEMORY: ## don't store a huge replay memory
                D.popleft()

            ######### NEXT SEXTION #########
            #only train if done observing
            if t > OBSERVE:
                #sample a minibatch to train on
                minibatch = random.sample(D, BATCH)
                inputs = []

                inputs = np.zeros((BATCH, s_t.shape[1]))   #16, 2
                targets = np.zeros((inputs.shape[0], ACTIONS))       #16, 9
                #Now we do the experience replay
                for i in range(0, len(minibatch)): # 0 -15 for batch 16
                    s_time_t = minibatch[i][0]
                    s_geohash_t = minibatch[i][1]
                    action_t = minibatch[i][2] # action index
                    reward_t = minibatch[i][3]
                    s_time_t1 = minibatch[i][4]
                    s_geohash_t1 = minibatch[i][5]
                    terminal = minibatch[i][6]
                    # if terminated, only equals reward
                    for col in range(inputs.shape[1]-1):
                        inputs[i,col] = s_time_t   #Save the time and geohash in the inputs to the model
                        inputs[i,col+1] = s_geohash_t

                    state_t = np.array([[s_time_t, s_geohash_t]]).reshape(1,1,2)
                    state_t1 = np.array([[s_time_t1,s_geohash_t1]]).reshape(1,1,2)

                    targets[i] = self.model_lstm.predict(state_t)  # update entire row
                    Q_sa = self.model_lstm.predict(state_t1)
                    #print(Q_sa, ' Q function for a given state')
                    if terminal==1: ## The day ended, pick a new starting geohash and time
                        targets[i, action_t] = reward_t

                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa) ## exponential discounting for each memory
                print(inputs,'inputs')
                print(inputs.shape,'inputs shape')
                print()
                print(targets,'targets')
                print(targets.shape,'tarets shape')
                print()
                inputs = inputs.reshape(1,BATCH,2)
                targets = targets.reshape(1,BATCH,9)
                loss += self.model_lstm.train_on_batch(inputs, targets)
                loss_list.append(loss)
                if self.return_metrics == True:
                    # only record fares once we start training
                    total_fare_received += fare_t
                    total_fare_received_over_time.append(total_fare_received)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            if save_model == True:
                if t % 1000 == 0:
                    print("Now we save model")
                    self.model_lstm.save_weights(args['save_model_weights'], overwrite=True)
                    with open("model_lstm.json", "w") as outfile:
                        json.dump(self.model_lstm.to_json(), outfile)

            if t % 500 == 0:
                print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss, "/ Total fare RL ", total_fare_received,
                "/ Total fare naive", total_naive_fare)
                now_time = time.time()
                print('500 steps took {}'.format(now_time - start_time))
                start_time = now_time

            if t ==training_length: ### end training
                if self.return_metrics == True and save_model == True:
                    print("Now we save model")
                    self.model_lstm.save_weights(args['save_model_weights'], overwrite=True)
                    with open("model_lstm.json", "w") as outfile:
                        json.dump(self.model_lstm.to_json(), outfile)
                    return loss_list, total_fare_received_over_time, \
                        list_of_geohashes_visited,  total_naive_fare_over_time,\
                        total_days_driven, list_of_naive_geohashes_visited
                elif self.return_metrics == True:
                    return loss_list, total_fare_received_over_time, \
                        list_of_geohashes_visited,  total_naive_fare_over_time,\
                        total_days_driven, list_of_naive_geohashes_visited
                elif self.return_training_data ==True:
                    return self.training_data_X, self.training_data_y
                elif save_model == True:
                    print("Now we save model")
                    self.model_lstm.save_weights(args['save_model_weights'], overwrite=True)
                    with open("model_lstm.json", "w") as outfile:
                        json.dump(self.model_lstm.to_json(), outfile)
                    break
                else:# something weird happened
                    break

            # increment the state and time information
            s_time = s_time1
            s_geohash = s_geohash1
            if self.return_metrics == True:
                list_of_geohashes_visited.append(starting_geohash)
            starting_geohash = new_geohash## update the starting geohash in case we stay here
            t = t + 1

def data_attributes(taxi_yellowcab_df):
    """Some random data objects needed to train the RL algorithm"""
    list_of_output_predictions_to_direction =\
        {0:'nw',1:'n',2:'ne',3:'w',4:'stay',5:'e',6:'sw',7:'s',8:'se'}
    list_of_unique_geohashes = taxi_yellowcab_df.geohash_pickup.unique()
    list_of_geohash_index  = defaultdict(int)
    for idx,hash_n in enumerate(list_of_unique_geohashes):
        list_of_geohash_index [hash_n] = idx
    list_of_inverse_heohash_index = defaultdict(str)
    for idx,hash_n in enumerate(list_of_unique_geohashes):
        list_of_inverse_heohash_index[idx] = hash_n
    hours = [str(_) for _ in range(24)]
    minutes = [str(_) for _ in range(0,60,10)]
    minutes.append('00')
    list_of_time_index =[]
    for h in hours:
        for m in minutes:
            list_of_time_index.append(int(str(h)+str(m)))

    list_of_time_index = list(set(list_of_time_index))
    return list_of_output_predictions_to_direction, list_of_unique_geohashes, \
        list_of_geohash_index, list_of_time_index , list_of_inverse_heohash_index


if __name__ =="__main__":
    import gc; gc.collect()

    with K.get_session(): ## TF session
        #yopen up the data
        taxi_yellowcab_df, final_data_structure= make_dataset.main()
        ## the the data structures needed for the RL calss
        list_of_output_predictions_to_direction, list_of_unique_geohashes, \
            list_of_geohash_index, list_of_time_index, list_of_inverse_heohash_index\
             = data_attributes(taxi_yellowcab_df)
        #
        arg = {'mode':'Test','save_model':True,'model_weights_load':'model_lstm_1mil.h5',
               'save_model_weights':'lstm_weight_1mils.h5'}
        train_rl_taxi = RLNYCTaxiCabLargeNetwork_LSTM(list_of_unique_geohashes,list_of_time_index, \
                                                      list_of_geohash_index,\
                list_of_inverse_heohash_index, final_data_structure, return_metrics=True)

        if arg['save_model']==True:
            loss_list, total_fare_received_over_time, list_of_geohashes_visited,\
            naive_fare_over_time, days_driven, naive_geohashes \
                =train_rl_taxi.trainNetworkNeuralNetworkTaxicab_LSTM(arg, training_length=1000000,
                            return_training_data = False, save_model= True)

            # with open('training_x','wb') as fp:
            #     pickle.dump(training_x, fp)
            # with open('training_y','wb') as fp:
            #     pickle.dump(training_y, fp)
            # # save your metrics
            with open('loss_over_time_lstm_1mil', 'wb') as fp:
                pickle.dump(loss_list, fp)
            with open('rl_total_fare_time_lstm_1mil','wb') as fp:
                pickle.dump(total_fare_received_over_time, fp)
            with open('naive_fare_time_lstm_1mil','wb') as fp:
                pickle.dump(naive_fare_over_time, fp)
            with open('total_day_lstm_1mil','wb') as fp:
                pickle.dump(days_driven, fp)
        else:
            train_rl_taxi.trainNetworkNeuralNetworkTaxicab(arg, training_length=1000000,
                                return_training_data =False, save_model= False)
