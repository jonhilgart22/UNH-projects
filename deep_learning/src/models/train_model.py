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
import time
import sys
import auxiliary_functions, make_dataset
from py_geohash_any import geohash as gh
from keras import backend as K
from auxiliary_functions import convert_miles_to_minutes_nyc, list_of_output_predictions_to_direction
__author__ = 'Jonathan Hilgart'


 #parameters
ACTIONS = 9 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 5000. # timesteps to observe before training
EXPLORE = 30000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 16 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4


class RLNYCTaxiCab(object):

    def __init__(self, list_of_unique_geohashes,list_of_time_index, list_of_geohash_index,
                list_of_inverse_heohash_index, final_data_structure, return_metrics=False):
        self.list_of_unique_geohashes = list_of_unique_geohashes
        self.list_of_time_index = list_of_time_index
        self. list_of_geohash_index =  list_of_geohash_index
        self.list_of_inverse_heohash_index = list_of_inverse_heohash_index
        self.final_data_structure = final_data_structure
        self.build_mlp_model()
        self.return_metrics = return_metrics


    def build_mlp_model(self):
        """Build a simple MLP model.
        Input  time follwoed by the  geohash index. """
        model_mlp = Sequential()
        model_mlp.add(Dense(100, activation='relu', input_shape= (2,)))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(500, activation='relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(500, activation='relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(100, activation='relu'))
        model_mlp.add(Dropout(.3))
        model_mlp.add(Dense(9, activation='softmax')) ## predict which geohash to move to next
        adam = Adam(lr=LEARNING_RATE)
        model_mlp.compile(loss='categorical_crossentropy',optimizer=adam)

        self.model_mlp = model_mlp

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
                #print(new_geohash,'new geohash stay')
            else:
                new_geohash = neighbors[direction_to_move_to]## give us the geohash to move to next
                #print(new_geohash,'new geohash moved to new mplace')
            #print(new_geohash,'new geohash we moved to')

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
            #print(r_t,'reward')
            #print(s_time,'current time')
            s_geohash1 = self.list_of_geohash_index[new_geohash]
            #print(s_time1,s_geohash1 , 'time1, geohash1')

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





    def trainNetworkNeuralNetworkTaxicab(self, args, training_length=1000,
                                         return_training_data = False):
        # Code adapted from https://github.com/yanpanlau/Keras-FlappyBird/blob/master/qlearn.py
        """Train a DQN algorithm to learn how the best geohashes to go to throughout the day. Each geohash is about
        3803 x 3803 meters (~15 minutes of driving time to traverse in NYC).
        This algoirthm incorporates experience replay to stablize the training procedure. """

        self.return_training_data = return_training_data
        # store the previous observations in replay memory
        D = deque()

        # get the first state by randomlly choosing a geohash to start at and random time to start at
        # Assume that the first state has no reward associated with it
        # Over multiple steps, starting geohash becomes the previous geohash we visited
        starting_geohash = np.random.choice(self.list_of_unique_geohashes)
        s_time = np.random.choice(self.list_of_time_index)
        s_geohash = self.list_of_geohash_index[starting_geohash]

        s_t = np.array([[s_time,
                         s_geohash]])
        #print(s_t,'starting time and geohash index')

        if args['mode'] == 'Run':
            OBSERVE = 20  #We keep observe, never train
            epsilon = FINAL_EPSILON
            print ("Now we load weight")
            self.model_mlp.load_weights("model_mlp_million.h5")
            adam = Adam(lr=LEARNING_RATE)
            self.model_mlp.compile(loss='mse',optimizer=adam)
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
            #print(t, ' t')
            #choose a random action action epsilon greedy
            if t % FRAME_PER_ACTION == 0: ## will always choose this if frame per action is 1
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(ACTIONS) # Randomlly choose another geohash to go to
                    a_t[action_index] = 1
                else:
                    q = self.model_mlp.predict(s_t)       #input the time followed by the geohash index
                    max_Q = np.argmax(q)  # find the position of the highest probability (which direction to go in)
                    action_index = max_Q
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
    #             print('trip length',possible_rewards[np.random.randint(0,len(possible_rewards))][1] )
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
            # Terminal should be a one if the day is over or a zero otherwise
            # time, geohash, action index, reward, time1, geohash 1, terminal

            D.append((s_time,s_geohash, action_index, r_t, s_time1, s_geohash1, terminal, fare_t))

            if return_training_data  == True: # append training data
                if r_t >0: ## normalize the values for hyperas
                    training_r = 1
                    self.training_data_X[t,:] = np.array([s_time, s_geohash])
                    self.training_data_y[t,action_index] = np.array([training_r])
                else:
                    training_r = 0
                    self.training_data_X[t,:] = np.array([s_time, s_geohash])
                    # action index for the reward
                    self.training_data_y[t,action_index] = training_r


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
                    fare_t = minibatch[i][7]
                    # if terminated, only equals reward
                    for col in range(inputs.shape[1]-1):
                        inputs[i,col] = s_time_t   #Save the time and geohash in the inputs to the model
                        inputs[i,col+1] = s_geohash_t

                    state_t = np.array([[s_time_t, s_geohash_t]])
                    state_t1 = np.array([[s_time_t1,s_geohash_t1]])

                    targets[i] = self.model_mlp.predict(state_t)  # update entire row
                    Q_sa = self.model_mlp.predict(state_t1)
                    #print(Q_sa, ' Q function for a given state')
                    if terminal==1: ## The day ended, pick a new starting geohash and time
                        targets[i, action_t] = reward_t

                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa) ## exponential discounting for each memory

                # targets2 = normalize(targets)
                loss += self.model_mlp.train_on_batch(inputs, targets)
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

            if t % 1000 == 0:
                print("Now we save model")
                self.model_mlp.save_weights("model_mlp.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(self.model_mlp.to_json(), outfile)

                print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss, "/ Total fare ", total_fare_received)
                now_time = time.time()
                print('1000 steps took {}'.format(now_time - start_time))
                start_time = now_time

            if t ==training_length:
                if self.return_metrics == True:
                    return loss_list, total_fare_received_over_time, \
                        list_of_geohashes_visited,  total_naive_fare_over_time,\
                        total_days_driven
                if self.return_training_data ==True:
                    return self.training_data_X, self.training_data_y

                else:
                    print("Now we save model")
                    self.model_mlp.save_weights("model_mlp.h5", overwrite=True)
                    with open("model.json", "w") as outfile:
                        json.dump(self.model_mlp.to_json(), outfile)
                    break

            # increment the state and time information
            s_time = s_time1
            s_geohash = s_geohash1
            if self.return_metrics == True:
                list_of_geohashes_visited.append(starting_geohash)


            starting_geohash = new_geohash## update the starting geohash in case we stay here
            t = t + 1




if __name__ =="__main__":
    import gc; gc.collect()

    with K.get_session(): ## TF session

        #yopen up the data
        taxi_yellowcab_df, final_data_structure= make_dataset.main()
        ## the the data structures needed for the RL calss
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
        #
        arg = {'mode':'Test'}
        train_rl_taxi = RLNYCTaxiCab(list_of_unique_geohashes,list_of_time_index,list_of_geohash_index,
                list_of_inverse_heohash_index, final_data_structure, return_metrics=False)

        train_rl_taxi.trainNetworkNeuralNetworkTaxicab(arg, training_length=1000000,
                                                       return_training_data =False)
