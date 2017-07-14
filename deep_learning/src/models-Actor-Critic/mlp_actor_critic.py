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
#sys.path.insert(0, '../data/') ## for running on local
import auxiliary_functions, make_dataset
from auxiliary_functions import convert_miles_to_minutes_nyc, list_of_output_predictions_to_direction
__author__ = ' Jonathan Hilgart'



class ActorCriticNYCMLP(object):

    """Train an actor critic model to maximize revenue for a NYC taxi driver.\
    Code inspired from http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.htmlCode.
    Each iteration in the code corresponds to a move between geohashes.
    Each epoch is a sample of batchsize 'memories' from recent transitions"""

    def __init__(self, args, ACTION_SPACE, OBSERVATION_SPACE,
                 list_of_unique_geohashes,list_of_time_index, list_of_geohash_index,
                             list_of_inverse_heohash_index, final_data_structure,
                             list_of_output_predictions_to_direction):
        """Store the data attributes needed to train the Actor Critic model"""

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


    def actor_model(self):
        """Build an actor model with mlp.
         http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.htmlCode """
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
        adam = Adam(clipnorm=1.0)
        model_mlp.compile(loss='mse',optimizer=adam)
        self.actor_model = model_mlp


    def critic_model(self):
        """Build a critic model.
        Code inspired from http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html"""
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
        model_mlp.add(Dense(1, activation='linear'))  # predict the value
        adam = Adam(clipnorm=1.0)
        model_mlp.compile(loss='mse', optimizer=adam)
        self.critic_model = model_mlp




    def NaiveApproach(self, s_time_, s_geohash_,starting_geo, input_fare_list = None,
                      historic_current_fare = None):
        """Assign the same probability to every state and
        keep track of the total fare received, total fare over time,
        and geohashes visited.
        Inputs are the starting geohash and time determined by the main algorithm.
        This is to ensure both the naive, and algorithmic, approach start at
        the same place.
        Terminates after a day is finished"""

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
            a_t = np.zeros([self.ACTION_SPACE])
            action_index = random.randrange(self.ACTION_SPACE)
            a_t[action_index] = 1
            #Get the neighbors from the current geohash - convert back to string
            current_geohash_string = self.list_of_inverse_heohash_index[s_geohash]
            neighbors = gh.neighbors(current_geohash_string)
            # Get the direction we should go
            direction_to_move_to = list_of_output_predictions_to_direction[action_index]
            # Get the geohash of the direction we moved to
            if direction_to_move_to =='stay':
                new_geohash = starting_geohash  # stay in current geohash, get the index of said geohash
                possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])
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
                s_time1 = s_time+10  # assume this took ten minutes
            else:
                reward_option = np.random.randint(0,len(possible_rewards))
                r_t =  possible_rewards[reward_option][2]  # get the ratio of fare / trip time
                fare_t = possible_rewards[reward_option][0]
                # get the trip length
                s_time1 = s_time + possible_rewards[reward_option][1]
            s_geohash1 = self.list_of_geohash_index[new_geohash]
            # store the transition in D
            if s_time1 <= 2350: # The last possible time for a trip
                terminal = 0
                 #get the naive implementation per day
            else:  # the day is over, pick a new starting geohash and time
                break  # the day is over


            total_fare += fare_t
            total_fare_over_time.append(total_fare)
            list_of_geohashes_visited.append(starting_geohash)
            # increment the state and time information
            s_time = s_time1
            s_geohash = s_geohash1
            starting_geohash = new_geohash## update the starting geohash in case we stay here
        return total_fare, total_fare_over_time, list_of_geohashes_visited


    def trainer(self, n_days=10, batchSize=16,
                gamma=0.975, epsilon=1, min_epsilon=0.1,
                buffer_size=50000):
        """Train a Actor Critic model for a given number of days.
        Code inspired from http://www.rage.net/~greg/2016-07-05-ActorCritic-with-OpenAI-Gym.html
        Returns actor_loss,critic_loss, total_fare_over_time, daily_fare.

        Most important parts of the Actor Critic model are
            1) Critic predicts a value of state A
            2) Actor makes a move from state A to state B
            3) Critic predicts a value of state B
            4) The value for state B is given by the temporal difference between
            A and B from the critic.
            5) This temporal differen is used by the actor for training(i.e. if it
            is a large value, the network will encourage the actor to take this move
            more often)

        Note: each geohash takes above 15minutes to drive through using
        average_fare_per_days peed of 8.6 mph
        """

        # Replay buffers
        actor_replay = []
        critic_replay = []
        # Track loss over time
        actor_loss = []
        critic_loss = []
        total_fare = 0
        total_fare_over_time = []
        average_fare_per_day = []
        percent_profitable_moves_over_time = []
        ACTIONS = 9
        total_days_driven = 0
        day_start = time.time()
        # naive implementation results
        total_naive_fare = 0
        total_naive_fare_over_time = []
        list_of_geohashes_visited_actor_critic = []
        naive_geohashes_visited = []
        # load existing weights
        if self.args['mode']=='Test':
            try:
                print ("Now we load weight")
                buffer_size = self.args['test_buffer_size']
                if self.args['reduce_epsilon_test']==True:
                    epsilon = .00001 # So that we don't take random actions
                self.actor_model.load_weights(self.args['model_weights_load_actor'])
                adam = Adam()
                self.actor_model.compile(loss='mse',optimizer=adam)
                self.critic_model.load_weights(self.args['model_weights_load_critic'])
                adam = Adam()
                self.critic_model.compile(loss='mse', optimizer= adam)
                print ("Weight loaded successfully")
            except Exception as e:
                print('We could not find weights')
                print(e)
        else: ## we will train the model weights
            pass

        for i in range(n_days):
            wins = 0
            losses = 0
            daily_fare = 0

            done = False
            reward = 0
            info = None
            move_counter = 0
            starting_geohash = np.random.choice(self.list_of_unique_geohashes)
            s_time = np.random.choice(self.list_of_time_index)
            s_geohash = self.list_of_geohash_index[starting_geohash]
            a_t = np.zeros([ACTIONS])
            # start the naive appraoch at the same point as
            # the actor critic model each day
            total_naive_fare, total_naive_fare_over_time,\
             new_naive_geohashes_visited = \
            self.NaiveApproach(s_time, s_geohash,
                starting_geohash, total_naive_fare_over_time, total_naive_fare)
            naive_geohashes_visited.extend(new_naive_geohashes_visited)

            orig_state = np.array([[s_time, s_geohash]])
            orig_reward = 0
            move_counter = 0

            while(not done):
                # keep track of the geohases visited by the actor critic
                list_of_geohashes_visited_actor_critic.append(starting_geohash)
                # Get original state, original reward, and critic's value for this state.

                orig_reward = reward
                orig_val = self.critic_model.predict(orig_state)

                if (random.random() < epsilon): #choose random action
                    print('----------We took a random action ----------')
                    action = np.random.randint(0,ACTIONS)
                else: #choose best action from Q(s,a) values
                    qval = self.actor_model.predict(orig_state)
                    action = np.argmax(qval)

                # take action and observe the reward

                # Get the neighbors from the current geohash - convert back to string
                current_geohash_string = self.list_of_inverse_heohash_index[s_geohash]
                neighbors = gh.neighbors(current_geohash_string)
                # Get the direction we should go
                direction_to_move_to = self.list_of_output_predictions_to_direction[action]
                # Get the geohash of the direction we moved to
                if direction_to_move_to =='stay':
                    new_geohash = starting_geohash
                    # stay in current geohash, get the index of said geohash
                    possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])
                    # hash with the letters  of the geohash above
                    new_geohash = self.list_of_geohash_index[starting_geohash]
                else:
                    new_geohash = neighbors[direction_to_move_to]## give us the geohash to move to next

                # get the reward of the geohash we just moved to (this is the ratio of fare /time of trip)
                # time, geohash, list of tuple ( fare, time ,ratio)
                possible_rewards = np.array(self.final_data_structure[s_time][new_geohash])

                if len (possible_rewards) == 0:
                    r_t = -.1
                     # we do not have information for this time and geohash, don't go here. waste gass
                    fare_t = 0  # no information so the fare = 0
                    s_time1 = s_time+10  # assume this took ten minutes
                else:
                    reward_option = np.random.randint(0, len(possible_rewards))
                    r_t = possible_rewards[reward_option][2]
                    # get the ratio of fare / trip time
                    fare_t = possible_rewards[reward_option][0]
                    # get the trip length
                    s_time1 = s_time + possible_rewards[reward_option][1]
                s_geohash1 = self.list_of_geohash_index[new_geohash]

                # get the new state that we moved to
                new_state = np.array([[s_time1, s_geohash1]])
                # Append the fare of the new state
                total_fare += fare_t
                daily_fare += fare_t
                total_fare_over_time.append(total_fare)

                # Critic's value for this new state.
                new_val = self.critic_model.predict(new_state)

                # See if we finished a day
                if s_time1 <= 2350:  # The last possible time for a tripn
                    terminal = 0
                else:  # the day is over, pick a new starting geohash and time
                    print('ONE DAY OVER!')
                    done = True
                    total_days_driven += 1


                if not done: # Non-terminal state.
                    target = orig_reward + (gamma * new_val)
                else:
                    # In terminal states, the environment tells us
                    # the value directly.
                    target = orig_reward + (gamma * r_t)

                # For our critic, we select the best/highest value.. The
                # value for this state is based on if the agent selected
                # the best possible moves from this state forward.
                #
                # BTW, we discount an original value provided by the
                # value network, to handle cases where its spitting
                # out unreasonably high values.. naturally decaying
                # these values to something reasonable.
                # If the reward is less than zero, need to make surcharge
                # this is captured for the experiene replay of the critic
                if r_t <0: # if the reward is negative, learn from the environment to
                    best_val = r_t
                    # prevent the critic from assigning high values in the future
                else:
                    best_val = max((orig_val*gamma), target)
                # Now append this to our critic replay buffer.
                critic_replay.append([orig_state, best_val])


                # Build the update for the Actor. The actor is updated
                # by using the difference of the value the critic
                # placed on the old state vs. the value the critic
                # places on the new state.. encouraging the actor
                # to move into more valuable states.
                actor_delta = new_val - orig_val
                actor_replay.append([orig_state, action, actor_delta])

                # Critic Replays...
                while(len(critic_replay) > buffer_size): # Trim replay buffer
                    critic_replay.pop(0)
                # Start training when we have enough samples.
                if(len(critic_replay) >= buffer_size):
                    minibatch = random.sample(critic_replay, batchSize)
                    X_train = []
                    y_train = []
                    for memory in minibatch:
                        m_state, m_value = memory
                        y = np.empty([1])
                        y[0] = m_value
                        X_train.append(m_state)
                        y_train.append(y.reshape((1,)))
                    X_train = np.vstack(X_train)
                    y_train = np.vstack(y_train)
                    loss = self.critic_model.train_on_batch(X_train, y_train)
                    critic_loss.append(loss)

                # Actor Replays...
                while(len(actor_replay) > buffer_size):
                    actor_replay.pop(0)
                if(len(actor_replay) >= buffer_size):
                    X_train = []
                    y_train = []
                    minibatch = random.sample(actor_replay, batchSize)
                    for memory in minibatch:
                        m_orig_state, m_action, m_value = memory
                        old_qval = self.actor_model.predict(m_orig_state)
                        y = np.zeros(( 1, ACTIONS))
                        y[:] = old_qval[:]
                        y[0][m_action] = m_value # Replace the value
                        X_train.append(m_orig_state)
                        y_train.append(y)
                    X_train = np.vstack(X_train)
                    y_train = np.vstack(y_train)
                    loss = self.actor_model.train_on_batch(X_train, y_train)
                    actor_loss.append(loss)

                # Bookkeeping at the end of the turn.

                if r_t > 0 :  # Win
                    wins += 1
                else:  # Loss
                    losses += 1

                # increment the state and time information
                s_time = s_time1
                s_geohash = s_geohash1
                orig_state = new_state
                starting_geohash = new_geohash
                 # update the starting geohash in case we stay here

            day_end_time = time.time()
            # Finised Day
            if n_days % 10 == 0:  # save every ten training days
                print('---------METRICS----------')
                print("Day #: %s" % (i+1,))
                print("Wins/Losses %s/%s" % (wins, losses))
                print('Percent of moves this day that were profitable {}'.format(wins/(wins+losses)))
                print('Epsilon is {}'.format(epsilon))
                print('This day took {}'.format(day_end_time - day_start))
                print('Critic last loss  = {}'.format(critic_loss[-1:]))
                print('Actor last loss = {}'.format(actor_loss[-1:]))
                print('Last Actor-Critic fare = {}'.format(total_fare_over_time[-1:]))
                print('Last Naive fare = {}'.format(
                    total_naive_fare_over_time[-1:]))
                print("--------METIRCS END---------")
                if self.args['save_model'] == True:
                    print("Now we save model")
                    self.actor_model.save_weights(self.args['save_model_weights_actor'],
                                                  overwrite=True)
                    self.critic_model.save_weights(self.args['save_model_weights_critic'],
                                                   overwrite=True)

                    with open("model_actor.json", "w") as outfile:
                        json.dump(self.actor_model.to_json(), outfile)
                    with open("model_critic.json", "w") as outfile:
                        json.dump(self.critic_model.to_json(), outfile)

            # keep track of the fare over time
            average_fare_per_day.append(daily_fare/(wins+losses))
            percent_profitable_moves_over_time.append(wins/(wins+losses))

            day_start = day_end_time
            if epsilon > min_epsilon:
                epsilon -= (1/n_days)

        if self.args['save_model'] == True:
            print("FINISHED TRAINING!!")
            self.actor_model.save_weights(self.args['save_model_weights_actor'],
                                          overwrite=True)
            self.critic_model.save_weights(self.args['save_model_weights_critic'],
                                           overwrite=True)
            with open("model_actor.json", "w") as outfile:
                json.dump(self.actor_model.to_json(), outfile)
            with open("model_critic.json", "w") as outfile:
                json.dump(self.critic_model.to_json(), outfile)

            return actor_loss,critic_loss, total_fare_over_time, average_fare_per_day,\
                percent_profitable_moves_over_time, total_naive_fare_over_time,\
                list_of_geohashes_visited_actor_critic, naive_geohashes_visited
        else:
            return actor_loss,critic_loss, total_fare_over_time, average_fare_per_day,\
                percent_profitable_moves_over_time, total_naive_fare_over_time,\
                list_of_geohashes_visited_actor_critic, naive_geohashes_visited

def data_attributes(taxi_yellowcab_df):
    """Some random data objects needed to train the RL algorithm.
    Includes a conversion from direction index (0-8) to a
    direction (n,s,w,e,...etc). Therefore, we can use the
    gh.neighbors attribute to find the geohashes associated with each
    direction.
    Also, has a dict for geohash : geohash_index
    Contains a dict for geohash_index : geohash
    Contains a list of all times
    Contains a list of all unique geohashes"""
    list_of_output_predictions_to_direction =\
        {0: 'nw', 1: 'n', 2: 'ne', 3: 'w', 4: 'stay', 5: 'e',
         6: 'sw', 7: 's', 8: 'se'}
    list_of_unique_geohashes = taxi_yellowcab_df.geohash_pickup.unique()
    list_of_geohash_index = defaultdict(int)
    for idx, hash_n in enumerate(list_of_unique_geohashes):
        list_of_geohash_index[hash_n] = idx
    list_of_inverse_heohash_index = defaultdict(str)
    for idx, hash_n in enumerate(list_of_unique_geohashes):
        list_of_inverse_heohash_index[idx] = hash_n
    hours = [str(_) for _ in range(24)]
    minutes = [str(_) for _ in range(0, 60, 10)]
    minutes.append('00')
    list_of_time_index =[]
    for h in hours:
        for m in minutes:
            list_of_time_index.append(int(str(h)+str(m)))

    list_of_time_index = list(set(list_of_time_index))

    return list_of_output_predictions_to_direction, list_of_unique_geohashes, \
        list_of_geohash_index, list_of_time_index, list_of_inverse_heohash_index


if __name__ == '__main__':

    taxi_yellowcab_df, final_data_structure= make_dataset.main()
    ## the the data structures needed for the RL calss
    list_of_output_predictions_to_direction, list_of_unique_geohashes, \
        list_of_geohash_index, list_of_time_index, list_of_inverse_heohash_index\
         = data_attributes(taxi_yellowcab_df)

    args = {'mode':'Train','save_model':True, 'model_weights_load_actor':'model_actor_updated.h5',
           'model_weights_load_critic':'model_critic_updated.h5',
           'save_model_weights_critic':'mlp_critic_updated_25k.h5',
           'save_model_weights_actor':'mlp_actor_updated_25k.h5','test_buffer_size':2000,
           'reduce_epsilon_test':False}

    actor_critic_model = ActorCriticNYCMLP(args, 9, 2,
                                list_of_unique_geohashes, list_of_time_index,
                                list_of_geohash_index, list_of_inverse_heohash_index,
                                final_data_structure,list_of_output_predictions_to_direction)
    # train our model
    actor_loss, critic_loss, AC_fare_over_time, average_fare_per_day,\
        percent_profitable_moves_over_time, naive_fare_over_time,\
        actor_critic_geohashes_visited , naive_geohashes_visited = \
        actor_critic_model.trainer(n_days=5000, buffer_size=2000)
    #print(actor_loss,'actor loss')
    print()
    #print(critic_loss,'critic_loss')
    print()
    #print(AC_fare_over_time,'fare over time rl')
    print()
    #print(naive_fare_over_time,'naive fare over time')
    print()
    #print(actor_critic_geohashes_visited,'geohashes visited')
    print()
    #print(percent_profitable_moves_over_time,' profitbale moves over time')

    # save your metrics
    with open('actor_loss_new_5k', 'wb') as fp:
        pickle.dump(actor_loss, fp)
    with open('critic_loss_new_5k','wb') as fp:
        pickle.dump(critic_loss, fp)
    with open('naive_fare_time_new_5k','wb') as fp:
        pickle.dump(naive_fare_over_time, fp)
    with open('AC_fare_over_time_new_5k','wb') as fp:
        pickle.dump(AC_fare_over_time, fp)
    with open('percent_profitable_moves_over_time_new_5k','wb') as fp:
        pickle.dump(percent_profitable_moves_over_time, fp)
