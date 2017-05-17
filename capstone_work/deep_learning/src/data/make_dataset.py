# -*- coding: utf-8 -*-
# usr/bin/env python
import numpy as np
import os
import sys
from collections import defaultdict
import pandas as pd
from auxiliary_functions import convert_miles_to_minutes_nyc, geohash_encoding, myround, time_to_int
__author__='Jonathan Hilgart'


def open_up_data():
    """Open up the processed data from the data folder."""
    PROJECT_DIR = "/Users/jonathanhilgart/galvanize-projects/deep_learning"
    data_ = pd.read_csv(os.path.join(
        PROJECT_DIR,'data','yellow_taxi_geohash_min_day.csv'))
    data_['geohash_dropoff']=data_['geohas_dropoff']
    return data_

def group_data_by_minutes(taxi_yellowcab_df):
    """Group the dataframe by geohash, day, and time.
    Return a dataframe grouped by geohash, every 10 minutes across all of January
    and the average fare for that geohash and time combination.
    This assumes that all days throughout January are stationary and the times
    throughout each day are NOT stationary."""
    taxi_yellowcab_df['trip_time_minutes'] = \
        taxi_yellowcab_df.trip_distance.apply(lambda x: convert_miles_to_minutes_nyc(x))
    average_fare_during_day =  taxi_yellowcab_df.loc[:,('tpep_pickup_datetime',
                'trip_time_minutes','tpep_dropoff_datetime','total_amount',
                'geohash_dropoff','geohash_pickup','jan_day')]
    average_fare_during_day.tpep_pickup_datetime = pd.to_datetime(
        average_fare_during_day.tpep_pickup_datetime)
    average_fare_during_day.index = average_fare_during_day.tpep_pickup_datetime
    # gropu DF every 10 minutes
    grouped_pickup_fares = average_fare_during_day.groupby(pd.TimeGrouper(freq='10Min')).apply(
    lambda x: x.groupby('geohash_pickup').mean())
    grouped_pickup_fares  = grouped_pickup_fares.reset_index()
    # groupby again only on minutes this time
    grouped_pickup_fares['time'] = grouped_pickup_fares.\
        tpep_pickup_datetime.apply(lambda x : x.time())
    # change the trip time to be a factor of ten again
    grouped_pickup_fares.trip_time_minutes =\
        grouped_pickup_fares.trip_time_minutes.apply(lambda x: myround(x))
    average_fares_throughout_day_geohash = grouped_pickup_fares.groupby(
        ['time','geohash_pickup','jan_day','trip_time_minutes']).mean().reset_index()
    average_fares_throughout_day_geohash['average_fare'] = \
        average_fares_throughout_day_geohash['total_amount']
    average_fares_throughout_day_geohash.drop('total_amount',inplace=True,axis=1)

    ## convert the string of time into an integer
    average_fares_throughout_day_geohash.time =\
        average_fares_throughout_day_geohash.time.apply(lambda x: time_to_int(x))

    return average_fares_throughout_day_geohash

def create_final_data_structure(input_df):
    """Turn this into a dictionary where the key is the time : key is the geohas :
     value is a list of a tuple of fares,
    time, fare divided by time
    on average for that day for that geohash """
    time_geohash_fare_dict = defaultdict(lambda : defaultdict(list))

    for row in input_df.iterrows():
        # Create a dict of dict of a list of tuples - tuple is (fare, trip-minutes, fare/trip-minutes)
        time_geohash_fare_dict[row[1]['time']][row[1]['geohash_pickup']].append(
        (row[1]['average_fare'],row[1]['trip_time_minutes'], \
         row[1]['average_fare']/(row[1]['trip_time_minutes']+.000000001)))

    return time_geohash_fare_dict

def main():
    """Run the above functions to process and return the original data
    and the  final data structure."""

    print('Opening up data')
    taxi_yellowcab_original_df = open_up_data()
    print('Group data')
    average_fare_during_day = group_data_by_minutes(taxi_yellowcab_original_df)
    print('Creating the final data structure')
    final_time_geohash_fare_dict = create_final_data_structure(average_fare_during_day)
    return taxi_yellowcab_original_df,final_time_geohash_fare_dict
