def geohash_encoding(taxi_df,precision_=5):
    """Encode the latitude and longtitude of the pickup and dropoff into a geohash. For reference, the precision
    of a geohash depends on the number of characters present.
    _PRECISION = { 0: 20000000, 1: 5003530, 2: 625441, 3: 123264, 4: 19545,
    5: 3803, 6: 610, 7: 118, 8: 19, 9: 3.71, 10: 0.6, } - all distances in meters.

    Average taxi speed in 2014 was 5.51 miles per hour, or 3.8 meters per second.
    Precision of 5, means that a taxi should traverse this square in ~15 minutes avg. speed 8.6 MPH.
    Precision of 6, means a taxi should traverse square in ~3 minutes using average speed of 8.6mph.
    Precision of 7, means a taxi should traverse the square in ~1 minute.

    Input: Pandas DF
    Output: Pandas DF with geohas column appended"""
    new_taxidf = taxi_df.copy()
    geo_hash_pickup = []
    geo_hash_dropoff = []
    for idx,row in enumerate(new_taxidf.iterrows()):
        if idx%100000==0:
            print("Finished row number {}".format(idx))
        # longittude followed by latitude

            pickup_vector = np.array( [row[1]['pickup_latitude'],row[1]['pickup_longitude']])
            dropoff_vector = np.array( [row[1]['dropoff_latitude'],row[1]['dropoff_longitude']])
            #geohash encoding
            has = gh.encode(pickup_vector[1],pickup_vector[0],precision_)
            geo_hash_pickup.append(gh.encode(pickup_vector[0],pickup_vector[1],precision_))
            geo_hash_dropoff.append(gh.encode(dropoff_vector[0],dropoff_vector[1],precision_))
        else:
            pickup_vector = np.array( [row[1]['pickup_latitude'],row[1]['pickup_longitude']])
            dropoff_vector = np.array( [row[1]['dropoff_latitude'],row[1]['dropoff_longitude']])
            #geohash encoding
            has = gh.encode(pickup_vector[1],pickup_vector[0],precision_)
            geo_hash_pickup.append(gh.encode(pickup_vector[0],pickup_vector[1],precision_))
            geo_hash_dropoff.append(gh.encode(dropoff_vector[0],dropoff_vector[1],precision_))

    ## add in the new geohash columns
    new_taxidf['geohash_pickup'] = geo_hash_pickup
    new_taxidf['geohash_dropoff'] = geo_hash_dropoff
    return new_taxidf


    # Convert trip distance into minutes using the average speed of 10 MPH ( 8.6 in Manhattan + faster in the burbs)
def convert_miles_to_minutes_nyc(input_distance):
    """Convert MPH to minutes using aveage speed of 10 mph. Round to this nearest 10 minutes"""
    mph=10
    def myround(x, base=10):
        nearest_five = int(base * round(float(x)/base))
        if nearest_five ==0:
            return 10 ## shortest trip is ten minutes
        else:
            return nearest_five
    minutes = input_distance /mph *60
    return myround(minutes)

# change the trip time to be a factor of ten again
def myround(x, base=10):
    """Round to the near 10 minutes"""
    nearest_five = int(base * round(float(x)/base))
    if nearest_five ==0:
        return 10 ## shortest trip is ten minutes
    else:
        return nearest_five

## convert the string of time into an integer
def time_to_int(time):
    """Convert a timestamp into an integer representation."""
    hour = str(time.hour)
    minutes = str(time.minute)
    if len(hour)==1:
        hour ='0'+hour
    if len(minutes)==1:
        minutes = '0'+minutes
    return int(hour+minutes)

def convert_timestamp_to_time_units(df_):
    """Transform a timestamp into minute column, da column, month column."""
    df_['jan_day']=\
    [dateutil.parser.parse(i).day for i in df_.tpep_dropoff_datetime]

    df_['jan_minute']=[dateutil.parser.parse(i).minute
                       for i in df_.tpep_dropoff_datetime]
    df_['month'] =[i.month for i in df_.tpep_pickup_datetime]

list_of_output_predictions_to_direction={0:'nw',1:'n',2:'ne',3:'w',4:'stay',5:'e',6:'sw',7:'s',8:'se'}
