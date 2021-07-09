import sys
from pathlib import Path

from numpy.core.fromnumeric import resize
sys.path.insert(0, str(Path(sys.path[0]).parent))

from datetime import datetime, date, timedelta
from scripts import datamodels, db
import pandas as pd
import numpy as np
import pytz
from flask import url_for
import csv


#Shared constants
TRAININGDATA_VALUE_DICT = {'0.0': 'Empty', '1.0': 'Awake', '2.0': 'Asleep'}

#Shared dataframe column calculations
#Requires field to be passed in:
get_file_name = lambda file_path: file_path.split('\\')[-1]
get_folder_name = lambda file_path: file_path.split('\\')[-2]
round_number_for_display = lambda number: round(number, 2)
time_string = lambda time_obj: time_obj.strftime('%I:%M %p')
date_string = lambda time_obj: time_obj.strftime('%a %b %d')
time_string = lambda time_obj: time_obj.strftime('%I:%M %p')
is_NaT = lambda time_obj: type(time_obj) == pd._libs.tslibs.nattype.NaTType
#Assumes column name, so row to be passed in:
get_val_string = lambda row: TRAININGDATA_VALUE_DICT.get(str(row['Value']))

def get_all_nap_data(with_upsample = False):
    """Returns dataframe of all bedtime readings, and optionally an upsampled dataframe"""

    df = get_readings(as_df = True, with_values = False)

    #Create boolean nap time column, synonymous with data with baby in it. Upsample to every 1 minute
    df['nap_time'] = df.apply(lambda row: int(row.value == 'Baby'), axis=1)
    df['file_name'] = df.apply(lambda row: get_file_name(row['image_resized_path']), axis= 1)

    create_URL = lambda row: url_for('static', filename= 'ReadingImagesResized/' + row['file_name'])
    df['photo_url'] = df.apply(create_URL, axis=1)
    
    if with_upsample:
        fillDF = df.resample('.5min').ffill()
        return df, fillDF
    else:
        return df


def aggregate_nap_data(df):
    """Return dictionary list of bedtimes with friendly rendered text"""

    #Build Recent Naps Table
    #Determine edge data points by comparing 'nap time' from one record to 'nap time' from the next
    df['change_event'] = df['nap_time'] != df['nap_time'].shift(1)
    df = df.dropna(subset=['nap_time'])

    #Calculate start and end time for each nap. Also get string representations
    napsDF = df.loc[df['change_event'] == True]

    napsDF['start_date_string'] = napsDF.apply(lambda row: today_or_yesterday(row['start_time']), axis=1)
    napsDF['start_time_string'] = napsDF.apply(lambda row: time_string(row['start_time']), axis=1)

    #Get ID# of datapoint for end of get_bedtime_date
    #Shift ID's by one. Fill na's with -1. This indicates bedtime is ongoing.
    #Convert to int to avoid decimals in URL args
    napsDF['end_id'] = napsDF['id'].shift(-1).fillna(-1)
    napsDF['end_id'] = napsDF['end_id'].astype('int')

    end_times = napsDF['start_time'].shift(-1)
    napsDF.insert(1, column= "end_time", value= end_times)
    napsDF['end_time_string'] = napsDF.apply(
        lambda row: time_string(row['end_time']) if not is_NaT(row['end_time'])
        else 'Now',
        axis=1
    )

    #Filter for spans labeled as a nap, and remove unneccesary columns
    napsDF = napsDF.loc[napsDF['nap_time'] == True]
    napsDF = napsDF.drop(['value', 'baby_reading', 'empty_reading', 'nap_time', 'change_event'], axis=1)


    get_duration = lambda row: row['end_time'] - row['start_time']
    napsDF['duration'] = napsDF.apply(
        #Calculate duration from dataframe if end_time exists
        lambda row: get_duration(row) if not is_NaT(row['end_time'])
        else datetime.now() - row['start_time'], #Sleep is ongoing, calculate current duration
        axis=1
    )

    napsDF['duration_string'] = napsDF.apply(duration_to_string, axis=1)

    return df_to_dict_list(napsDF, reverse= True)

def get_training_data():
    """Returns list of all training data and value counts"""

    df = pd.read_csv('data/data.csv')
    df['value'] = df.apply(get_val_string, axis=1)
    df['folder_resized'] = df.apply(lambda row: get_folder_name(row['ResizedPath']), axis=1)
    df['folder_original'] = df.apply(lambda row: get_folder_name(row['FilePath']), axis=1)
    df['file_name'] = df.apply(lambda row: get_file_name(row['FilePath']), axis=1)

    #Info for bar chart on data page
    counts = df['value'].value_counts().to_list()
    values = df['value'].value_counts().index.to_list()

    return df_to_dict_list(df), counts, values

def get_readings(as_df = False, with_values = True):

    query = db.session.query(datamodels.DataPoint)
    df = pd.read_sql_query(query.statement, db.session.bind, index_col='timestamp')
    df['file_name'] = df.apply(lambda row: get_file_name(row['image_orig_path']), axis=1)

    start_times = df.index.values
    df.insert(0, column= "start_time", value= convert_timezone_np(start_times))
    
    if with_values:
        for field in ['baby_reading', 'empty_reading']:
            df[field] = df.apply(lambda row: round_number_for_display(row[field]), axis=1)
    if as_df:
        return df
    else:
        return df_to_dict_list(df)


def get_bedtime_data(start_id, end_id):
    """Returns datalist, beginning time string, and ending time string
    of photos between a start id and an end id"""

    if end_id == -1:
        #Get ID of most recent datapoint
        end_id = datamodels.DataPoint.query.order_by(-datamodels.DataPoint.id).first().id
        #Set string to represent that the current bed time hasn't ended
        time_string_last = 'Now'
    else:
        time_string_last = ''

    query = db.session.query(datamodels.DataPoint).filter(
        datamodels.DataPoint.id.between(start_id, end_id)
    )

    df = pd.read_sql_query(query.statement, db.session.bind, index_col='timestamp')
    df.insert(0, column= "time", value= convert_timezone_np(df.index.values))

    df['time_string'] = df.apply(lambda row: time_string(row['time']), axis=1)
    df['file_name'] = df.apply(lambda row: get_file_name(row['image_resized_path']), axis=1)

    date_string = today_or_yesterday(df.iloc[0]['time'])
    time_string_first = df.iloc[0]['time_string']
    #Check if time_string_last has already been set to "Now"
    if not time_string_last == 'Now':
        time_string_last = df.iloc[-1]['time_string']

    return df_to_dict_list(df), date_string, time_string_first, time_string_last

def df_to_dict_list(df, reverse= True):
    """Returns data as list of dictionaries for each item in a dataframe"""

    data = df.values.tolist()
    column_names = df.columns.values.tolist()
    data = [dict(zip(column_names, item)) for item in data]
    if reverse:
        data.reverse()
    return data

def duration_to_string(row):
    """Returns friendly string version of duration"""
    #0:51 -> 51 minutes
    #1:51 -> 1 hour 51 minutes
    #2:51 -> 2 hours 51 mintues
    plural_or_not = lambda item: '' if item == 1 else 's'
    so_far = lambda now: '' if not now == 'Now' else ' so far'
    
    returned_string = ''
    hours = row['duration'].components.hours
    minutes = row['duration'].components.minutes + 1*(row['duration'].components.seconds > 29)
    #Round up to next hour if minutes is 60
    if minutes == 60:
        hours += 1
        minutes = 0

    if hours:
        returned_string += (
            str(hours) +
            f' hour{plural_or_not(hours)}'
        )
        if minutes:
           returned_string += ', '
    if minutes:
        returned_string += (
            str(minutes) +
            f' minute{plural_or_not(minutes)}{so_far(row["end_time_string"])}'
        )
    return returned_string

def save_reading_to_training_data(id, value):
    """Saves existing database reading to training data CSV"""
    data = datamodels.DataPoint.query.filter(datamodels.DataPoint.id == id)[0]
    original_path = data.image_orig_path
    resized_path = data.image_resized_path

    with open('./data/data.csv', 'a', newline='') as f:
        csv.writer(f).writerow([original_path,resized_path,value])

    data.in_training_data = True
    datamodels.commit_item(data)

    return f'{original_path} {resized_path} {id} {value}'

def update_training_csv(filename, new_value):
    
    df = pd.read_csv('data/data.csv')
    df['file_name'] = df.apply(lambda row: get_file_name(row['FilePath']), axis=1)
    df.loc[df['file_name'] == filename, 'Value'] = new_value
    df.drop(['file_name'], axis=1, inplace=True)
    df.to_csv('data/data.csv', index=False)

    return 'Success'

def get_model_summary(path):
    """Returns friendlier version of model summary text"""

    with open(path, 'r') as f:
        lines = f.readlines()
    shitty_strings = ['====', '____']
    stringlist = [line[:-2] for line in lines]
    stringlist = [
        line.capitalize() for line in stringlist 
        if shitty_strings[0] not in line 
        and shitty_strings[1] not in line
    ]
    return stringlist

def get_model_results(predictions_csv):
    
    df = pd.read_csv(predictions_csv)
    accuracy = df['Incorrect'].value_counts(normalize=True).to_dict()
    accuracy = str(round(accuracy.get(0)*100, 2)) + "%"

    df['y'] = .5

    df['folder_resized'] = df.apply(lambda row: get_folder_name(row['ResizedPath']), axis=1)
    df['file_name'] = df.apply(lambda row: get_file_name(row['ResizedPath']), axis=1)

    create_URL = lambda row: url_for('static', filename= row['folder_resized'] + '/' + row['file_name'])
    df['photo_url'] = df.apply(create_URL, axis=1)

    babyDF = df.loc[df['Value'] == 'Baby'].reset_index()
    baby_likeliness = lambda row: row['LikelyBaby'] - row['LikelyEmpty']
    babyDF['BabyLikeliness'] = babyDF.apply(baby_likeliness, axis=1)

    nobabyDF = df.loc[df['Value'] == 'No Baby'].reset_index()
    nobaby_likeliness = lambda row: -1*(row['LikelyEmpty'] - row['LikelyBaby'])
    nobabyDF['NoBabyLikeliness'] = nobabyDF.apply(nobaby_likeliness, axis=1)
    
    return babyDF, nobabyDF, accuracy

def convert_timezone(timestamp):
    ts = pd.to_datetime(timestamp)
    datetime_obj = datetime(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)
    #return pytz.utc.localize(datetime_obj)
    return pytz.utc.localize(datetime_obj).astimezone(pytz.timezone('US/Pacific')).replace(tzinfo=None)

convert_timezone_np = np.vectorize(convert_timezone)

def today_or_yesterday(date_obj):
    """Returns friendly string version of a date"""
    obj_date_string = date_string(date_obj)
    if obj_date_string == date_string(date.today()):
        friendly_string = 'Today'
    elif obj_date_string == date_string(date.today() - timedelta(hours= 24)):
        friendly_string = 'Yesterday'
    else:
        friendly_string = obj_date_string
    return friendly_string

def paginate(list_to_pag, items_per_pag, page_requested):
    """Returns data to construct pagination buttons"""
    
    pages = 1 + len(list_to_pag)//items_per_pag
    pagination_rad = 3
    

    first_index =(page_requested - 1)*12
    last_index = min(page_requested*12, len(list_to_pag))

    #Adjust pagination radius for the pages on the end
    if page_requested <= pagination_rad:
        pagination = [
            i for i in range(1, pages + 1)
            if abs(page_requested - i) < abs(2*pagination_rad - page_requested)
        ]
    elif page_requested > pages - pagination_rad:
        pagination = [
            i for i in range(1, pages + 1)
            if abs(page_requested - i) < abs(2*pagination_rad - (pages - page_requested +1))
        ]
    #Get results within pagination radius for the pages in the middle
    else:
        pagination = [
            i for i in range(1, pages + 1)
            if abs(page_requested - i) <= pagination_rad-1
        ]
    
    #Get values for "previous" and "next" page buttons
    previous_page = max(1, page_requested-1)
    next_page = min(pages, page_requested+1)
    
    #Compose one list of all button values
    pagination.insert(0, previous_page)
    pagination.append(next_page)

    return {
        'pagination_list': pagination,
        'first_ind': first_index,
        'last_ind': last_index,
        'page_length': pages
    }