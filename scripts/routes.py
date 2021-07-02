from flask import render_template, request, url_for, Response, stream_with_context, jsonify
from bokeh.models.sources import AjaxDataSource
from scripts import streamer, app, graphs, datamodels, db_abs_path
import csv
import pandas as pd
import os
from threading import Event
from queue import Queue
import pytz
from datetime import datetime
import sqlite3
from scripts.graphs import convert_timezone_np


#Queues and events to allow live classification of images from user
classification_q = Queue()
class_success_q = Queue()
class_success_ev = Event()


@app.route("/")
@app.route("/home")
def home():
    """Route for current sleep schedule summary"""
        
    con = sqlite3.connect(db_abs_path)

    #Get data for 24-hour graph
    #Select all data, convert to time series index
    df = pd.read_sql_query('SELECT * from data_point', con, index_col='timestamp')
    df.index = pd.to_datetime(df.index)

    #Create boolean nap time column, synonymous with data with baby in it. Upsample to every 1 minute
    df['nap_time'] = df.apply(lambda row: int(row.value == 'Baby'), axis=1)
    
    get_file_name = lambda row: row['image_resized_path'].split('\\')[-1]
    df['FileName'] = df.apply(get_file_name, axis=1)

    create_URL = lambda row: url_for('static', filename= 'ReadingImagesResized/' + row['FileName'])
    df['PhotoURL'] = df.apply(create_URL, axis=1)
    filldf = df.resample('.5min').ffill()

    #Build 24hr graph
    script, div = graphs.bedtime_graph(sourceDF= df, fillsourceDF = filldf)

    #Build Recent Naps Table
    #Determine edge data points by comparing 'nap time' from one record to 'nap time' from the next
    df['change_event'] = df['nap_time'] != df['nap_time'].shift(1)
    df = df.dropna()

    #Calculate start and end time for each nap. Also get string representations
    napsDF = df.loc[df['change_event'] == True]
    start_times = napsDF.index.values
    napsDF.insert(0, column= "start_time", value= convert_timezone_np(start_times))
    napsDF['start_date_string'] = napsDF['start_time'].dt.strftime('%a %b %d')
    napsDF['start_time_string'] = napsDF['start_time'].dt.strftime('%I:%M%p')

    end_times = napsDF['start_time'].shift(-1)
    napsDF.insert(1, column= "end_time", value= end_times)
    napsDF['end_time_string'] = napsDF['end_time'].dt.strftime('%I:%M%p')
    napsDF['end_time_string'].fillna('Now', inplace= True)


    #Filter for spans labeled as a nap, and remove unneccesary columns
    napsDF = napsDF.loc[napsDF['nap_time'] == True]
    napsDF = napsDF.drop(['value', 'baby_reading', 'empty_reading', 'nap_time', 'change_event'], axis=1)


    get_duration = lambda row: row['end_time'] - row['start_time']
    napsDF['duration'] = napsDF.apply(
        #Calculate duration from dataframe if end_time exists
        lambda row: get_duration(row) if not type(row['end_time']) == pd._libs.tslibs.nattype.NaTType
        else datetime.now() - row['start_time'], #Sleep is ongoing, calculate current duration
        axis=1
    )

    hour_vs_hours = lambda hour: 'hours' if hour > 1 else 'hour'
    minute_vs_sofar = lambda now: 'minutes' if not now == 'Now' else 'minutes so far'
    duration_to_string = lambda row: (
        str(row['duration'].components.hours) +
        f' {hour_vs_hours(row["duration"].components.hours)}, ' +
        str(row['duration'].components.minutes) +
        f' {minute_vs_sofar(row["end_time_string"])}'
    )
    napsDF['duration_string'] = napsDF.apply(lambda row: duration_to_string(row), axis=1)

    sleep_data = napsDF.values.tolist()
    column_names = napsDF.columns.values.tolist()
    data = [dict(zip(column_names, item)) for item in sleep_data]
    data.reverse()

    return render_template(
        'home.html',
        data= data,
        graph_script = script,
        graph_div = div
    )

@app.route('/dummyajax', methods= ['POST'])
def dummy_ajax():
    """Called on home page load to un-cache AJAX requests in iOS"""

    return jsonify(message= 'Received')

@app.route('/video_feed')
def video_feed(inqueue = classification_q, outqueue = class_success_q, event = class_success_ev):
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(stream_with_context(iter(streamer.Streamer(inqueue, outqueue, event))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ajaxprediction', methods= ['POST'])
def ajaxprediction():
    """Returns live data for ajax prediction graph"""

    global preductions_list
    return jsonify(x= streamer.predictions_list, y= [0,0,0,0])

@app.route('/classify', methods = ['POST'])
def classify(outqueue = classification_q, inqueue = class_success_q, event = class_success_ev):
    """Receives button press from user and puts value in queue for imagegenerator"""

    if request.method == 'POST':
        #Get value from request, and pass it to imagegenerator function via queue
        value = request.get_data().decode('UTF-8')
        outqueue.put(value)
        event.wait()
        response = inqueue.get()
        event.clear()
    return response

@app.route('/livefeed')
def livefeed():
    """Route for home page with video feed and live prediction reading"""

    #Data source for live prediction reading, initialize as empty lists
    ajax_source = AjaxDataSource(
        data_url= url_for('ajaxprediction'),
        polling_interval=1000,
        mode='replace'
    )
    ajax_source.data = dict(x=[], y=[])

    script, div = graphs.live_prediction_graph(ajax_source)

    return render_template('livefeed.html',
        graph_div = div,
        graph_script = script
    )

@app.route('/photo')
def photo():
    """Route for viewing individual photo"""

    photo_name = request.args.get('photo', 1, type=str)
    return render_template('photo.html', file_name = photo_name)

@app.route("/data")
def datapage():
    """Route for viewing multiple photos"""

    with open('data/data.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        csv_data = []
        value_dict = {'0.0': 'Empty', '1.0': 'Awake', '2.0': 'Asleep'}
        for original_path, resized_path, value in reader:
            file_name = resized_path.split("\\")[-1]
            csv_data.append(
                {
                    'orig': original_path,
                    'file_name': file_name,
                    'val': value_dict.get(value)
                }
            )
        csv_data.reverse()

    data_requested_page = request.args.get('page', 1, type=int)

    data_page = paginate(csv_data, 12, data_requested_page)

    return render_template(
        'trainingdata.html',
        data= csv_data[data_page.get('first_ind'):data_page.get('last_ind')],
        pagination= data_page.get('pagination_list'),
        current_page= data_requested_page,
        page_length= data_page.get('page_length')
    )

@app.route('/correct', methods = ['POST'])
def correct():
    """Updates DataPoint table with new value via ID"""
    if request.method == 'POST':
        #Get value from request, and pass it to imagegenerator function via queue
        data = request.get_json()
        item_to_update = datamodels.DataPoint.query.get(data['id'])
        item_to_update.value = data['value']
        datamodels.commit_item(item_to_update)
        print(f'ID: {data["id"]}')
        print(f'Value: {data["value"]}')
    return 'all good'


@app.route('/readings')
def readings():
    """Route for browsing raw database readings"""

    #Get list of photos/data to display on page
    readings = datamodels.DataPoint.query.all()
    readings.reverse()
    for reading in readings:
        reading.file_name = reading.image_orig_path.split('\\')[-1]
        reading.baby_reading = round(reading.baby_reading, 2)
        reading.empty_reading = round(reading.empty_reading, 2)
    readings_requested_page = request.args.get('page', 1, type=int)
    readings_page = paginate(readings, 12, readings_requested_page)

    return render_template(
        'readings.html',
        data= readings[readings_page.get('first_ind'):readings_page.get('last_ind')],
        pagination = readings_page.get('pagination_list'),
        current_page = readings_requested_page,
        page_length = readings_page.get('page_length')
    )

@app.route("/models")
def models():
    """Route for browsing paginated models"""

    model_list = os.listdir(os.path.join(os.getcwd(), 'scripts', 'static', 'modelpredictions'))
    model_list.reverse()
    model_requested_page = request.args.get('page', 1, type=int)
    model_page = paginate(model_list, 12, model_requested_page)
    
    return render_template(
        'models.html',
        data= model_list[model_page.get('first_ind'):model_page.get('last_ind')],
        pagination= model_page.get('pagination_list'),
        current_page= model_requested_page,
        page_length= model_page.get('page_length')
    )

@app.route("/models/<modelname>")
def model(modelname):
    """Route for individual model pages"""

    model_folder_path = os.path.join(os.getcwd(), 'scripts', 'static', 'modelpredictions', modelname)
    model_predictions = os.path.join(model_folder_path, 'predictions.csv')
    model_summ_path = os.path.join(model_folder_path, 'summary.txt')

    summary = get_model_summary(model_summ_path)
    
    #Determine values for previous model, next model buttons
    all_models = os.listdir('models')
    all_models.reverse()
    cur_index = all_models.index(modelname)
    if cur_index == 0:
        next_model = all_models[1]
        prev_model = modelname
    elif cur_index == len(all_models) - 1:
        next_model = modelname
        prev_model = all_models[-2]
    else:
        next_model = all_models[cur_index + 1]
        prev_model = all_models[cur_index - 1]

    model_strings = os.path.join(model_folder_path, 'strings.csv') #Not used currently

    #Load data from predictions, create URLs for each image
    df = pd.read_csv(model_predictions)
    accuracy = df['Incorrect'].value_counts(normalize=True).to_dict()
    accuracy = str(round(accuracy.get(0)*100, 2)) + "%"

    df['y'] = .5

    get_file_name = lambda row: row['ResizedPath'].split('\\')[-1]
    df['FileName'] = df.apply(get_file_name, axis=1)

    create_URL = lambda row: url_for('static', filename= 'Resized/' + row['FileName'])
    df['PhotoURL'] = df.apply(create_URL, axis=1)

    babyDF = df.loc[df['Value'] == 'Baby'].reset_index()
    baby_likeliness = lambda row: row['LikelyBaby'] - row['LikelyEmpty']
    babyDF['BabyLikeliness'] = babyDF.apply(baby_likeliness, axis=1)

    nobabyDF = df.loc[df['Value'] == 'No Baby'].reset_index()
    nobaby_likeliness = lambda row: -1*(row['LikelyEmpty'] - row['LikelyBaby'])
    nobabyDF['NoBabyLikeliness'] = nobabyDF.apply(nobaby_likeliness, axis=1)

    #Get graph components
    script, div = graphs.model_performance_graph(babyDF, nobabyDF)

    return render_template(
        'model.html',
        model= modelname,
        graph_script = script,
        graph_div = div,
        accuracy= accuracy,
        next = next_model,
        previous = prev_model,
        summary= summary
    )

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

