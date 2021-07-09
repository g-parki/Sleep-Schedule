from flask import render_template, request, url_for, Response, stream_with_context, jsonify
from bokeh.models.sources import AjaxDataSource
from scripts import streamer, app, graphs, datamodels, datahelpers
from scripts import datahelpers
import os
from threading import Event
from queue import Queue

#Queues and events to allow live classification of images from user
classification_q = Queue()
class_success_q = Queue()
class_success_ev = Event()


@app.route("/")
@app.route("/home")
def home():
    """Route for current sleep schedule summary"""

    #Get dataframe of sleep data, plus another upsampled to every minute for filling a bar graph
    df, fillDF = datahelpers.get_all_nap_data(with_upsample= True)

    #Build 24hr graph
    script, div = graphs.bedtime_graph(sourceDF= df.iloc[::2, :], fillsourceDF = fillDF)

    #Get list of all recent bedtimes, including friendly rendered text
    data = datahelpers.aggregate_nap_data(df)

    return render_template(
        'home.html',
        graph_script = script,
        graph_div = div,
        rendered_table = render_template(
            'subtemplates/bedtimestable.html',
            data= data
        )
    )

@app.route('/refreshbedtimes')
def refresh_bedtimes():

    data = datahelpers.aggregate_nap_data(datahelpers.get_all_nap_data())

    return render_template(
            'subtemplates/bedtimestable.html',
            data= data
        )

@app.route('/bedtime')
def bedtime():
    start_id = request.args.get('start', 1, type=int)
    end_id = request.args.get('end', 1, type=int)
    data, date_string, time_string_first, time_string_last = datahelpers.get_bedtime_data(start_id, end_id)
    
    return render_template(
        'bedtime.html',
        data = data,
        date_string = date_string,
        start_time_string = time_string_first,
        end_time_string = time_string_last
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

@app.route('/classifyreading', methods= ['POST'])
def classify_reading():
    """Receive ID and value of database reading to save to training data"""
    data = request.get_json()
    id = int(data['id'])
    value = str(data['value'])

    response = datahelpers.save_reading_to_training_data(id, value)
    
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

@app.route("/data", methods=['GET', 'POST'])
def datapage():
    """Route for viewing multiple photos"""

    #Data to feed to page table
    data, counts, value_names = datahelpers.get_training_data()
    data_requested_page = request.args.get('page', 1, type=int)
    is_ajax = request.args.get('ajax', False, type=bool)
    data_page = datahelpers.paginate(data, 12, data_requested_page)

    script, div = graphs.training_data_counts_bar(counts, value_names)

    table = render_template(
        'subtemplates/trainingimagestable.html',
        data= data[data_page.get('first_ind'):data_page.get('last_ind')],
    )

    pagination_nav = render_template(
        'subtemplates/paginationnav.html',
        pagination = data_page.get('pagination_list'),
        current_page= data_requested_page,
        page_length= data_page.get('page_length'),
    )

    if is_ajax:
        return {'table': table,
            'pagination_nav': pagination_nav,
            'graph_script': script,
            'graph_div': div}
    else:
        return render_template(
            'trainingdata.html',
            graph_script = script,
            graph_div = div,
            table = table,
            pagination_nav = pagination_nav
        )

@app.route('/correctdatapoint', methods = ['POST'])
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

@app.route('/correcttrainingdata', methods= ['POST'])
def correct_training_data():
    """Updates training data point with new value via file name"""
    if request.method == 'POST':
        #Get value from request, and pass it to imagegenerator function via queue
        data = request.get_json()
        print(data)
        response = datahelpers.update_training_csv(
            filename= data['file_name'],
            new_value= data['value']
        )

    return response

@app.route('/readings')
def readings():
    """Route for browsing raw database readings"""

    #Get list of photos/data to display on page
    readings = datahelpers.get_readings()
    readings_requested_page = request.args.get('page', 1, type=int)
    readings_page = datahelpers.paginate(readings, 12, readings_requested_page)

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
    model_page = datahelpers.paginate(model_list, 12, model_requested_page)
    
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
    model_predictions_csv = os.path.join(model_folder_path, 'predictions.csv')
    model_summ_path = os.path.join(model_folder_path, 'summary.txt')

    summary = datahelpers.get_model_summary(model_summ_path)
    
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
    babyDF, nobabyDF, accuracy = datahelpers.get_model_results(model_predictions_csv)

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



