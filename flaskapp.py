from flask import Flask, render_template, request, url_for, Response, redirect
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.embed import components
from bokeh.models import PolyAnnotation, Range1d, PanTool, WheelZoomTool, ResetTool
from bokeh.transform import jitter
import main
import csv
import pandas as pd
from urllib.request import pathname2url
import os
from queue import Queue

app = Flask(__name__)
classification_q = Queue()
class_success_q = Queue()

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed(inqueue = classification_q, outqueue = class_success_q):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(image_generator(inqueue, outqueue),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify', methods = ['POST'])
def classify(outqueue = classification_q, inqueue = class_success_q):
    if request.method == 'POST':
        value = request.get_data().decode('UTF-8')
        outqueue.put(value)
        while inqueue.empty():
            pass
    return inqueue.get()

@app.route("/data")
def datapage():
    
    with open('data.csv', 'r', newline='') as f:
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
        'data.html',
        data= csv_data[data_page.get('first_ind'):data_page.get('last_ind')],
        pagination= data_page.get('pagination_list'),
        current_page= data_requested_page,
        page_length= data_page.get('page_length')
    )

@app.route("/models")
def models():
    model_list = os.listdir(os.path.join(os.getcwd(), 'static', 'modelpredictions'))
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
    model_folder_path = os.path.join(os.getcwd(), 'static', 'modelpredictions', modelname)
    model_predictions = os.path.join(model_folder_path, 'predictions.csv')
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

    #Create plot
    tools = [PanTool(), WheelZoomTool(maintain_focus= False), ResetTool()]
    TOOLTIPS = '<img src= "@PhotoURL">'

    baby_src = ColumnDataSource(data= babyDF)
    nobaby_src = ColumnDataSource(data= nobabyDF)
    JITTER_RADIUS_X = .11
    JITTER_RADIUS_Y = 1
    DOT_SIZE = 10
    DOT_ALPHA = .1
    BACK_ALPHA = .05

    p = figure(
        tooltips= TOOLTIPS,
        tools=tools,
        toolbar_location="below",
        toolbar_sticky=False,
        active_scroll= tools[1],
        x_axis_location="above",
        y_range=Range1d(start=-.02, end=1.08, bounds=(-.25,1.25)),
        x_range=Range1d(start=-(1+1.5*(JITTER_RADIUS_X)), end=1+1.5*JITTER_RADIUS_X, bounds=(-1.25,1.25))
    )

    red_polygon = PolyAnnotation(
        fill_color="crimson",
        fill_alpha=BACK_ALPHA,
        xs=[0, 0, -10, -10],
        ys=[-10, 10, 10, -10],
    )
    blue_polygon = PolyAnnotation(
        fill_color="dodgerblue",
        fill_alpha=BACK_ALPHA,
        xs=[0, 0, 10, 10],
        ys=[-10, 10, 10, -10],
    )
    p.add_layout(red_polygon)
    p.add_layout(blue_polygon)

    p.circle(
        jitter('NoBabyLikeliness', JITTER_RADIUS_X),
        jitter('y', JITTER_RADIUS_Y),
        source= nobaby_src,
        size=DOT_SIZE,
        color='red', alpha=DOT_ALPHA,
        legend_label="Photos without baby  "
    )

    p.circle(
        jitter('BabyLikeliness', JITTER_RADIUS_X),
        jitter('y', JITTER_RADIUS_Y),
        source= baby_src,
        size=DOT_SIZE,
        color='blue', alpha=DOT_ALPHA,
        legend_label="Photos with baby   "
    )

    p.sizing_mode = 'scale_both'

    p.yaxis.visible = False
    p.ygrid.visible = False
    p.xaxis.axis_label = '<- Predicted to Not Have Baby    Predicted to Have Baby ->      '
    p.xaxis.axis_label_text_font_size = '8pt'
    p.xaxis.axis_label_text_font_style = 'normal'

    p.xaxis.major_label_text_font_size = '0pt'  #turn off x-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt' #turn off y-axid tick labels
    

    p.legend.location = "top_center"
    p.legend.click_policy="hide"
    p.legend.label_text_font_size = '8pt'
    p.legend.label_text_font_style = 'normal'
    p.legend.orientation='vertical'
    p.legend.glyph_height= 20
    p.legend.background_fill_alpha = 0.7
    p.legend.border_line_width = 1

    script, div = components(p)

    return render_template(
        'model.html',
        model= modelname,
        graph_script = script,
        graph_div = div,
        accuracy= accuracy
    )


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

def image_generator(inqueue, outqueue):
    
    import time
    from threading import Thread, Event, enumerate
    from queue import Queue
    import shareglobals
    from datetime import datetime, timedelta
    from cv2 import waitKey, cv2
    
    stream_url = main.start_stream()
    frame_q = Queue()
    main_end_event = Event()
    response_event = Event()
    stream_gone_event = Event()
    OUTPUT_DIRECTORY_ORIGINALS = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep Schedule\\static\\Originals'
    OUTPUT_DIRECTORY_RESIZED = 'C:\\Users\\parki\\Documents\\GitHub\\Python-Practice\\Sleep Schedule\\static\\Resized'

    model = main.load_model(main.get_recent_model())

    initial_reader_thread = Thread(
        target= main.stream_reader,
        args= [frame_q, main_end_event, response_event, stream_gone_event, stream_url],
        daemon= True
    )
    initial_reader_thread.start()
    
    while True:
        #Exit loop if stream reader threads can't read frame
        if stream_gone_event.isSet():
            break

        #Start refreshed stream if current one only has 30 seconds left
        if shareglobals.current_stream_expiration_time < datetime.now() + timedelta(seconds=30) \
            and 'Refresh Thread' not in [thread.name for thread in enumerate()]:
            t = Thread(
                name= 'Refresh Thread',
                target= main.refresh_stream_token,
                args= [],
                daemon= True
            )
            t.start()
            print(f'{t.name} started')
            print(f'Active threads: {[thread.name for thread in enumerate()]}')
        
        #Check if reader thread has placed frame in queue
        if not frame_q.empty():
            frame = frame_q.get()
            frame = main.predictor(frame, model)

            if not inqueue.empty() and frame.filename not in os.listdir(OUTPUT_DIRECTORY_ORIGINALS):
            #Save Original
                output_path_originals = f'{OUTPUT_DIRECTORY_ORIGINALS}\\{frame.filename}'
                cv2.imwrite(output_path_originals, frame.original)
            
            #Save resized/greyscale copy
                output_path_resized = f'{OUTPUT_DIRECTORY_RESIZED}\\{frame.filename}'
                cv2.imwrite(output_path_resized, frame.smallsize)
                value = inqueue.get()
                print(f'{output_path_originals} value {value}')
                with open('data.csv', 'a', newline='') as f:
                    csv.writer(f).writerow([output_path_originals,output_path_resized,value])
                
                outqueue.put(f'{frame.filename} saved with value {value}')
                
            key = waitKey(35)

            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame.prediction_image_en + b'\r\n')
            # yield (b'--frame\r\n'
            # b'Content-Type: image/jpeg\r\n\r\n' + frame.frame_grey_en + b'\r\n')
            
            #Listen for keypress
            
    main_end_event.set()

if __name__ == '__main__':
    app.run(debug= True, host='192.168.1.17')