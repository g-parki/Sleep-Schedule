from flask import Flask, render_template, request, url_for
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.embed import components
from bokeh.models import PolyAnnotation, Range1d
import csv
import pandas as pd
from urllib.request import pathname2url
import os

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    
    return render_template('home.html')

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
        page_length= data_page.get('pages')
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
        page_length= model_page.get('pages')
    )

@app.route("/models/<modelname>")
def model(modelname):
    model_folder_path = os.path.join(os.getcwd(), 'static', 'modelpredictions', modelname)
    model_predictions = os.path.join(model_folder_path, 'predictions.csv')
    model_strings = os.path.join(model_folder_path, 'strings.csv') #Not used currently

    #Load data from predictions, create URLs for each image
    df = pd.read_csv(model_predictions)
    get_file_name = lambda row: row['ResizedPath'].split('\\')[-1]
    df['FileName'] = df.apply(get_file_name, axis=1)
    create_URL = lambda row: url_for('static', filename= 'Resized/' + row['FileName'])
    df['PhotoURL'] = df.apply(create_URL, axis=1)
    accurateDF = df.loc[df['Incorrect'] == 0].reset_index()
    inaccurateDF = df.loc[df['Incorrect'] == 1].reset_index()

    #Create plot
    tools = 'pan, wheel_zoom, reset'
    TOOLTIPS = '<img src= "@PhotoURL">'

    p = figure(
        title= f'Model {modelname} Predictions',
        tooltips= TOOLTIPS,
        tools=tools,
        active_scroll= 'wheel_zoom',
        y_range=Range1d(start=-.05, end=1.05, bounds=(-2,2)),
        x_range=Range1d(start=-.05, end=1.05, bounds=(-2,2))
    )
    accurate_baby = ColumnDataSource(data= accurateDF.loc[accurateDF['Prediction'] == 'Baby'])
    accurate_empty = ColumnDataSource(data= accurateDF.loc[accurateDF['Prediction'] == 'No Baby'])
    inaccurate_baby = ColumnDataSource(data= inaccurateDF.loc[inaccurateDF['Prediction'] == 'No Baby'])
    inaccurate_empty = ColumnDataSource(data= inaccurateDF.loc[inaccurateDF['Prediction'] == 'Baby'])
    p.circle('LikelyEmpty','LikelyBaby', source= accurate_baby, size=15, color='blue', alpha=0.4)
    p.circle('LikelyEmpty', 'LikelyBaby', source= accurate_empty, size=15, color='red', alpha=0.4)
    p.circle('LikelyEmpty','LikelyBaby', source= inaccurate_baby, size=15, color='blue', alpha=0.4)
    p.circle('LikelyEmpty', 'LikelyBaby', source= inaccurate_empty, size=15, color='red', alpha=0.4)
    p.sizing_mode = 'scale_width'
    red_polygon = PolyAnnotation(
        fill_color="red",
        fill_alpha=0.08,
        xs=[-10, 10, 10],
        ys=[-10, 10, -10],
    )
    blue_polygon = PolyAnnotation(
        fill_color="blue",
        fill_alpha=0.05,
        xs=[-10, 10, -10],
        ys=[-10, 10, 10],
    )
    p.add_layout(red_polygon)
    p.add_layout(blue_polygon)
    script, div = components(p)

    return render_template(
        'model.html',
        model= modelname,
        graph_script = script,
        graph_div = div
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

if __name__ == '__main__':
    app.run(debug= True, host='192.168.1.17')