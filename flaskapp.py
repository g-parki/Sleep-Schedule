from flask import Flask, render_template, request, url_for
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.embed import components
from bokeh.models import PolyAnnotation, Range1d
from bokeh.transform import jitter
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
    tools = 'pan, wheel_zoom, reset'
    TOOLTIPS = '<img src= "@PhotoURL">'

    baby_src = ColumnDataSource(data= babyDF)
    nobaby_src = ColumnDataSource(data= nobabyDF)
    JITTER_RADIUS_X = .1
    JITTER_RADIUS_Y = 1
    DOT_SIZE = 10
    DOT_ALPHA = .1
    BACK_ALPHA = .05

    p = figure(
        tooltips= TOOLTIPS,
        tools=tools,
        active_scroll= 'wheel_zoom',
        y_range=Range1d(start=-.1, end=1.05, bounds=(-2,2)),
        x_range=Range1d(start=-(1+1.5*(JITTER_RADIUS_X)), end=1+1.5*JITTER_RADIUS_X, bounds=(-2,2))
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
        legend_label="Photos Without Baby            "
    )

    p.circle(
        jitter('BabyLikeliness', JITTER_RADIUS_X),
        jitter('y', JITTER_RADIUS_Y),
        source= baby_src,
        size=DOT_SIZE,
        color='blue', alpha=DOT_ALPHA,
        legend_label="Photos With Baby   "
    )

    p.sizing_mode = 'scale_width'

    p.yaxis.visible = False
    p.ygrid.visible = False
    p.xaxis.axis_label = '<- Predicted to Not Have Baby             Predicted to Have Baby ->      '
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_font_style = 'bold'
    p.xaxis.major_label_text_font_size = '0pt'  #turn off x-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt' #turn off y-axid tick labels
    

    p.legend.location = "bottom_center"
    p.legend.click_policy="hide"
    p.legend.label_text_font_size = '12pt'
    p.legend.label_text_font_style = 'normal'
    p.legend.orientation='horizontal'
    p.legend.glyph_height= 20
    p.legend.background_fill_alpha = 1.0
    p.legend.border_line_width = 0

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