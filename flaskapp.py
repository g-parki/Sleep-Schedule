from flask import Flask, render_template, request
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