from flask import Flask, render_template, request
import csv
import pandas as pd
from urllib.request import pathname2url

app = Flask(__name__)
data = pd.read_csv('data.csv')


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

    ITEMS_PER_PAGE = 12
    pagination_rad = 3
    pages = 1 + (len(csv_data)//ITEMS_PER_PAGE)
    requested_page = request.args.get('page', 1, type=int)

    first_index =(requested_page - 1)*12
    last_index = min(requested_page*12, len(csv_data))

    #Adjust pagination radius for the pages on the end
    if requested_page <= pagination_rad:
        pagination = [i for i in range(1, pages + 1) if abs(requested_page - i) < abs(2*pagination_rad - requested_page)]
    elif requested_page > pages - pagination_rad:
        pagination = [i for i in range(1, pages + 1) if abs(requested_page - i) < abs(2*pagination_rad - (pages - requested_page +1))]
    #Get results within pagination radius for the pages in the middle
    else:
        pagination = [i for i in range(1, pages + 1) if abs(requested_page - i) <= pagination_rad-1]
    previous_page = max(1, requested_page-1)
    next_page = min(pages, requested_page+1)
    pagination.insert(0, previous_page)
    pagination.append(next_page)

    return render_template(
        'data.html',
        data= csv_data[first_index:last_index],
        pagination= pagination,
        current_page= requested_page,
        page_length= pages
    )

@app.route("/models")
def models():
    return render_template('models.html')

if __name__ == '__main__':
    app.run(debug= True, host='192.168.1.17')