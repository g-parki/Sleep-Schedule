import csv
import webbrowser


def updateHTML():
    with open('data.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)

        data = []
        for original_path, resized_path, value in reader:
            data.append([original_path, resized_path, value])


    value_dict = {'0.0': 'Empty', '1.0': 'Awake', '2.0': 'Asleep' }
    color_dict = {'Awake': '#DEC9F5', 'Asleep': '#9F78CC', 'Empty': '#66BAFA'}
    with open('pics.html', 'w') as f:
        f.truncate(0)
        
        COLUMNS = 4
        rows = len(data)//COLUMNS
        remainder = len(data)%COLUMNS
        i=0 #global index
        
        values = [item[2] for item in data]
  
        count_data_script = ("<script type='text/javascript' src='https://www.gstatic.com/charts/loader.js'></script>"
                     +"<script type='text/javascript'>"
                     +"google.charts.load('current', {'packages':['corechart']});"
                     +"google.charts.setOnLoadCallback(drawChart);"
                     +"function drawChart(){"
                     +"var data = google.visualization.arrayToDataTable(["
                     +"['Value', 'Data Count'],"
                     +f"['Awake', {values.count('1.0')}],"
                     +f"['Asleep', {values.count('2.0')}],"
                     +f"['Empty', {values.count('0.0')}]"
                     +"]);"
                     +"var options = {"
                     +f"title: 'Data Counts - {len(values)} Total', "
                     +"pieSliceText: 'value', "
                     +f"colors:['{color_dict.get('Awake')}', '{color_dict.get('Asleep')}', '{color_dict.get('Empty')}'], "
                     +"slices: {2: {offset: 0.1}}, "
                     +"legend: ''};"
                     +"var chart = new google.visualization.PieChart(document.getElementById('graphcontainer'));"
                     +"chart.draw(data, options);}"
                     +"</script>")

        
        f.write('<html>')
        f.write('<head>')
        f.write(count_data_script)
        f.write('</head>')
        f.write('<body>')
        f.write('<div id= "graphcontainer" style="height: 35%">')
        f.write('</div>')
        f.write('<div id= "tablecontainer">')
        f.write('<table style="overflow-y: scroll; height: 65%; display: block">')
        for _ in range(rows):
            f.write('<tr>')
            for _ in range(COLUMNS):
                original_file_path = data[len(data)-i-1][0]
                resized_file_path = data[len(data)-i-1][1]
                value = value_dict.get(data[len(data)-i-1][2])
                f.write(f'<td style="border: 1px solid black; background-color:{color_dict.get(value)}">')
                f.write(f"<p>{value}</p><a href='{original_file_path}'><image src='{resized_file_path}'></image></a>")
                f.write('</td>')
                i += 1
            f.write('</tr>')

        if remainder:
            f.write('<tr>')
            for _ in range(remainder):
                original_file_path = data[len(data)-i-1][0]
                resized_file_path = data[len(data)-i-1][1]
                value = value_dict.get(data[len(data)-i-1][2])
                f.write('<td>')
                f.write(f"<p>{value}</p><a href='{original_file_path}'><image src='{resized_file_path}'></image></a>")
                f.write('</td>')
                i += 1
            f.write('</tr>')
        f.write('</table>')
        f.write('</div>')
        f.write('</body>')
        f.write('</html>')
    return None

if __name__ == '__main__':
    updateHTML()
    webbrowser.open('pics.html')