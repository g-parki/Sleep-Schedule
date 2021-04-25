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
    with open('pics.html', 'w') as f:
        f.truncate(0)
        f.write('<html>')
        COLUMNS = 4
        rows = len(data)//COLUMNS
        remainder = len(data)%COLUMNS
        i=0 #global index
        f.write('<table>')
        for _ in range(rows):
            f.write('<tr>')
            for _ in range(COLUMNS):
                original_file_path = data[len(data)-i-1][0]
                resized_file_path = data[len(data)-i-1][1]
                value = value_dict.get(data[len(data)-i-1][2])
                f.write('<td style="border: 1px solid black">')
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
        f.write('</html>')
    return None

if __name__ == '__main__':
    updateHTML()
    webbrowser.open('pics.html')