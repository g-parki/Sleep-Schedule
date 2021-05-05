from flask import Flask, render_template

app = Flask(__name__)
data = [1, 2, 3, 4]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', data= data)

@app.route("/about")
def about():
    return render_template('about.html', data= data)


if __name__ == '__main__':
    app.run(debug= False, host='192.168.1.17')