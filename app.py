# -*- coding: utf-8 -*-
import json
from module import *
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("home.html")

@app.route("/predict", methods=['GET', 'POST'])
def main_get(title=None, body=None):
    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        return show_result(title, body)
    else:
        return render_template("home.html")

@app.route("/show_result", methods=['GET', 'POST'])
def show_result(title=None, body=None):
    label_probs, title_color, title_token, body_color, body_token = Visualization_Result(title, body, 1)

    if request.method == "POST":
        return render_template('home.html', Title='Title: ', Body='Body: ', label_probs=label_probs, title_color=title_color, title_token=title_token, body_color=body_color, body_token=body_token)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/fakenews')
def fakenews():
    return render_template('fakenews.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)