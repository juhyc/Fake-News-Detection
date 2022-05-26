# -*- coding: utf-8 -*-
import json
from module import *
from flask import Flask, request, render_template, jsonify


app = Flask(__name__)

@app.route("/")
def main():
    return render_template("home.html", ishidden='hidden')

@app.route("/predict", methods=['GET', 'POST'])
def main_get(title=None, body=None):
    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        return show_result(title, body)
    else:
        return render_template("home.html", ishidden='hidden')

@app.route("/show_result", methods=["POST", "GET"])
def show_result(title=None, body=None):
    label_probs, title_color, title_token, body_color, body_token = Visualization_Result(title, body, 1)
    label_probs = label_probs.astype(np.float)
    label_probs[0][0] = round(label_probs[0][0] * 100, 2)
    label_probs[0][1] = round(label_probs[0][1] * 100, 2)
    label_probs = label_probs[0].tolist()

    if request.method == "POST":
        return render_template('home.html', title=title, body=body, label_probs=label_probs, title_color=title_color, title_token=title_token, body_color=body_color, body_token=body_token,  ishidden='visible')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/fakenews')
def fakenews():
    return render_template('fakenews.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route("/mysuperplot", methods=["GET"])
def plotView():
    # Generate plot


    return render_template("image.html", image=pngImageB64String)


if __name__ == "__main__":
    app.run(debug=True)