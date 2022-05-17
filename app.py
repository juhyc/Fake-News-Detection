import json
from module import *
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def main_get(title=None, body=None):
    if request.method == "POST":
        title = request.form["title"]
        body = request.form["body"]
        return show_result(title, body)
    else:
        return render_template("predict.html")

@app.route("/show_result", methods=["POST", "GET"])
def show_result(title=None, body=None):
    label_probs, title_color, title_token, body_color, body_token = Visualization_Result(title, body, 1)

    if request.method == "POST":
        return render_template('result.html', label_probs=label_probs, title_color=title_color, title_token=title_token, body_color=body_color, body_token=body_token)
    # else:
    #     return render_template("main.html")

if __name__ == "__main__":
    app.run(debug=True)