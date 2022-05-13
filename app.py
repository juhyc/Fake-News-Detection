import json
from module import *
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
@app.route("/main", methods=["POST"])
def main(title=None, body=None):
    if request.method == "POST":
        # title = request.form["title"]
        # body = request.form["body"]
        # label_probs, title_color, title_token, body_color, body_token = Visualization_Result(title, body, 1)
        label_probs, title_color, title_token, body_color, body_token = [[9.9950576e-01, 4.9425469e-04]], ['#ff0000', '#ff0000', '#ff0000', '#ff0000'], ['금호석유화학', '분기', '매출', '종합'], ['#ff0000', '#ff0000', '#ff0000', '#ff0000'], ['바보', '메롱']
        return render_template('result.html', label_probs=label_probs, title_color=title_color, title_token=title_token, body_color=body_color, body_token=body_token)
    else:
        return render_template("main.html")

if __name__ == "__main__":
    app.run(debug=True)