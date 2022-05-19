import json
from module import *
from flask import Flask, request, render_template
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)

@app.route("/")
def main():
    return render_template("home.html")

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
        ## 여기부터 스코어 바 플랏 (모듈화 가능할지는 모르겠네요)
        _labels = ['Real News', 'Fake News']
        probs = np.zeros(2)
        probs[1] = label_probs[0][1]
        probs[0] = 1 - label_probs[0][1]

        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_title("result")
        axis.set_ylabel("scores")
        axis.set_xticks(np.arange(len(_labels)), _labels)
        axis.grid()
        axis.bar(np.arange(len(_labels)), probs.squeeze(), align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])

        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)

        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        ## 여기까지 스코어 바 플랏 이미지 생성

        return render_template('result.html', label_probs=label_probs, title_color=title_color, title_token=title_token, body_color=body_color, body_token=body_token, image=pngImageB64String)
    # else:
    #     return render_template("main.html")





@app.route("/mysuperplot", methods=["GET"])
def plotView():
    # Generate plot


    return render_template("image.html", image=pngImageB64String)


if __name__ == "__main__":
    app.run(debug=True)