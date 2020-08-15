import joblib
import numpy as np
import flask
from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
from src.training_Validation import train_validation
from src import columns
import os

application = Flask(__name__)

def valuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 53)
    loaded_model = joblib.load("model/model.sav")
    result = loaded_model.predict(to_predict)
    return result[0]


@application.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@application.route('/training', methods=['GET', 'POST'])  # route to train the model
def train():
    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.join("data", file.filename))
        return flask.Response(train_validation('data/'+file.filename), mimetype='text/html')
    return render_template("upload.html", message="Upload")


@application.route('/result', methods=['POST'])
def result():
    data = [0] * 53
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        data[0] = np.log1p(int(to_predict_list[0]))
        data[1] = np.log1p(int(to_predict_list[1]))
        for i in to_predict_list:
            if i in columns.col:
                data[columns.col.index(i)] = 1
        result = valuePredictor(data)
        if int(result) == 1:
            prediction = 'Credit Risk is Good !!!'
        else:
            prediction = 'Credit Risk is Bad !!!'
    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    application.run(debug=True)
