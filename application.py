import joblib
import numpy as np
import flask
import datetime
from flask import Flask, render_template, request,send_file
from flask_cors import CORS,cross_origin
from src.training_Validation import train_validation
from src import columns
import os
import flask_monitoringdashboard as dashboard
from src.bulkPredection import bulkPredict
import pandas as pd

application = Flask(__name__)
dashboard.bind(application)

def valuePredictor(to_predict_list):
    with open('Logs/PredectionLogs.txt', 'a') as f:
        try:
            to_predict = np.array(to_predict_list).reshape(1, 54)
            loaded_model = joblib.load("model/model.sav")
            result = loaded_model.predict(to_predict)
            return result[0]
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))

def bulkPredictor(df):
    with open('Logs/PredectionLogs.txt', 'a') as f:
        try:
            loaded_model = joblib.load("model/model.sav")
            result = loaded_model.predict(df)
            return result
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))

@application.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html", prediction="default")

@application.route('/bulk',methods=['GET','POST'])
def bulk():
    with open('Logs/PredectionLogs.txt', 'a') as f:
        try:
            if request.method == "POST":
                file = request.files["file"]
                file.save(os.path.join("bulk_pred", file.filename))
                data,val = bulkPredict(file.filename)
                if val!=True:
                    return render_template("index2.html", message=data)
                result = bulkPredictor(data)
                df = pd.read_csv(os.path.join("bulk_pred", file.filename))
                df['Credit Risk'] = result
                df.to_csv(os.path.join("bulk_pred", file.filename))
                return send_file(os.path.join("bulk_pred", file.filename), as_attachment=True, attachment_filename='predection.csv')
            return render_template("index2.html",message="")
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))

@application.route('/training', methods=['GET', 'POST'])  # route to train the model
def train():
    with open('Logs/PredectionLogs.txt', 'a') as f:
        try:
            if request.method == "POST":
                file = request.files["file"]
                file.save(os.path.join("data", file.filename))
                return flask.Response(train_validation(file.filename), mimetype='text/html')
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
    return render_template("upload.html", message="Upload asc File")



@application.route('/result', methods=['POST'])
def result():
    with open('Logs/PredectionLogs.txt', 'a') as f:
        try:
            data = [0] * 54
            if request.method == 'POST':
                to_predict_list = request.form.to_dict()
                to_predict_list = list(to_predict_list.values())
                data[0] = np.log1p(int(to_predict_list[0]))
                data[1] = np.log1p(int(to_predict_list[1]))
                data[2] = np.log1p(int(to_predict_list[2]))
                for i in to_predict_list:
                    if i in columns.col:
                        data[columns.col.index(i)] = 1
                result = valuePredictor(data)
                f.write(str(datetime.datetime.now()) + ' Predectied [class: {}]\n'.format(result))
            return render_template("index.html", prediction=int(result))
        except Exception as e:
            f.write(str(datetime.datetime.now()) + ' {}\n'.format(e))
            

if __name__ == "__main__":
    application.run('0.0.0.0',debug=True)
