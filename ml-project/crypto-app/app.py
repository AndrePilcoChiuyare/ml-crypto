from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from services.model import Model

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    body = request.json
    category = body["category"]
    days_to_predict = int(body["days_to_predict"])
    model_name = body["model"]

    model = Model()
    model.train_predict(category=category, days_to_predict=days_to_predict, model=model_name)
    predictions = model.complete_time_series

    return predictions

@app.route("/predict-all", methods=["POST"])
def predict_all():
    body = request.json
    days_to_predict = int(body["days_to_predict"])
    model_name = body["model"]

    categories = ["ai", "meme", "rwa", "gaming"]
    for category in categories:
        model = Model()
        model.train_predict(category=category, days_to_predict=days_to_predict, model=model_name)
    
    return "Predictions completed"

@app.route("/predictionsComplete/<category>", methods=["GET"])
def predict_get(category):
    model = Model()
    predictions = model.load_complete_time_series(category)

    return predictions

@app.route("/predictions-basic/<category>", methods=["GET"])
def get_basic_info(category):
    model = Model()
    basic_info = model.get_basic_prediction_info(category)

    return basic_info

@app.route("/categories/<category>/tokens/<id>", methods=["GET"])
def get_prediction_by_id(category, id):
    model = Model()
    prediction = model.get_prediction_by_id(category=category, token_id=id)

    return prediction

@app.route("/get-data", methods=["POST"])
def get_data():
    model = Model()
    data = model.getData()

    return "Data retrieved"

@app.route("/get-data/predict-all", methods=["POST"])
def get_data_predict_all():
    body = request.json
    days_to_predict = int(body["days_to_predict"])
    model_name = body["model"]

    model = Model()
    data = model.getData()

    categories = ["ai", "meme", "rwa", "gaming"]
    for category in categories:
        model = Model()
        model.train_predict(category=category, days_to_predict=days_to_predict, model=model_name)
    
    return "Predictions completed"

@app.route("/last-date/<category>", methods=["GET"])
def get_last_date(category):
    model = Model()
    last_date = model.get_last_date(category)

    return last_date
    
if __name__ == '__main__':
    app.run(debug=True)
    