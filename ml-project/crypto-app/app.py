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

@app.route("/predictions", methods=["GET"])
def get_prediction_by_id():
    body = request.json
    category = body["category"]
    token_id = body["id"]

    model = Model()
    prediction = model.get_prediction_by_id(category=category, token_id=token_id)

    return prediction
    
if __name__ == '__main__':
    app.run(debug=True)
    