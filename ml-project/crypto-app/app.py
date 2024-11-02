from flask import Flask, request, jsonify
import json
from services.model import Model

app = Flask(__name__)

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

@app.route("/prediction/<category>", methods=["GET"])
def predict_get(category):
    model = Model()
    predictions = model.load_complete_time_series(category)

    return predictions
    
if __name__ == '__main__':
    app.run(debug=True)
    