from flask import Flask, request, jsonify, send_from_directory
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

history = []

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    prediction = model.predict([text])[0]
    score = 0.8  # replace with real probability if available

    result = {
        "label": "Positive" if prediction == 1 else "Negative",
        "score": float(score),
        "text": text
    }

    history.append(result)

    return jsonify(result)

@app.route("/history")
def get_history():
    return jsonify(history)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
