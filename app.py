from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer once at startup
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "No email text provided"}), 400

    email_text = data['text']
    transformed_text = vectorizer.transform([email_text])
    prediction = model.predict(transformed_text)

    return jsonify({
        "spam": bool(prediction[0])  # Convert numpy int to bool
    })

if __name__ == '__main__':
    app.run(debug=True)
