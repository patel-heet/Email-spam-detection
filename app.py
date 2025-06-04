from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app, origins=["https://mail.google.com"])

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

model = joblib.load("spam_classifier_model.pkl")
cv = joblib.load("count_vectorizer.pkl")

@app.before_request
def restrict_to_gmail():
    origin = request.headers.get('Origin') or request.headers.get('Referer')
    if origin and "mail.google.com" not in origin:
        return jsonify({"error": "Unauthorized origin"}), 403

# Main prediction route
@app.route('/predict', methods=['POST'])
@limiter.limit("20 per minute")
def predict():
    data = request.get_json()
    message = data.get("message", "")
    print("📨 Received:", message)

    if not message.strip():
        return jsonify({"error": "No email text provided"}), 400

    try:
        vectorized = cv.transform([message])
        prediction = model.predict(vectorized)[0]
        print("📈 Prediction:", prediction)
        return jsonify({"spam": bool(prediction)})
    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
