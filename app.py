from flask_cors import CORSAdd commentMore actions
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and vectorizer once at startup
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")
cv = joblib.load("count_vectorizer.pkl")  # use same name 'cv' throughout

# Test route to check if prediction works
@app.route('/test', methods=['GET'])
def test():
    sample = "You are missing out on your prize. Click here to claim you money"
    sample = "You are missing out on your prize. Click here to claim your money"
    vect = cv.transform([sample])
    pred = model.predict(vect)[0]
    return jsonify({"spam": bool(pred)})


# Main prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    print("📨 Received:", message)

    if not data or 'text' not in data:
    if not message.strip():
        return jsonify({"error": "No email text provided"}), 400

    email_text = data['text']
    transformed_text = vectorizer.transform([email_text])
    prediction = model.predict(transformed_text)
    try:
        vectorized = cv.transform([message])
        prediction = model.predict(vectorized)[0]
        print("📈 Prediction:", prediction)
        return jsonify({"spam": bool(prediction)})
    except Exception as e:
        print("❌ Error in prediction:", str(e))
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

    return jsonify({
        "spam": bool(prediction[0])  # Convert numpy int to bool
    })
# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
