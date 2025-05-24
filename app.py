from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return "Email Spam Detection API is Live"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text']
    vect_text = cv.transform([text]).toarray()
    prediction = model.predict(vect_text)
    return jsonify({'spam': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
