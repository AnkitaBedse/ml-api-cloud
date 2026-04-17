from flask import Flask, request, jsonify
import joblib
import os

# IMPORTANT: must be named 'application' for AWS
application = Flask(__name__)

# Load model
model = joblib.load('sentiment_model.joblib')

@application.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    prediction = model.predict([text])[0]

    return jsonify({
        'input_text': text,
        'sentiment_prediction': prediction
    })

# Local run (also works for AWS with PORT)
if __name__ == '__main__':
    application.run(
        host='0.0.0.0',
        port=int(os.environ.get("PORT", 5000))
    )