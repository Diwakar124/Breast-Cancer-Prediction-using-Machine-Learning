from flask import Flask, render_template, request
import numpy as np
import pickle

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.form['feature']
        features = features.split(',')
        np_features = np.asarray(features, dtype=np.float32)

        # Prediction
        pred = model.predict(np_features.reshape(1, -1))
        message = 'Cancerous' if pred[0] == 1 else 'Not Cancerous'

        # Optional: confidence (probability)
        if hasattr(model, "predict_proba"):
            confidence = round(np.max(model.predict_proba(np_features.reshape(1, -1))) * 100, 2)
        else:
            confidence = None

        return render_template('index.html', message=message, confidence=confidence, features=features)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
