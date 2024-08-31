from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form['data']
        # Convert the input data into the required format
        input_data = np.array([float(i) for i in data.split(',')]).reshape(1, -1)
        # Make prediction
        prediction = model.predict(input_data)
        return render_template('index.html', prediction_text='Fraudulent Transaction: {}'.format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)
