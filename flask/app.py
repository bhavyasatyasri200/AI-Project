from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        holiday = float(request.form['holiday'])
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = float(request.form['weather'])
        day = float(request.form['day'])
        month = float(request.form['month'])
        year = float(request.form['year'])
        hours = float(request.form['hours'])
        minutes = float(request.form['minutes'])
        seconds = float(request.form['seconds'])

        # Prepare the input array
        input_data = np.array([[holiday, temp, rain, snow, weather, day, month, year, hours, minutes, seconds]])

        # Scale the data
        input_scaled = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_scaled)[0]

        return render_template('output.html', result=f"Predicted Traffic Volume: {int(prediction)}")

    except Exception as e:
        return render_template('output.html', result=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
    
    
