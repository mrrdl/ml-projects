from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('LinearRegModel.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        brand = request.form['brand']
        fuel_type = request.form['fuel_type']
        kms_driven = float(request.form['kms_driven'])

        # You need to preprocess the input data according to the preprocessing done before training the model

        # Example preprocessing:
        # Convert categorical variables to numerical using one-hot encoding or label encoding
        # Scale numerical variables if necessary

        # Make prediction
        prediction = model.predict([[name, brand, fuel_type, kms_driven]])

        # You may need to post-process the prediction if necessary

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
