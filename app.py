from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model from the file
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    name = data['cName']
    age = int(data['cAge'])
    gender = data['cGender']
    bmi = float(data['cBmi'])
    children = int(data['cChildren'])
    is_smoker = data['cIsSmoker']
    
    # Encode gender and is_smoker
    gender_encoded = 1 if gender == 'Male' else 0
    is_smoker_encoded = 1 if is_smoker == 'Yes' else 0
    
    # Prepare input data for prediction
    input_data = np.array([[age, gender_encoded, bmi, children, is_smoker_encoded]])
    
    # Make prediction
    prediction = int(model.predict(input_data))
    #predicted_class = tf.argmax(prediction[0]).numpy()
    
    result = f" Price of Insurance of {name} is RS: {prediction}"
    
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
