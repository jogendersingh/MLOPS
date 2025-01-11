from flask import Flask, request, jsonify, render_template
#import pickle
import joblib
import pandas as pd
import os

# Load the model
MODEL_PATH = "log_model_Salary_prediction"
with open(MODEL_PATH, 'rb') as model_file:
    model = joblib.load(model_file)

# Initialize the Flask app
app = Flask(__name__)

# Set up template directory
app.template_folder = os.path.join(os.path.dirname(__file__), 'templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        years_experience = request.form.get('YearsExperience')
        
        if not years_experience:
            return jsonify({"error": "Missing 'YearsExperience' in request"}), 400

        # Prepare the input data for prediction
        years_experience = float(years_experience)
        input_df = pd.DataFrame([[years_experience]], columns=['YearsExperience'])

        # Perform prediction
        prediction = model.predict(input_df)[0]

        # Return the prediction result
        return render_template('index.html', prediction_text=f'Predicted Salary: ${prediction:.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
