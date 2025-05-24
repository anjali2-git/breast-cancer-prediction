from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

app = Flask(__name__)

# Global variables to store model and scaler
model = None
scaler = None
label_encoder = None
feature_names = None

def train_model():
    global model, scaler, label_encoder, feature_names
    
    try:
        # Read the dataset
        data = pd.read_csv('data.csv')
        
        # Define features and target
        features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                   'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
                   'fractal_dimension_mean']
        target = 'diagnosis'
        
        # Prepare the data
        X = data[features]
        y = data[target]
        
        # Handle categorical target variable
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_scaled, y)
        
        # Save the model, scaler, and label encoder
        joblib.dump(model, 'model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        joblib.dump(le, 'label_encoder.joblib')
        joblib.dump(features, 'features.joblib')
        
        return True
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

def load_model():
    global model, scaler, label_encoder, feature_names
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        feature_names = joblib.load('features.joblib')
        return True
    except:
        return train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.get_json()
        
        # Load the features list
        features = joblib.load('features.joblib')
        
        # Create input array with the same order as features
        input_array = np.array([[input_data[feature] for feature in features]])
        
        # Load the model and scaler
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Scale the input data
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get probability and ensure it's between 5% and 95%
        prob = model.predict_proba(input_scaled)[0][1]  # Probability of malignant class
        prob_percentage = min(95, max(5, round(prob * 100, 2)))
        
        # Convert prediction to label
        prediction_label = 'M' if prediction == 1 else 'B'
        
        return jsonify({
            'prediction': prediction_label,
            'probability': prob_percentage
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if load_model():
        print("Model loaded successfully")
        app.run(debug=True)
    else:
        print("Failed to load or train model")
