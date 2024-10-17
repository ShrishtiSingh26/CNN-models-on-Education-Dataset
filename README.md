# CNN-models-on-Education-Dataset (Task 1)
# Student Performance Prediction using Convolutional Neural Networks (CNNs)

## Project Overview

This machine learning project leverages Convolutional Neural Networks (CNNs) to predict student performance based on various factors such as teacher quality, parental education, and distance from home. The model uses these factors to predict a normalized score representing a student's exam performance.

Two models were developed and trained for this purpose:
1. **Model 1:** A CNN architecture with three convolutional blocks, batch normalization, max pooling, and a dense layer for final predictions.
2. **Model 2:** A simpler CNN architecture with fewer convolutional filters, which still delivers good performance.

The project aims to help educators and institutions identify key factors influencing student success and assist in better decision-making for personalized educational interventions.

### Problem Statement

Student performance is influenced by multiple factors, many of which are non-linear and interdependent. Traditional statistical models fail to capture the complex relationships among these factors. This project applies deep learning techniques to build an intelligent system that can accurately predict student exam scores, helping stakeholders improve learning outcomes and interventions.

### Dataset

The dataset used in this project contains student-related features such as:
- **Teacher_Quality**
- **Parental_Education_Level**
- **Distance_from_Home**
- **Other academic and socio-economic indicators**

The target variable is `Exam_Score`, which is normalized between 0 and 1 using `MinMaxScaler`.

### Models

Two models are developed:
- **Model 1:** Uses three Conv1D layers with different filter sizes (128, 256, and 512), batch normalization, max pooling, dense layers, and dropout regularization.
- **Model 2:** A lighter CNN model with Conv1D layers having smaller filters (64, 128, and 256) and fewer parameters, making it faster but still effective.

Both models are trained using the Adam optimizer, with early stopping and learning rate reduction to prevent overfitting.

## Results

After training and evaluation, the performance of both models was measured using accuracy, precision, and F1-score:

- **Model 1:**
  - Accuracy: 58.02%
  - Precision: 60.03%
  - F1 Score: 57.50%
  
- **Model 2:**
  - Accuracy: 60.29%
  - Precision: 61.24%
  - F1 Score: 59.64%

## Usage and Installation

### Prerequisites

To run this project locally, you'll need the following:
- Python 3.7+
- Required libraries: `tensorflow`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`

Install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Instructions to Clone this repository:
```bash
git clone https://github.com/yourusername/StudentPerformancePrediction.git
```
Navigate to the project directory:

```bash
cd StudentPerformancePrediction
```

Add the dataset `StudentPerformanceFactors.csv` to the project folder.

Run the script to preprocess the data, train the models, and evaluate the performance:

```bash

python main.py
```

## Application and Use Cases
**1. Educational Insights:**
The model can be used by educators to analyze student performance trends and identify key factors that are influencing their learning outcomes. It allows for early intervention and helps in optimizing teaching strategies and parental involvement.

**2. Personalized Learning Systems:**
The model can be integrated into school management systems or e-learning platforms to provide personalized learning recommendations based on predicted performance scores.

**3. Policy-Making in Education:**
The insights from this model can be valuable for policymakers in identifying systemic issues, thereby aiding in the design of better education policies.

## Model Integration for Web Developers
Here’s how a web developer can integrate this trained model into a web project:

**1. Exporting the Model:**
After training, the model can be saved in `.h5` format:

```bash
model_1.save('student_performance_model.h5')
```
Similarly, save Model 2 if desired:
```bash
model_2.save('student_performance_model_2.h5')
```
**2. Flask/Django API:**
A common approach is to create a RESTful API using Flask or Django to serve the model:

**Flask Example:**

```bash
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('student_performance_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON input
    input_data = np.array(data['input']).reshape(1, -1, 1)  # Reshape appropriately
    prediction = model.predict(input_data)
    result = np.argmax(prediction)  # For multi-class classification
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
```
In the frontend, you can use an AJAX request or fetch API in JavaScript to send the data to the server and receive the prediction.

**3. Front-End Integration:**
A web developer can collect input from users (e.g., parent education level, teacher quality, etc.) via an HTML form. This input can then be processed and sent to the Flask/Django API for prediction. The returned prediction can be displayed to users on the web interface.

## Future Work
Feature Engineering: Explore more sophisticated feature engineering techniques to improve model accuracy.
Explainability: Implement methods like SHAP or LIME to make the model’s predictions more interpretable for non-technical stakeholders.
Real-time Predictions: Deploy the model on cloud infrastructure (e.g., AWS, GCP) for real-time student performance predictions in educational platforms.
