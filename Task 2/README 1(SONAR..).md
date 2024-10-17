# SONAR-Rock-vs-Mine-Prediction


# Sonar Data Classification

This project demonstrates the classification of sonar signals using various machine learning models. The dataset used is the **Sonar Data** which contains 60 features of sonar returns and a target variable indicating whether the sonar return is from a metal cylinder (M) or a rock (R). We apply different classification algorithms and evaluate their performance based on accuracy.


## Overview
This project implements a machine learning model to classify sonar signals into two categories: Rock (R) and Mine (M). The dataset consists of sonar signals reflected off the sea floor and has been widely used for training classification algorithms.

## Problem Statement
In underwater exploration, distinguishing between rocks and potential mines is crucial for safe navigation and exploration. This project leverages machine learning algorithms to automate the classification process, enhancing the efficiency and accuracy of sonar data interpretation.

## Dataset

- The dataset `sonar_data.csv` consists of 60 features representing sonar returns.
- The target variable (at index 60) represents the class label: **R** for rock and **M** for metal.


### Features
- 60 numeric attributes representing sonar signal features.
- Target variable indicating the classification (Rock or Mine).

## Application
This model can be applied in various domains, including:
- **Marine Navigation**: Helping vessels identify hazardous underwater obstacles.
- **Underwater Mining**: Assisting in the exploration of mineral resources while ensuring safety.
- **Environmental Monitoring**: Providing data for research on underwater ecosystems.

## Project Structure

```plaintext
.
├── sonar_data.csv          # Dataset file
├── sonar_minerock.py  # Python script for training models
└── README.md               # Project documentation
```

## Dependencies

Make sure you have the following Python libraries installed before running the script:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Steps

1. **Loading the Dataset:**
   - The dataset is loaded from the CSV file and basic information about the dataset is displayed, such as its shape, column names, and statistical summary.
   - The target variable is plotted to observe the distribution of the two classes.

2. **Data Preprocessing:**
   - Features (`X`) and target variable (`y`) are separated.
   - The data is split into training (80%) and testing (20%) sets using `train_test_split`.

3. **Model Training:**
   We train four different classification models:

   - **Logistic Regression:** A basic linear model for binary classification.
   - **K-Nearest Neighbors (KNN):** A distance-based classification algorithm using 3 neighbors.
   - **Random Forest Classifier:** An ensemble learning method using multiple decision trees.
   - **Stochastic Gradient Descent (SGD):** A gradient-based optimization algorithm suitable for large datasets.

4. **Model Evaluation:**
   - The accuracy of each model is evaluated on the test set.
   - **Accuracy Scores:**
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Random Forest Classifier
     - Stochastic Gradient Descent (SGD)

5. **Partial Fit for SGD:**
   - The `partial_fit` method is used to incrementally train the SGD model in a loop over the training dataset.
## Model Accuracy
**Logistic Regression Accuracy:** 0.7619047619047619

**KNN Accuracy:** 0.7857142857142857

**Random Forest Accuracy:** 0.8333333333333334

**SGD Classifier Accuracy:** 0.5714285714285714

## Usage

1. Clone this repository and navigate to the project directory.

2. Run the Python script to train the models and evaluate their accuracy:

```bash
python sonar_classification.py
```

3. The script will display the accuracy of each model in the terminal.

## Results

The accuracy of the following models is printed at the end of the script:

- **Logistic Regression Accuracy**
- **K-Nearest Neighbors (KNN) Accuracy**
- **Random Forest Accuracy**
- **SGD Classifier Accuracy**

## Future Work

- Experiment with hyperparameter tuning to improve model performance.
- Explore additional classification models, such as Support Vector Machines (SVM) or Neural Networks.
- Implement feature selection techniques to improve model accuracy and reduce overfitting.

##Integration into a Web Project
To integrate the trained model into a web project, follow these steps:

Model Serialization: Use joblib or pickle to save the trained model for deployment:

```bash
import joblib
joblib.dump(lr, 'sonar_model.pkl')  # Save the Logistic Regression model
API Creation: Create a REST API using Flask or FastAPI to serve the model. Below is a simple example using Flask:
```
```bash
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('sonar_model.pkl')  # Load the trained model

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Expecting input data in JSON format
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
```

**Front-End Integration:** A front-end developer can create a form to accept sonar signal data input, which will then send a POST request to the `/predict` endpoint to retrieve predictions. The predictions can be displayed on the web page for user interaction.


