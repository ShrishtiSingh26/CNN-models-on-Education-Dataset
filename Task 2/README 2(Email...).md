# email-spam-detection

# Spam Detection using Machine Learning

## Project Overview

This project implements a spam detection system using machine learning techniques to classify email messages as either spam or ham (not spam). The model is trained on a dataset of emails, utilizing natural language processing (NLP) for feature extraction and logistic regression for classification.

## Problem Statement

With the increasing volume of emails received daily, distinguishing between spam and legitimate messages is crucial for maintaining productivity and security. This project aims to automate the classification process, helping users filter out unwanted spam emails efficiently.

## Dataset

The dataset used for this project is a CSV file containing email messages labeled as either "spam" or "ham." The data is processed to handle missing values, and the categories are converted to numerical values for model training.

## How It Works

1. **Data Loading**: The email data is loaded from a CSV file.
2. **Data Preprocessing**: 
   - Null values are replaced.
   - The 'Category' labels are encoded (spam as `0` and ham as `1`).
3. **Train-Test Split**: The data is split into training (80%) and testing (20%) sets.
4. **Feature Extraction**: 
   - The `TfidfVectorizer` is used to convert the text messages into numerical feature vectors.
5. **Model Training**: A logistic regression model is trained using the feature vectors from the training set.
6. **Model Evaluation**: The model's accuracy is assessed on both training and testing datasets.
7. **Prediction**: The trained model can predict whether a new email message is spam or ham.

![security-email_spam_mobile](https://github.com/ShrishtiSingh26/email-spam-detection-/assets/142707684/5b1357b2-0316-4463-af58-303be27b1957)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-detection.git
   cd spam-detection
2. Install the required libraries:
```bash
pip install numpy pandas scikit-learn
  ```
3.Place your dataset `mail_data.csv` in the project directory.

##Usage
To run the spam detection model, execute the following command in your terminal or command prompt:
```bash
python spam_detection.py
```
You can modify the `input_mail` variable in the script to test with different email messages.

## Model Accuracy
The model gives upto *96.70%* accuracy over training data and *96.59%* accuracy score for testing data.

## Integrating the Model into a Web Project
To integrate this spam detection model into a web project, follow these steps:

Export the Model: After training, you can save the model using joblib or pickle:

```bash
import joblib
joblib.dump(model, 'spam_detection_model.pkl')
Create a Flask/Django Web API: Set up a simple web server using Flask or Django to handle incoming email messages.
```
Load the Model in Your API:

```bash
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('spam_detection_model.pkl')
feature_extraction = joblib.load('tfidf_vectorizer.pkl')  # Save your vectorizer similarly

@app.route('/predict', methods=['POST'])
def predict():
    input_mail = request.json['message']
    input_data_features = feature_extraction.transform([input_mail])
    prediction = model.predict(input_data_features)
    return jsonify({'prediction': 'Ham mail' if prediction[0] == 1 else 'Spam mail'})
```
**Run Your Web Server:**

```bash
flask run
Testing the API: Use tools like Postman to send POST requests with email messages to your API and receive predictions.
```
## Conclusion

Spam email detection is a crucial task in email security, as it helps users avoid unsolicited and potentially harmful emails. In this project, we developed a machine learning model to automatically classify emails as spam or legitimate (ham). By leveraging a labelled dataset and employing preprocessing techniques like text cleaning and numerical representation, we trained and evaluated several machine learning algorithms. Despite the inherent challenges such as dataset quality and the presence of false positives/negatives, our model demonstrated promising performance. Through optimization and fine-tuning, we were able to achieve better results, albeit with some limitations. Overall, this project underscores the significance of machine learning in combating spam emails and highlights the continuous effort required to enhance email security and user experience.
