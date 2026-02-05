# Heart Disease Prediction API

## Project Overview

This project is a machine learning–based heart disease prediction system.
It trains a classification model on medical data and exposes the model through a REST API using FastAPI.

The system predicts the probability of heart disease based on patient health metrics.

---

## Features

* Trained machine learning models (Logistic Regression, Random Forest)
* Reduced false negatives using threshold tuning
* Model evaluation using:

  * Confusion Matrix
  * Precision, Recall, F1-score
  * ROC-AUC
* REST API deployment with FastAPI
* Interactive Swagger UI for testing

---

## Dataset

The model is trained on a heart disease dataset containing features such as:

* Age
* Sex
* Chest pain type (cp)
* Blood pressure (trestbps)
* Cholesterol (chol)
* Maximum heart rate (thalach)
* Exercise-induced angina (exang)
* And other clinical indicators

Target:

* `0` → No heart disease
* `1` → Heart disease risk

---

## Model Strategy

In medical applications, **false negatives (FN)** are more dangerous than false positives.

Therefore:

* Model threshold was reduced to **0.3**
* This increases recall for patients with heart disease
* Reduces the chance of missing risky patients

---

## Tech Stack

* Python
* Scikit-learn
* FastAPI
* Joblib
* Uvicorn

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

### 3. Start the API

```bash
uvicorn app:app --reload
```

### 4. Open Swagger UI

Go to:

```
http://127.0.0.1:8000/docs
```

---

## Example API Request

POST `/predict`

```json
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 140,
  "chol": 240,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

---

## Output

```json
{
  "heart_disease_probability": 0.777,
  "prediction": 1,
  "threshold": 0.3
}
```

---

## Author

Computer Engineering student focusing on:

* Machine Learning
* Backend development
* ML-powered applications
