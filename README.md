# Heart Disease Prediction API

## Project Overview

This project is a machine learning–based heart disease prediction system.
It trains classification models on medical data and exposes the best-performing model through a REST API using FastAPI.

The system predicts the probability of heart disease based on patient health metrics and is designed with a focus on **reducing false negatives**, which is critical in medical applications.

---

## Features

* Trained machine learning models (Logistic Regression, Random Forest)
* Threshold tuning to reduce false negatives
* Model evaluation using:

  * Confusion Matrix
  * Precision, Recall, F1-score
  * Accuracy
* REST API deployment with FastAPI
* Interactive Swagger UI for testing

---

## Dataset

The model is trained on a heart disease dataset containing features such as:

* Age
* Sex
* Chest pain type (cp)
* Resting blood pressure (trestbps)
* Cholesterol (chol)
* Maximum heart rate (thalach)
* Exercise-induced angina (exang)
* ST depression (oldpeak)
* Number of major vessels (ca)
* Thalassemia (thal)

**Target:**

* `0` → No heart disease
* `1` → Heart disease risk

---

## Model Strategy

In medical prediction systems, **false negatives (FN)** are more dangerous than false positives because:

* A false negative means a sick patient is classified as healthy.
* This may delay diagnosis and treatment.

Therefore:

* The decision threshold was reduced to **0.3**.
* This increases recall for patients with heart disease.
* Reduces the chance of missing high-risk patients.

---

## Model Performance

**Random Forest (Test Set Results):**

* Accuracy: **0.85**
* Precision (heart disease): **0.81**
* Recall (heart disease): **0.92**
* F1-score: **0.86**

This configuration prioritizes **high recall** to minimize missed diagnoses.

---

## Tech Stack

* Python
* Scikit-learn
* FastAPI
* Joblib
* Uvicorn

---

## Project Structure

```
heart-disease-api/
│
├── app.py            # FastAPI application
├── train.py          # Model training script
├── model.pkl         # Trained model
├── requirements.txt
└── README.md
```

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

**POST** `/predict`

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

## Example Output

```json
{
  "heart_disease_probability": 0.777,
  "prediction": 1,
  "threshold": 0.3
}
```

---

## Future Improvements

* Hyperparameter tuning
* Cross-validation for more robust evaluation
* Model versioning
* Docker containerization
* Cloud deployment (AWS, GCP, or Render)

---

## Author

Umut Seyhan
Computer Engineering Student
Interested in Machine Learning, Mobile Development, and Backend Systems
