# AI-Powered Credit Risk Intelligence Platform

## Project Overview

This project is an end-to-end Credit Risk Intelligence Platform built using Machine Learning, Explainable AI, Generative AI, and Docker.

The platform predicts the probability of loan default, explains model decisions using SHAP, applies business risk rules, and provides a Gemini-powered Talk-to-Data chatbot for natural language analysis of credit risk data.

---

## Features

### Credit Risk Prediction

* LightGBM-based classification model
* Predicts loan default probability
* Risk categorization (Low, Medium, High)

### Explainable AI

* SHAP feature importance visualization
* Model interpretability for decision making

### Business Rules Engine

* Rule-based risk assessment
* Complements machine learning predictions

### Talk-to-Data Chatbot

* Powered by Google Gemini
* Converts natural language questions into SQL queries
* Retrieves insights from the credit risk database

### Interactive Dashboard

* Built using Streamlit
* User-friendly interface for risk analysis

### Docker Deployment

* Fully containerized application
* Consistent deployment across environments

---

## Dataset

Home Credit Default Risk Dataset

Main dataset:

* application_train.csv

Total records:

* 307,511 loan applications

---

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 71.46% |
| Precision | 17.32% |
| Recall    | 67.17% |
| F1 Score  | 27.53% |
| ROC-AUC   | 76.13% |

---

## Project Structure

credit_risk_platform/

├── app.py

├── Dockerfile

├── requirements.txt

├── data/

├── models/

├── outputs/

├── src/

└── README.md

---

## Installation

### Clone Repository

git clone <repository-url>

cd credit_risk_platform

### Install Dependencies

pip install -r requirements.txt

### Run Application

streamlit run app.py

---

## Docker Deployment

Build Docker Image

docker build -t credit-risk-platform .

Run Docker Container

docker run -p 8501:8501 credit-risk-platform

Open:

http://localhost:8501

---

## Technologies Used

* Python
* Streamlit
* LightGBM
* SHAP
* Pandas
* SQLite
* Google Gemini API
* Docker

---

## Author

Ramya
