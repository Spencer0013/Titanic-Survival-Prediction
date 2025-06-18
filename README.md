# Titanic Survival Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-orange)

## Overview
An end-to-end machine learning solution that predicts passenger survival probabilities from the Titanic disaster. The project demonstrates full ML pipeline implementation from data preprocessing to web application deployment.

## Key Features
- **CatBoost implementation**: Gradient boosting algorithm for classification
- **Feature engineering**: Custom preprocessing pipeline
- **Dockerized deployment**: Containerized application
- **CI/CD pipeline**: Automated testing and deployment
- **Streamlit UI**: Interactive web interface


## Technical Implementation

### 1. Data Processing
- Comprehensive feature engineering (`feature_engineering.py`)
- Missing value handling (Age, Cabin, Embarked)
- Title extraction from passenger names
- Family size calculation
- Categorical feature encoding

### 2. Modeling
- **Algorithm**: CatBoostClassifier
- **Validation**: Cross-validation
- **Feature importance**: Native CatBoost feature ranking

### 3. Deployment
- **Web framework**: Streamlit
- **Containerization**: Docker
- **Cloud deployment**: Azure via GitHub Actions CI/CD
- **Version control**: Model and code versioning

## Setup Instructions

### Local Installation

git clone https://github.com/Spencer0013/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
pip install -r requirements.txt
streamlit run app.py


## Docker Deployment 
docker build -t titanic-predictor .
docker run -p 8501:8501 titanic-predictor

# Usage
The web application allows users to input passenger details and receive survival predictions through an intuitive interface.

# CI/CD Pipeline
- Automated Docker builds on push to main branch

- Azure Container Registry integration

- Continuous deployment to Azure App Service

# Development Roadmap

- Enhance test coverage

- Add monitoring capabilities

- Implement model version comparison





