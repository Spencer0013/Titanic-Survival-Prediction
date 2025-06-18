# 🚢 Titanic Survival Prediction

![Titanic Banner](https://ychef.files.bbci.co.uk/624x351/p0hqh71l.jpg)

## 🔗 Deployment
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://drive.google.com/file/d/1yerVKIuXvvu2JPkuQNGY9H2HAPlrK9bH/view?usp=sharing) 

# Overview
This end-to-end machine learning project predicts Titanic passenger survival using a comprehensive pipeline from data ingestion to model deployment. The solution features:

- Automated pipeline with data ingestion, transformation, and model training

- Feature engineering extracting titles and ticket prefixes from raw data

- Hyperparameter tuning using GridSearchCV across 8 classifiers

- Interactive web interface built with Streamlit

- Industry-standard architecture with modular components

- CI/CD Pipeline with GitHub Actions for automated Docker builds

- Azure deployment for production hosting


## Features

# 🛠️ Data Transformation

- Title extraction from passenger names (Mr, Mrs, Dr, etc.)

- Ticket prefix parsing from complex ticket numbers

- Family size calculation combining SibSp and Parch

# Automated preprocessing:

- Median imputation for numerical features

- One-hot encoding for categorical features

- Standard scaling for all features


## 🤖 Model Training
# 8 Classifiers evaluated:

- Random Forest

- XGBoost

- CatBoost

- Logistic Regression

- Gradient Boosting

- AdaBoost

- Decision Trees

- K-Nearest Neighbors

- Hyperparameter tuning with GridSearchCV

- Best model selection based on test accuracy

- Artifact persistence (preprocessor and model)


## 🖥️ Streamlit Web App
- Train/retrain pipeline with one click

- Interactive prediction interface with input controls

- Survival probability display with visual feedback

- Artifact management for reproducible results

## ☁️ Azure Deployment
- Docker containerization for consistent environments

- GitHub Actions CI/CD for automated builds

- Azure Container Registry for image storage

- Azure Web App for production hosting


## Setup

# Clone repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

# Install dependencies
pip install -r requirements.txt

# Install local package
pip install -e .

# Run locally
streamlit run app.py

## Deployment

# Azure Deployment via GitHub Actions

The project includes a CI/CD pipeline that automatically builds and deploys to Azure:

On push to main branch:

- Docker image is built using the Dockerfile

- Image is pushed to Azure Container Registry

- Application is deployed to Azure Web App

# Pipeline Steps:

name: Build and deploy container app to Azure Web App - testsurvival

jobs:
  build:
    runs-on: 'ubuntu-latest'
    steps:
    - Docker setup and build
    - Login to Azure Container Registry
    - Push container image to registry

  deploy:
    runs-on: ubuntu-latest
    needs: build
    steps:
    - Deploy to Azure Web App using publish profile

## Results

The model achieves ~82-85% accuracy on test data based on Titanic passenger characteristics. Key influential features:

- Passenger gender

- Ticket class (Pclass)

- Age

- Family size

- Extracted title (Mr, Mrs, Master)

- Fare paid

# Contribution

Contributions are welcome! Please open an issue first to discuss proposed changes.

# License

This project is licensed under the MIT License - see the LICENSE file for details.