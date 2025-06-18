<<<<<<< HEAD
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





=======
## Titanic Survival Prediction Project

This production-ready machine learning solution addresses the  Kaggle Titanic competition, demonstrating a complete ML Ops implementation from raw data ingestion to deployment-ready prediction interface. The system delivers 83.2% accuracy in passenger survival prediction through optimized feature engineering and model selection, providing a template for enterprise-grade binary classification systems in risk assessment, customer outcome prediction, and resource prioritization scenarios.

## 🧩 Project Overview
This project implements a complete ML pipeline for predicting Titanic passenger survival. The system:

Ingests and preprocesses passenger data

Performs feature engineering and transformation

Trains multiple classifiers with hyperparameter tuning

Provides an interactive web interface for predictions

Automatically deploys to Azure via GitHub Actions

## Key Features

End-to-end ML pipeline from data ingestion to prediction

Streamlit-based web application

Azure container deployment with CI/CD

Hyperparameter optimization with GridSearchCV

Feature engineering for passenger data

Model comparison and selection

## Datset

Features:

Passenger demographics (age, sex, class)

Ticket information

Family relationships

Embarkation port

## Feature Engineering:

def extract_title(self, name):
    match = re.search(r',\s*([^\.]+)\.', name)
    return match.group(1).strip() if match else 'Missing'

def extract_ticket_prefix(self, ticket):
    parts = ticket.split()
    if len(parts) > 1:
        return parts[0].replace('.', '').replace('/', '').upper()
    else:
        return 'None'

## Preprocessing:

Family size calculation (SibSp + Parch)

Title extraction from names

Ticket prefix extraction

Handling missing values

One-hot encoding of categorical features

Standard scaling of numerical features

## ⚙️ ML Pipeline

# 1. Data Ingestion

class DataIngestion:
    def initiate_data_ingestion(self):
        df = pd.read_csv('Notebook/Data/train.csv')
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        train_set.to_csv(self.ingestion_config.train_data_path)
        test_set.to_csv(self.ingestion_config.test_data_path)

# 2. Data Transformation
class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        for df in [train_df, test_df]:
            df['family'] = df['SibSp'] + df['Parch']
            df['ticket_update'] = df['Ticket'].apply(self.extract_ticket_prefix)
            df['title'] = df['Name'].apply(self.extract_title)
        
        preprocessing_obj = ColumnTransformer(transformers=[
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipeline", cat_pipeline, categorical_columns)
        ])

## 3. Model Training

class ModelTrainer:
    def initiate_model_trainer(self, train_array, test_array):
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "XGBClassifier": XGBClassifier(random_state=42),
            "CatBoost Classifier": CatBoostClassifier(verbose=False, random_seed=42),
            "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }
        
        model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

# 🌐 Web Application

# Streamlit app interface
mode = st.sidebar.radio("Mode", ["Train Model", "Predict Passenger"])

if mode == "Train Model":
    if st.button("Run Full Pipeline"):
        preproc_path, model_path, test_acc = run_pipeline()
        st.success(f"✅ Pipeline complete. Test accuracy = {test_acc:.4f}")

elif mode == "Predict Passenger":
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", 0.0, 100.0, 30.0)
    # ... additional inputs ...
    
    if st.button("Predict Survival"):
        features = preprocessor.transform(single_df)
        pred_class = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0][1]
        
        if pred_class == 1:
            st.success(f"✅ Survived (probability = {pred_proba:.2f})")
        else:
            st.error(f"❌ Did Not Survive (probability = {1 - pred_proba:.2f})")

# Features:

- Train/Retrain Pipeline: Full ML pipeline execution

- Passenger Prediction: Interactive survival prediction

- Probability Display: Visual confidence scores

- Feature Engineering: Automatic title and ticket extraction

- Responsive Design: Mobile-friendly interface

# Deployment Architecture:

- GitHub Actions: Triggers on push to main branch

- Azure Container Registry (ACR): Stores Docker images

- Azure Web App: Hosts containerized application

- Production Slot: Zero-downtime deployments

# 🚀 Installation & Execution

Local Development:

# Clone repository
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit application
streamlit run app.py

# Azure Deployment:

Fork repository

Create Azure resources:

Container Registry

Web App for Containers

Configure GitHub Secrets:

AZURE_CR_USERNAME

AZURE_CR_PASSWORD

AZURE_PUBLISH_PROFILE

Push changes to trigger deployment

## 🔍 Model Performance

models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(),
    "XGBClassifier": XGBClassifier(),
    "CatBoost Classifier": CatBoostClassifier(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

## Hyperparameter Tuning:

Each model uses GridSearchCV for optimization

## Evaluation:

Model selection based on test accuracy

Best model saved as artifacts/model.pkl

Accuracy displayed in web interface after training

## 📚 References

Kaggle Titanic Competition

Streamlit Documentation

Azure Web Apps Documentation

Scikit-learn Documentation

GitHub Actions Documentation

>>>>>>> 823399ea5bb0205451f54c6d4cb263071c383786
