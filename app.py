# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import load_object

# ───────────────────────────────────────────────────────────────────────────────
# 1) STREAMLIT CONFIG & HEADER
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction")
st.write(
    """
    Choose **Train Model** to run ingestion (from `Notebook/Data/train.csv`), 
    transform, and train. Or choose **Predict Passenger** to load saved artifacts 
    from `artifacts/` and predict survival for a single passenger.
    """
)

# ───────────────────────────────────────────────────────────────────────────────
# 2) MODES: TRAIN VS. PREDICT
# ───────────────────────────────────────────────────────────────────────────────

mode = st.sidebar.radio("Mode", ["Train Model", "Predict Passenger"])

# ───────────────────────────────────────────────────────────────────────────────
# 3) MODE 1: RUN THE WHOLE PIPELINE (INGEST → TRANSFORM → TRAIN)
# ───────────────────────────────────────────────────────────────────────────────

@st.cache(allow_output_mutation=True)
def run_pipeline():
    """
    1) Ingest raw CSV from Notebook/Data/train.csv → split into artifacts/train.csv & artifacts/test.csv  
    2) Transform train/test → save artifacts/preprocessor.pkl → return arrays  
    3) Train classifiers with GridSearchCV → save artifacts/model.pkl → return test accuracy  
    """
    # 3.1 Data Ingestion (reads Notebook/Data/train.csv internally)
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 3.2 Data Transformation (saves artifacts/preprocessor.pkl)
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
        train_path, test_path
    )

    # 3.3 Model Training (saves artifacts/model.pkl)
    trainer = ModelTrainer()
    test_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)
    model_path = trainer.model_trainer_config.trained_model_file_path

    return preprocessor_path, model_path, test_accuracy


if mode == "Train Model":
    st.subheader("🚀 Train / Retrain Pipeline")
    st.write(
        """
        - Ingests raw data from `Notebook/Data/train.csv`  
        - Creates and saves `artifacts/train.csv` & `artifacts/test.csv`  
        - Saves `artifacts/preprocessor.pkl` (ColumnTransformer)  
        - Saves `artifacts/model.pkl` (best classifier)  
        - Reports test accuracy  
        """
    )

    if st.button("Run Full Pipeline"):
        with st.spinner("Ingesting → Transforming → Training…"):
            try:
                preproc_path, model_path, test_acc = run_pipeline()
                st.success(f"✅ Pipeline complete. Test accuracy = {test_acc:.4f}")
                st.write(f"- Preprocessor saved at `{preproc_path}`")  
                st.write(f"- Model saved at `{model_path}`")
            except Exception as e:
                st.error(f"Pipeline failed:\n{e}")


# ───────────────────────────────────────────────────────────────────────────────
# 4) MODE 2: PREDICT A SINGLE PASSENGER
# ───────────────────────────────────────────────────────────────────────────────

elif mode == "Predict Passenger":
    st.subheader("🔮 Predict Survival for One Passenger")

    # 4.1 Ensure artifacts exist
    preproc_path = os.path.join("artifacts", "preprocessor.pkl")
    model_path = os.path.join("artifacts", "model.pkl")
    if not (os.path.exists(preproc_path) and os.path.exists(model_path)):
        st.warning("No trained artifacts found. Please switch to **Train Model** first.")
        st.stop()

    # 4.2 Load preprocessor and model
    preprocessor = load_object(preproc_path)
    model = load_object(model_path)

    # 4.3 Collect user inputs (matching features engineered in DataTransformation)
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", 0.0, 100.0, 30.0, 0.5)
    sibsp = st.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.20, 0.01)
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"], index=2)

    name = st.text_input("Full Name (e.g., `Braund, Mr. Owen Harris`)", "Braund, Mr. Owen Harris")
    ticket = st.text_input("Ticket (e.g., `A/5 21171`)", "A/5 21171")

    if st.button("Predict Survival"):
        # 4.4 Feature Engineering (reuse DataTransformation methods)
        dt = DataTransformation()
        family = sibsp + parch
        title = dt.extract_title(name)
        ticket_prefix = dt.extract_ticket_prefix(ticket)

        # 4.5 Build single-row DataFrame exactly as training did
        single_df = pd.DataFrame(
            {
                "Pclass": [pclass],
                "Age": [age],
                "SibSp": [sibsp],
                "Parch": [parch],
                "Fare": [fare],
                "family": [family],
                "title": [title],
                "Sex": [sex],
                "Embarked": [embarked],
                "ticket_update": [ticket_prefix],
            }
        )

        # 4.6 Preprocess
        features = preprocessor.transform(single_df)
        if hasattr(features, "toarray"):
            features = features.toarray()

        # 4.7 Predict & display
        pred_class = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0][1]

        st.markdown("### Prediction Result")
        if pred_class == 1:
            st.success(f"✅ Survived (probability = {pred_proba:.2f})")
        else:
            st.error(f"❌ Did Not Survive (probability = {1 - pred_proba:.2f})")

        proba_df = pd.DataFrame(
            {
                "Did Not Survive (0)": [f"{1 - pred_proba:.2f}"],
                "Survived (1)": [f"{pred_proba:.2f}"],
            }
        )
        st.table(proba_df)
