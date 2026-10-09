
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_prediction_model.pkl"
DATA_PATH = BASE_DIR / "datasets" / "Diabetes dataset.CSV"

st.title("🩺 Diabetes Prediction System")
st.write(
    "An educational machine-learning demonstration "
    "using Logistic Regression."
)

st.warning(
    "Educational use only. This model cannot diagnose diabetes "
    "or replace professional medical advice."
)

# ==========================================
# 2. LOAD MODEL AND DATASET
# ==========================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_dataset_columns(path):
    # Read only the header; no need to load the complete dataset
    return list(pd.read_csv(path, nrows=0).columns)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Save your trained model as diabetes_prediction_model.pkl "
        "inside D:\\Diabete."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.info(
        "Check that the CSV exists inside the datasets folder "
        "and that its filename is exactly 'Diabetes dataset.CSV'."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    dataset_columns = load_dataset_columns(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load model or dataset: {error}")
    st.stop()

# ==========================================
# 3. DETERMINE FEATURE NAMES
# ==========================================
# Prefer feature names saved by the complete pipeline,
# if available. Otherwise use the CSV header.
if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = [
        col for col in dataset_columns
        if col.strip().lower() != "outcome"
    ]

# Find the number of input features expected by the model
try:
    expected_features = model.n_features_in_
except AttributeError:
    try:
        expected_features = (
            model.named_steps["classifier"].n_features_in_
        )
    except (AttributeError, KeyError, TypeError):
        st.error(
            "Cannot determine the number of model input features. "
            "Please check how the model was trained."
        )
        st.stop()

if len(feature_names) != expected_features:
    st.error(
        f"Feature mismatch: the app has {len(feature_names)} "
        f"features, but the model expects {expected_features}."
    )
    st.write("Dataset columns:", feature_names)
    st.info(
        "Retrain the model using the same input columns and their "
        "original order, then save the updated pipeline."
    )
    st.stop()

if any(col not in dataset_columns for col in feature_names):
    st.error(
        "The model feature names do not match the dataset columns. "
        "Check the training dataset and column order."
    )
    st.write("Model features:", feature_names)
    st.write("CSV columns:", dataset_columns)
    st.stop()

# ==========================================
# 4. INPUT FORM
# ==========================================
st.subheader("Enter Feature Values")
st.caption(
    "Enter values using the units and ranges appropriate to your "
    "training dataset. Example defaults are not medical advice."
)

with st.form("diabetes_prediction_form"):
    input_values = {}

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_names):
        display_name = feature.replace("_", " ").title()

        # Defaults are illustrative only; change to match your dataset.
        if feature.lower() == "pregnancies":
            with col1 if index % 2 == 0 else col2:
                input_values[feature] = st.number_input(
                    display_name,
                    min_value=0,
                    value=1,
                    step=1
                )

        elif feature.lower() == "age":
            with col1 if index % 2 == 0 else col2:
                input_values[feature] = st.number_input(
                    display_name,
                    min_value=1,
                    value=30,
                    step=1
                )

        else:
            with col1 if index % 2 == 0 else col2:
                input_values[feature] = st.number_input(
                    display_name,
                    min_value=0.0,
                    value=1.0,
                    step=1.0,
                    format="%.3f"
                )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

# ==========================================
# 5. MAKE PREDICTION
# ==========================================
if submitted:
    try:
        input_df = pd.DataFrame(
            [[input_values[col] for col in feature_names]],
            columns=feature_names
        )

        prediction = model.predict(input_df)[0]

        st.divider()
        st.subheader("Model Output")

        if prediction == 1:
            st.warning(
                "The model predicted the positive class (1)."
            )
        else:
            st.success(
                "The model predicted the negative class (0)."
            )

        # Display probability if supported by the saved model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                positive_probability = probabilities[
                    classes.index(1)
                ]

                st.metric(
                    "Estimated positive-class probability",
                    f"{positive_probability:.1%}"
                )

        st.caption(
            "This is a machine-learning output, not a diagnosis "
            "or a validated estimate of personal medical risk. "
            "Consult a qualified healthcare professional for "
            "medical assessment."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check the training feature order, data types, and "
            "saved model pipeline."
        )

# ==========================================
# 6. FOOTER
# ==========================================
st.divider()
st.caption("Diabetes Prediction System | Python • Pandas • Logistic Regression")