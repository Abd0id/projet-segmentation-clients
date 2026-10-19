import os
import sys
import streamlit as st
import pandas as pd
import joblib

# Hardcoded path to the saved model file
MODEL_PATH = "../../models/pipeline_segmentation.joblib"

# This folder is where the source code lives, we add it so Python can find pipeline.py
HERE = os.path.dirname(os.path.abspath(__file__))

def run_pipeline():
    # Add the source folder to Python's search path so we can import pipeline
    sys.path.insert(0, HERE)
    from pipeline import run_full_pipeline
    run_full_pipeline()


st.set_page_config(page_title="Customer Segment Predictor")

# Check if the model file exists before trying to load it
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Running the full pipeline, this may take a few minutes.")
    with st.spinner("Training in progress..."):
        run_pipeline()
    st.success("Pipeline done. Model is ready.")
    st.rerun()


# cache_resource makes Streamlit load the model only once and reuse it
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
# feature_names_in_ gives us the list of column names the model was trained on
features = list(model.feature_names_in_)

st.title("Customer Segment Predictor")
st.write("Enter customer data to predict their segment.")

values = {}
for feature in features:
    values[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    new_customer = pd.DataFrame([values])
    segment = model.predict(new_customer)[0]
    st.success("Predicted segment: " + str(segment))

    # hasattr checks if the model supports probability predictions
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(new_customer)[0]
        st.bar_chart(pd.Series(proba, index=model.classes_))