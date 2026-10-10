import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL_PATH = ROOT / "models" / "pipeline_segmentation.joblib"


def run_pipeline():
    sys.path.insert(0, str(HERE))
    from pipeline import run_full_pipeline
    run_full_pipeline()


st.set_page_config(page_title="Customer Segment Predictor")

if not MODEL_PATH.exists():
    st.warning("Model not found. Running the full pipeline, this may take a few minutes.")
    with st.spinner("Training in progress..."):
        run_pipeline()
    st.success("Pipeline done. Model is ready.")
    st.rerun()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
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

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(new_customer)[0]
        st.bar_chart(pd.Series(proba, index=model.classes_))