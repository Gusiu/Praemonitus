"""Streamlit interface. Run with: streamlit run app.py"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import database
from ml.predictor import Predictor
from ml.validation import FIELD_LABELS, InputError

st.set_page_config(page_title="Praemonitus", layout="centered")


@st.cache_resource
def get_predictor() -> Predictor:
    """Loaded once per server process; Streamlit re-runs this script on every interaction."""
    return Predictor()


predictor = get_predictor()
card = predictor.card
ranges = card["training_ranges"]

st.title("Praemonitus")
st.caption("Machine failure prediction from process parameters (AI4I 2020 data)")

with st.form("prediction"):
    product_type = st.selectbox(FIELD_LABELS["type"], ["L", "M", "H"], index=1, key="type")
    cols = st.columns(2)
    defaults = {
        "air_temp_k": 300.0,
        "process_temp_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
    }
    values = {}
    for i, name in enumerate(defaults):
        bounds = ranges[name]
        values[name] = cols[i % 2].number_input(
            FIELD_LABELS[name],
            value=defaults[name],
            step=0.1 if "temp" in name else 1.0,
            format="%.1f",
            help=f"Training range: {bounds['min']:g} - {bounds['max']:g}",
            key=name,
        )
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        result = predictor.predict({"type": product_type, **values}, source="streamlit")
    except InputError as exc:
        for message in exc.messages:
            st.error(message)
    else:
        if result.prediction:
            st.error(f"FAILURE expected - stop the process and inspect the machine "
                     f"(probability {result.probability:.1%})")
        else:
            st.success(f"No failure expected (probability {result.probability:.1%})")
        st.metric("Failure probability", f"{result.probability:.1%}",
                  help=f"Decision threshold {result.threshold:.3f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature difference", f"{result.derived['temp_diff_k']:.1f} K")
        c2.metric("Power", f"{result.derived['power_w']:.0f} W")
        c3.metric("Tool strain", f"{result.derived['strain_min_nm']:.0f} min*Nm")
        if result.out_of_range:
            st.warning(
                "Outside the training data range: "
                + ", ".join(FIELD_LABELS[f] for f in result.out_of_range_fields)
                + ". The prediction is less reliable."
            )

st.subheader("History")
only = st.radio("Show", ["all", "failures", "no failures"], horizontal=True, key="filter")
filter_value = {"all": None, "failures": 1, "no failures": 0}[only]
history = database.get_history(50, prediction=filter_value, db_path=predictor.db_path)
if history:
    frame = pd.DataFrame(history)
    frame["created_at"] = [
        datetime.fromisoformat(t).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        for t in frame["created_at"]
    ]
    st.dataframe(
        frame[["id", "created_at", "source", "probability", "prediction", "out_of_range", "input_json"]],
        hide_index=True,
    )
else:
    st.info("No predictions yet.")
st.download_button(
    "Export history to CSV",
    database.history_csv_text(db_path=predictor.db_path),
    file_name="predictions.csv",
    mime="text/csv",
)

with st.expander("Model information"):
    st.write(f"**Model:** {card['model']} ({card['feature_set']} features)")
    st.write(f"**Model id:** `{card['model_id']}`, trained {card['trained_at']}")
    st.write(f"**Threshold:** {card['threshold']:.3f} (cost of a missed failure : false alarm "
             f"= {card['cost_ratio_fn_fp']}:1)")
    test = card["test_metrics"]
    st.write(
        f"**Test set:** PR-AUC {test['pr_auc']:.3f}, recall {test['recall']:.3f}, "
        f"precision {test['precision']:.3f}, {test['fp_per_1000']:.1f} false alarms per 1000 runs"
    )
