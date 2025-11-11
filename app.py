import json, joblib, numpy as np, streamlit as st
from pathlib import Path

ARTIFACTS = Path("artifacts")
MODEL = joblib.load(ARTIFACTS / "model.joblib")
SCALER = joblib.load(ARTIFACTS / "scaler.joblib")
CLASSES = json.loads((ARTIFACTS / "label_encoder_classes.json").read_text())

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]

st.set_page_config(page_title="ANN Crop Recommendation", page_icon="🌾", layout="centered")
st.title("🌾 ANN Crop Recommendation (scikit-learn MLP)")

st.markdown("Enter soil nutrients and environmental values to get a crop recommendation.")

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7 = st.columns(3)

vals = {}
vals["N"] = c1.number_input("N (Nitrogen)", 0.0, 200.0, 90.0)
vals["P"] = c2.number_input("P (Phosphorus)", 0.0, 200.0, 42.0)
vals["K"] = c3.number_input("K (Potassium)", 0.0, 200.0, 43.0)
vals["temperature"] = c4.number_input("Temperature (°C)", -5.0, 55.0, 26.5)
vals["humidity"] = c5.number_input("Humidity (%)", 0.0, 100.0, 80.0)
vals["ph"] = c6.number_input("Soil pH", 0.0, 14.0, 6.5)
vals["rainfall"] = c7.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

if st.button("Recommend Crop"):
    x = np.array([[vals[f] for f in FEATURES]], dtype="float32")
    x_s = SCALER.transform(x)
    probs = MODEL.predict_proba(x_s)[0]
    idx = int(np.argmax(probs))
    st.success(f"✅ Recommended crop: **{CLASSES[idx]}**")
    with st.expander("See class probabilities"):
        st.json({c: float(p) for c, p in zip(CLASSES, probs)})
st.caption("Model: scikit-learn MLP (64-32) | Optimizer: Adam | Loss: log-loss")
