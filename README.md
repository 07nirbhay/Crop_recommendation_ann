# 🌾 Crop Recommendation System using Artificial Neural Network (ANN)

A machine learning project that recommends the most suitable crop based on soil nutrients and environmental conditions.  
This implementation uses a **fast scikit-learn MLPClassifier (Artificial Neural Network)** and is deployed as an interactive **Streamlit web application**.

---

## 📌 Objective

To assist farmers and agricultural decision-makers by identifying the ideal crop to cultivate using measurable data such as soil nutrients and weather conditions.

---

## 📊 Dataset

**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

**Features Used:**
- **N** — Nitrogen content in soil  
- **P** — Phosphorus content  
- **K** — Potassium content  
- **Temperature** (°C)  
- **Humidity** (%)  
- **pH** — Soil acidity  
- **Rainfall** (mm)  

**Target:** Recommended crop (22 possible classes)

---

## 🧠 Model Used (ANN)

A fast Artificial Neural Network (ANN) using **scikit-learn MLPClassifier**:

```
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42
)
```

---

## ✅ Model Performance

**Test Accuracy:** 96.97%

**Metrics:**
- High precision and recall across all classes  
- Balanced dataset performance  
- Works well for real-world prediction

---

## 📦 Project Structure

```
📁 crop-recommendation-ann
│
├── app.py
├── requirements.txt
├── README.md
```

---

## 🚀 How to Run Locally

### 1) Install dependencies
```
pip install -r requirements.txt
```

### 2) Launch Streamlit app
```
streamlit run app.py
```

The app will open in your browser at:
```
http://localhost:8501
```

---

## 🧪 Usage Example

Enter soil & environment values:

```
N: 90
P: 42
K: 43
Temperature: 26.5 °C
Humidity: 80 %
pH: 6.5
Rainfall: 200 mm
```

The system outputs:
```
✅ Recommended crop: rice
```

---

## ☁️ Deployment Guide

### Streamlit Community Cloud
1. Push the repo to GitHub
2. Go to https://share.streamlit.io
3. Select your repo
4. Deploy using `app.py`

---

## 🔧 Technical Highlights

- Fast ANN model
- Lightweight and deployment-ready
- Feature scaling using StandardScaler
- Multi-class classification
- Interactive Streamlit UI

---

## 🧩 Limitations

- Dataset is not location-sensitive
- No time-series forecasting
- No profitability analysis

---

## 🔮 Future Improvements

- Integrate rainfall forecasting using LSTM
- Add soil-type classification
- Add GIS-based recommendations
- Add yield prediction

---

## 👨‍💻 Author

Your Name  
(Add your contact link if desired)
