# 🚀 Predictive Maintenance using GRU, LSTM & Ensemble Learning

An end-to-end **Predictive Maintenance System** for estimating **Remaining Useful Life (RUL)** using the NASA CMAPSS turbofan engine dataset.
This project integrates **deep learning models, ensemble techniques, alert systems, and an interactive dashboard**.

---

## 📌 Project Overview

This project predicts how many cycles (time steps) are left before an engine fails.

# 🔧 PrognosAI-RUL: Remaining Useful Life Prediction

An end-to-end machine learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using NASA CMAPSS dataset.

---

## 🚀 Key Features
- 📊 Data preprocessing (normalization, clipping, feature engineering)
- 🤖 Deep Learning models (GRU & LSTM)
- 📈 Model evaluation (MAE, RMSE, R²)
- 📉 Visualization of predictions vs actual RUL
- 🖥️ Interactive dashboard for insights

---

## 🧠 Tech Stack
- Python
- PyTorch
- Pandas, NumPy
- Matplotlib, Plotly
- Streamlit (for dashboard)

---

### 🔁 Pipeline:

* Data preprocessing & feature engineering
* Time-series sequence generation
* GRU & LSTM model training
* Ensemble prediction (improved accuracy)
* Alert classification (Critical / Warning / Healthy)
* Interactive Streamlit dashboard

---

## 🎯 Objectives

* Predict Remaining Useful Life (RUL) from sensor data
* Compare GRU vs LSTM performance
* Improve results using ensemble learning
* Convert predictions into actionable maintenance alerts
* Visualize everything in a real-time dashboard

---

## 🧠 Models Used

### 🔹 GRU (Gated Recurrent Unit)

Efficient for time-series modeling with fewer parameters.

### 🔹 LSTM (Long Short-Term Memory)

Captures long-term dependencies in sequential data.

### 🔹 Ensemble Model

Final prediction combines both models:

```python
final_rul = 0.6 * GRU + 0.4 * LSTM
```

👉 GRU is given higher weight due to better performance.

---

## ⚠️ Alert System

Predicted RUL is converted into maintenance alerts:

| RUL Range | Alert       | Action                          |
| --------- | ----------- | ------------------------------- |
| ≤ 35      | 🔴 CRITICAL | Immediate maintenance required  |
| 35 – 85   | 🟠 WARNING  | Maintenance should be scheduled |
| > 85      | 🟢 HEALTHY  | Operating normally              |

---

## 📊 Model Performance

| Model      |         MAE |        RMSE |   R² Score |
| ---------- | ----------: | ----------: | ---------: |
| GRU        |     10.5485 |     17.1120 |     0.8318 |
| LSTM       |     10.7154 |     17.5535 |     0.8230 |
| ⭐ Ensemble | **10.3259** | **16.7996** | **0.8379** |

👉 Ensemble model performs best.

---

## 📈 Dashboard Features

* 📊 Alert distribution visualization
* 📉 RUL distribution histogram
* 🔍 Engine-wise prediction explorer
* ⚖️ Model comparison (GRU vs LSTM vs Ensemble)
* 📉 Training loss curves
* 🚨 Top critical engines

---

## 🖥️ How to Run Locally

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Prognos-AI/cmapss-rul-prediction.git
cd cmapss-rul-prediction
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate (Windows):

```bash
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Dashboard

```bash
streamlit run dashboard.py
```

---

## 📂 Project Structure

```
cmapss-rul-prediction/
│
├── main.ipynb
├── dashboard.py
├── best_gru_model.pth
├── best_lstm_model.pth
├── scaler.pkl
├── feature_cols.pkl
├── final_test_alert_results.csv
├── model_comparison.csv
├── ensemble_metrics.csv
├── gru_loss_history.csv
├── lstm_loss_history.csv
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* PyTorch
* Matplotlib
* Streamlit

---

## 🚀 Deployment

This project can be deployed using **Streamlit Community Cloud**:

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Select repo
4. Choose `dashboard.py`
5. Deploy

---

## 🔮 Future Improvements

* Live real-time prediction system
* Advanced ensemble (stacking)
* Maintenance cost optimization
* API integration for industrial use

---

## 👨‍💻 Author

Developed as part of a predictive maintenance project using deep learning and real-world turbofan engine data.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it really helps!
