# 🌍 Air Pollution Monitoring Dashboard

An AI-powered interactive dashboard to monitor, forecast, and visualize global air pollution levels using satellite and ground data. Built using **Streamlit**, **Plotly**, **Folium**, and **scikit-learn**, this tool helps analyze trends, generate reports, and raise alerts for environmental health awareness.

---

## 🚀 Key Features

- 🔎 **Search & Filter** by Country, City, and Pollutant
- 📈 **Pollution Trend Visualization** with Plotly
- 🛰️ **Satellite vs Ground Data Comparison**
- 🗺️ **City-wise Marker Map** with pollutant values
- 🌡️ **Time-lapse Heatmap** using Folium
- 🔮 **7-day Forecasting** using ML & fallback smoothing
- 🧠 **Auto-Classification of Pollution Levels**
- 📄 **Export Pollution Report as PDF**
- ✅ **Supports all major pollutants**:  
  PM2.5, PM10, NO₂, O₃, CO, VOCs, SO₂
- 🌐 **Works globally**: India, USA, UK, Germany, Japan, China, Australia, France, South Africa, etc.

---

## 🧪 Tech Stack

- **Frontend**: Streamlit, Plotly, Folium  
- **Backend**: Python (Pandas, NumPy, Scikit-learn, Joblib)  
- **ML Models**: Linear Regression for time-series forecasting  
- **Utilities**: FPDF for PDF generation

---

## 📂 Folder Structure

airpollution-monitor/
│
├── dashboard/
│ ├── app.py ← Main Streamlit app
│
├── data/
│ └── merged_pollution_data.csv ← Merged satellite + ground dataset
│
├── models/
│ └── *.pkl ← Trained ML models for each pollutant
│
├── scripts/
│ └── pollution_predictor.py ← Forecasting + classification logic
│
├── train_model.py ← Model trainer for all pollutants
└── requirements.txt

yaml
Copy
Edit

---

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/<your-username>/airpollution-monitor.git
cd airpollution-monitor
