import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from fpdf import FPDF
import sys
import os
import joblib
import numpy as np
from datetime import timedelta

# Add script path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.pollution_predictor import predict_pollution, classify_pollution_level

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), "merged_data.csv")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# Rename columns
df.rename(columns={
    'PM2.5_x': 'PM2.5_sat', 'PM2.5_y': 'PM2.5_ground',
    'PM10_x': 'PM10_sat', 'PM10_y': 'PM10_ground',
    'NO2_level_x': 'NO2_level_sat', 'NO2_level_y': 'NO2_level_ground',
    'O3_x': 'O3_sat', 'O3_y': 'O3_ground',
    'CO_x': 'CO_sat', 'CO_y': 'CO_ground',
    'VOCs_x': 'VOCs_sat', 'VOCs_y': 'VOCs_ground',
    'SO2_x': 'SO2_sat', 'SO2_y': 'SO2_ground',
    'lat_x': 'lat', 'lon_x': 'lon'
}, inplace=True)

# Sidebar filters
st.sidebar.title("🌍 Air Pollution Dashboard")
countries = df['country'].unique().tolist()
sel_country = st.sidebar.selectbox("Country", countries)
df = df[df['country'] == sel_country]

# Global City Search
selected_city = st.text_input("🔍 Search City")
if selected_city:
    df_sel = df[df["city"].str.contains(selected_city, case=False)]
else:
    st.sidebar.info("Cities: " + ", ".join(sorted(df['city'].unique())))
    df_sel = df.copy()

# Pollutant selection
pollutants = [col for col in df.columns if any(p in col for p in ['PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'VOCs', 'SO2'])]
col = st.sidebar.selectbox("Pollutant", pollutants)
df_sel = df_sel.sort_values('date')

# Classification
if col in df_sel.columns:
    df_sel['category'] = df_sel[col].apply(classify_pollution_level)

# Line Chart
st.subheader(f"📈 {col} Trend")
if col in df_sel.columns:
    fig = px.line(df_sel, x='date', y=col, color='city', markers=True)
    st.plotly_chart(fig)

# Satellite vs Ground
if '_sat' in col:
    ground_col = col.replace('_sat', '_ground')
    if ground_col in df_sel.columns:
        st.subheader("🛰️ Satellite vs Ground Comparison")
        fig2 = px.line(df_sel, x='date', y=[col, ground_col], color='city', markers=True)
        st.plotly_chart(fig2)

# Heatmap
st.subheader("🌡️ Time-lapse Heatmap")
if 'lat' in df_sel.columns and 'lon' in df_sel.columns:
    from folium.plugins import HeatMapWithTime
    heatmap_data = df_sel.groupby(['date']).apply(
        lambda g: [[row['lat'], row['lon'], row.get(col, 0)] for _, row in g.iterrows()]
    ).tolist()
    m = folium.Map(location=[df_sel['lat'].mean(), df_sel['lon'].mean()], zoom_start=4)
    HeatMapWithTime(heatmap_data, index=df_sel['date'].dt.strftime('%Y-%m-%d').unique().tolist()).add_to(m)
    folium_static(m)

# Marker Map
st.subheader(f"🗺️ City Map with {col} Markers")
marker_map = folium.Map(location=[df_sel['lat'].mean(), df_sel['lon'].mean()], zoom_start=4)
city_group = df_sel.groupby("city").last().reset_index()
for _, row in city_group.iterrows():
    value = row[col]
    popup_text = f"{row['city']}, {row['country']}<br>{col}: {value}"
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=popup_text,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(marker_map)
folium_static(marker_map)

# Export to PDF
def export_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Air Pollution Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Country: {sel_country}", ln=2)
    for city in df_sel['city'].unique():
        city_data = df_sel[df_sel['city'] == city]
        if not city_data.empty:
            latest = city_data.iloc[-1]
            value = latest[col]
            category = classify_pollution_level(value)
            pdf.cell(200, 10, txt=f"{city} ➤ {col}: {value:.2f} → {category}", ln=1)
    pdf.output("pollution_report.pdf")

if st.button("📄 Export PDF Report"):
    export_pdf()
    st.success("✅ Report saved as `pollution_report.pdf`")

# Alert Summary
st.subheader("🚨 Pollution Alerts")
for city in df_sel['city'].unique():
    city_data = df_sel[df_sel['city'] == city]
    if not city_data.empty:
        latest = city_data.iloc[-1]
        value = latest[col]
        category = classify_pollution_level(value)
        st.markdown(f"**{city}** ➤ {col}: `{value}` → **{category}**")

# Forecast
st.subheader("🔮 Forecast for Next 7 Days")
st.info(f"🧪 Available data points: {df_sel[col].dropna().shape[0]}")

try:
    pred_df = predict_pollution(df_sel[['date', col]].dropna(), column=col, days=7)
    fig_forecast = px.line(pred_df, x='date', y=f'predicted_{col}', markers=True,
                           title=f"Predicted {col} for Next 7 Days")
    st.plotly_chart(fig_forecast)
except Exception as e:
    st.warning(f"⚠️ Could not forecast: {e}")

# st.subheader("🚨 Latest Pollution Alert by City")
# try:
#     df_latest["PM2.5_ground"] = df_latest["PM2.5_ground"].astype(str)
#     df_latest["Alert"] = df_latest["Alert"].astype(str)
#     st.dataframe(df_latest[["city", "PM2.5_ground", "Alert"]])
# except Exception as e:
#     st.error(f"Error showing alert table: {e}")

