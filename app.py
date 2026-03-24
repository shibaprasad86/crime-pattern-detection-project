import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from mlxtend.frequent_patterns import fpgrowth, association_rules

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Crime Pattern Detection", layout="wide")
st.title("🚨 Crime Pattern Detection using Data Mining")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("01_District_wise_crimes_committed_IPC_2001_2012.csv")
    df.columns = df.columns.str.strip().str.lower()

    df = df.rename(columns={
        'state/ut': 'state',
        'total ipc crimes': 'total_crimes',
        'hurt/grevious hurt': 'hurt'
    })

    df['state'] = df['state'].str.title()
    df['district'] = df['district'].str.replace('ZZ TOTAL', 'TOTAL').str.title()
    return df

df = load_data()

# -------------------------------------------------
# Common Features (MUST MATCH TRAINING)
# -------------------------------------------------
FEATURES = [
    'murder','rape','theft',
    'riots','robbery','burglary',
    'kidnapping & abduction'
]

# -------------------------------------------------
# Load Saved Models (.pkl)
# -------------------------------------------------
@st.cache_resource
def load_models():
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("models/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    return scaler, rf_model, kmeans

scaler, rf_model, kmeans = load_models()

# -------------------------------------------------
# State Coordinates (for Hotspot Map)
# -------------------------------------------------
STATE_COORDINATES = {
    'Andhra Pradesh':[15.9,79.7],
    'Assam':[26.2,92.9],
    'Bihar':[25.1,85.3],
    'Chhattisgarh':[21.3,81.9],
    'Delhi':[28.6,77.2],
    'Gujarat':[22.3,71.2],
    'Haryana':[29.1,76.1],
    'Jharkhand':[23.6,85.3],
    'Karnataka':[15.3,75.7],
    'Kerala':[10.9,76.3],
    'Madhya Pradesh':[23.0,78.7],
    'Maharashtra':[19.8,75.7],
    'Odisha':[21.0,85.1],
    'Punjab':[31.1,75.3],
    'Rajasthan':[27.0,74.2],
    'Tamil Nadu':[11.1,78.7],
    'Uttar Pradesh':[26.8,80.9],
    'West Bengal':[23.0,87.9]
}

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
page = st.sidebar.radio(
    "Select Module",
    [
        "EDA",
        "Crime Prediction",
        "K-Means Clustering",
        "FP-Growth",
        "Hotspot Map"
    ]
)

# =================================================
# 1️⃣ EDA
# =================================================
if page == "EDA":
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        yearly = df.groupby("year")["total_crimes"].sum()
        st.line_chart(yearly)

    with col2:
        crime_sum = df[FEATURES].sum()
        fig, ax = plt.subplots()
        ax.pie(crime_sum, labels=crime_sum.index, autopct='%1.1f%%')
        st.pyplot(fig)

    top_states = df.groupby("state")["total_crimes"].sum().nlargest(10)
    st.subheader("Top 10 States by Crime")
    fig, ax = plt.subplots()
    sns.barplot(x=top_states.values, y=top_states.index, ax=ax)
    st.pyplot(fig)

    st.subheader("Crime Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# =================================================
# 2️⃣ USER INPUT PREDICTION (PKL MODEL)
# =================================================
elif page == "Crime Prediction":
    st.header("🔮 Crime Level Prediction")

    user_values = []
    for col in FEATURES:
        user_values.append(st.number_input(col.capitalize(), min_value=0, value=10))

    if st.button("Predict Crime Level"):
        X = np.array(user_values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = rf_model.predict(X_scaled)[0]

        result = "HIGH CRIME AREA" if pred == 1 else "LOW CRIME AREA"
        st.success(f"✅ Predicted Result: **{result}**")

# =================================================
# 3️⃣ K-MEANS CLUSTERING (PKL MODEL)
# =================================================
elif page == "K-Means Clustering":
    st.header("🔗 District Clustering")

    cluster_df = df[df['district'] != 'Total'].copy()
    X = cluster_df[FEATURES]
    X_scaled = scaler.transform(X)

    cluster_df['cluster'] = kmeans.predict(X_scaled)

    st.bar_chart(cluster_df['cluster'].value_counts().sort_index())

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=cluster_df,
        x='theft',
        y='murder',
        hue='cluster',
        palette='tab10',
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Cluster Profiles")
    st.dataframe(cluster_df.groupby('cluster')[FEATURES].mean().round(1))

# =================================================
# 4️⃣ FP-GROWTH (OPTIMIZED)
# =================================================
elif page == "FP-Growth":
    st.header("🔍 Frequent Pattern Mining (FP-Growth)")

    support = st.slider("Min Support", 0.2, 0.6, 0.3, 0.05)
    confidence = st.slider("Min Confidence", 0.5, 0.9, 0.7, 0.05)

    binary_df = df[FEATURES].apply(lambda x: x > x.median())

    itemsets = fpgrowth(binary_df, min_support=support, use_colnames=True)
    itemsets['length'] = itemsets['itemsets'].apply(len)
    itemsets = itemsets[itemsets['length'] <= 3]

    rules = association_rules(
        itemsets,
        metric='confidence',
        min_threshold=confidence
    ).sort_values(by='lift', ascending=False)

    st.dataframe(
        rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    )

# =================================================
# 5️⃣ HOTSPOT MAP
# =================================================
elif page == "Hotspot Map":
    st.header("🔥 Crime Hotspot Map (India)")

    state_crime = df.groupby('state')['total_crimes'].sum().reset_index()
    heat_data = []

    for _, row in state_crime.iterrows():
        if row['state'] in STATE_COORDINATES:
            lat, lon = STATE_COORDINATES[row['state']]
            heat_data.append([lat, lon, row['total_crimes']])

    m = folium.Map(location=[22.5, 80], zoom_start=5, tiles='cartodbpositron')
    HeatMap(heat_data, radius=25, blur=15).add_to(m)

    st_folium(m, width=900, height=500)
