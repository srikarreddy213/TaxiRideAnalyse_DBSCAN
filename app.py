import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="NYC Taxi DBSCAN Clustering",
    page_icon="🚕",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
    }
    h1 {
        text-align: center;
        color: #00F5A0;
    }
    .stMetric {
        background-color: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown("<h1>🚕 NYC Taxi Pickup Clustering using DBSCAN</h1>", unsafe_allow_html=True)
st.markdown("<center><span style='color:#B0BEC5'>Density-Based Spatial Clustering Visualization</span></center>", unsafe_allow_html=True)

# --------------------------------------------------
# Safe Dataset Loader
# --------------------------------------------------
@st.cache_data
def load_data():
    try:
        file_path = os.path.join(
            os.path.dirname(__file__),
            "NewYorkCityTaxiTripDuration.csv"
        )

        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(
            file_path,
            usecols=["pickup_latitude", "pickup_longitude"]
        )

        df = df.dropna()

        if df.empty:
            return None

        return df

    except:
        return None


# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = load_data()

if df is None:
    st.error("❌ Dataset not found OR invalid format.")
    st.info("Make sure NewYorkCityTaxiTripDuration.csv is in SAME folder as app.py")
    st.stop()

st.success(f"Dataset Loaded Successfully ✅ | Total Records: {len(df)}")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ DBSCAN Parameters")

eps = st.sidebar.slider("eps value", 0.1, 1.0, 0.3, 0.1)
min_samples = st.sidebar.slider("min_samples", 3, 20, 5)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --------------------------------------------------
# Model
# --------------------------------------------------
db = DBSCAN(eps=eps, min_samples=min_samples)
labels = db.fit_predict(X_scaled)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
noise_ratio = n_noise / len(labels)

mask = labels != -1

if len(set(labels[mask])) > 1:
    silhouette = silhouette_score(X_scaled[mask], labels[mask])
else:
    silhouette = None

col1, col2, col3 = st.columns(3)

col1.metric("Clusters", n_clusters)
col2.metric("Noise Points", n_noise)
col3.metric("Noise Ratio", round(noise_ratio, 4))

if silhouette is not None:
    st.info(f"Silhouette Score: {round(silhouette, 4)}")
else:
    st.warning("Silhouette Score Not Applicable")

# --------------------------------------------------
# Plot
# --------------------------------------------------
st.subheader("📊 Cluster Visualization")

fig, ax = plt.subplots(figsize=(8, 6))

unique_labels = set(labels)
colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        ax.scatter(
            X_scaled[labels == label, 0],
            X_scaled[labels == label, 1],
            c="white",
            marker="x",
            s=15,
            label="Noise"
        )
    else:
        ax.scatter(
            X_scaled[labels == label, 0],
            X_scaled[labels == label, 1],
            color=color,
            s=15,
            label=f"Cluster {label}"
        )

ax.set_facecolor("#0E1117")
ax.set_xlabel("Latitude (Scaled)", color="white")
ax.set_ylabel("Longitude (Scaled)", color="white")
ax.tick_params(colors='white')
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div style='text-align:center; color:#9CA3AF; padding-top:20px;'>
Built with ❤️ using Streamlit | DBSCAN Clustering Project
</div>
""", unsafe_allow_html=True)
