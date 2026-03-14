import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main {
background: linear-gradient(90deg,#f5f7fa,#c3cfe2);
}
h1 {
color:#2c3e50;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🛍️ Customer Segmentation Dashboard")
st.write("K-Means based customer segmentation for marketing insights")

# Load dataset
df = pd.read_csv("store_customers.csv")
df = df.dropna()

# Features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# Sidebar
st.sidebar.header("🔍 Customer Prediction")

income = st.sidebar.slider("Annual Income (k$)",0,150,50)
spending = st.sidebar.slider("Spending Score (1-100)",0,100,50)

# Prediction
if st.sidebar.button("Predict Cluster"):

    new_customer = np.array([[income,spending]])
    new_customer_scaled = scaler.transform(new_customer)

    cluster = kmeans.predict(new_customer_scaled)

    st.sidebar.success(f"Predicted Cluster: {cluster[0]}")

# Metrics
col1,col2,col3 = st.columns(3)

col1.metric("Total Customers",len(df))
col2.metric("Average Income",round(df["Annual Income (k$)"].mean(),2))
col3.metric("Average Spending Score",round(df["Spending Score (1-100)"].mean(),2))

st.write("---")

# Layout
col1,col2 = st.columns(2)

# Dataset
with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

# Cluster plot
with col2:
    st.subheader("📍 Customer Segments")

    fig,ax = plt.subplots()

    scatter=ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score")

    st.pyplot(fig)

st.write("---")

st.subheader("📈 Cluster Distribution")

cluster_count=df["Cluster"].value_counts()

st.bar_chart(cluster_count)

st.write("---")
st.write("Built using Streamlit | Machine Learning Project")