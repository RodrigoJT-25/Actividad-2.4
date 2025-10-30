import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Encabezado ---
st.set_page_config(page_title="k-means básico", layout="centered")
st.title("Aprendizaje no supervisado: k-means")

st.subheader("cargar datos")
file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if not file:
    st.stop()

# --- Leer datos ---
df = pd.read_csv(file)
st.subheader("Datos")
st.dataframe(df.head())

# --- Tomar las 2 primeras columnas numéricas ---
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
if len(num_cols) < 2:
    st.error("El archivo necesita al menos dos columnas numéricas.")
    st.stop()

col1, col2 = num_cols[0], num_cols[1]
X = df[[col1, col2]]

# --- Normalizar ---
scaler = MinMaxScaler()
Xn = scaler.fit_transform(X)

# --- K-Means ---
k = 3
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
labels = kmeans.fit_predict(Xn)
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

# --- Mostrar centroides e inercia ---
st.write(centroids.tolist())
st.write(inertia)

# --- Gráfico de dispersión ---
fig, ax = plt.subplots()
for i in range(k):
    cluster_points = Xn[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.8)
ax.scatter(centroids[:, 0], centroids[:, 1], marker="+", s=300)
ax.set_title("clientes")
ax.set_xlabel(col1)
ax.set_ylabel(col2)
ax.text(1.02, 0.75, f"k={k}\nInercia = {inertia:.2f}", transform=ax.transAxes)
st.pyplot(fig)

# --- Método del codo ---
st.subheader("método del codo")
k_values = range(2, 11)
inertias = []
for kk in k_values:
    km = KMeans(n_clusters=kk, n_init=10, random_state=42)
    km.fit(Xn)
    inertias.append(km.inertia_)

fig2, ax2 = plt.subplots()
ax2.plot(k_values, inertias, marker="o")
ax2.set_xlabel("k")
ax2.set_ylabel("Inercia")
ax2.set_title("Curva del codo")
st.pyplot(fig2)
