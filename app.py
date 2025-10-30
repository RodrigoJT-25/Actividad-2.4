import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------- Encabezado ----------------
st.set_page_config(page_title="k-means simple", layout="centered")
st.title("Aprendizaje no supervisado: k-means")
st.caption("By Oziel Velazquez ITC")

st.subheader("cargar datos")
file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if not file:
    st.stop()

# ---------------- Datos ----------------
df = pd.read_csv(file)
st.subheader("Datos")
st.dataframe(df.head())

# columnas numéricas y controles
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) < 2:
    st.warning("El archivo necesita al menos dos columnas numéricas.")
    st.stop()

col1, col2 = st.selectbox("Columna X", num_cols, index=0), st.selectbox("Columna Y", num_cols, index=1)
if col1 == col2:
    st.warning("Selecciona columnas distintas para X y Y.")
    st.stop()

k = st.slider("k (número de clusters)", 2, 10, 3, 1)

# ---------------- Preparación ----------------
X = df[[col1, col2]].copy()

# Normalizar a [0, 1] (como en tu ejemplo de ejes 0..1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- K-Means ----------------
kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
inertia = float(kmeans.inertia_)
centroids = kmeans.cluster_centers_

# ---------------- Dispersión con centroides ----------------
fig1, ax1 = plt.subplots()
# puntos por cluster
for c in range(k):
    idx = labels == c
    ax1.scatter(X_scaled[idx, 0], X_scaled[idx, 1], alpha=0.8)

# centroides (cruces grandes)
ax1.scatter(centroids[:, 0], centroids[:, 1], marker="+", s=300)

ax1.set_title("clientes" if "cliente" in df.columns[0].lower() else "Clusters")
ax1.set_xlabel(col1)
ax1.set_ylabel(col2)

# texto de k e inercia a la derecha del gráfico
xmax = 1.05
ypos = 0.85
ax1.text(xmax, ypos, f"k={k}\nInercia = {inertia:.2f}", transform=ax1.transAxes)

st.pyplot(fig1)

# ---------------- Método del codo ----------------
st.subheader("método del codo")
k_vals = list(range(2, 11))
inertias = []
for kk in k_vals:
    km = KMeans(n_clusters=kk, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
    inertias.append(float(km.inertia_))

fig2, ax2 = plt.subplots()
ax2.plot(k_vals, inertias, marker="o")
ax2.set_xlabel("k")
ax2.set_ylabel("Inercia")
ax2.set_title("Curva del codo")
st.pyplot(fig2)
