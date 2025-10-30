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

# ---------------- Cargar datos ----------------
st.subheader("cargar datos")
file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])
if not file:
    st.stop()

df = pd.read_csv(file)

# ---------------- Vista rápida ----------------
st.subheader("Datos")
st.dataframe(df.head())

# ---------------- Selecciones ----------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) < 2:
    st.warning("El archivo necesita al menos dos columnas numéricas.")
    st.stop()

col1 = st.selectbox("Columna X", num_cols, index=0)
col2 = st.selectbox("Columna Y", num_cols, index=1)
if col1 == col2:
    st.warning("Selecciona columnas distintas para X y Y.")
    st.stop()

k = st.slider("k (número de clusters)", 2, 10, 3, 1)

# ---------------- Preparar & modelar ----------------
X = df[[col1, col2]].copy()

# Normaliza 0..1 (igual que en tu referencia de ejes)
scaler = MinMaxScaler()
Xn = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
labels = kmeans.fit_predict(Xn)
inertia = float(kmeans.inertia_)
centroids = kmeans.cluster_centers_

# ---- Mostrar centroides (como arreglo) e inercia, igual que tu ejemplo
st.write(centroids.tolist())
st.write(inertia)

# ---------------- Dispersión con centroides ----------------
fig1, ax1 = plt.subplots()

# puntos por cluster
for c in range(k):
    idx = labels == c
    ax1.scatter(Xn[idx, 0], Xn[idx, 1], alpha=0.85)

# centroides con cruz grande
ax1.scatter(centroids[:, 0], centroids[:, 1], marker="+", s=300)

# títulos y ejes
ax1.set_title("clientes")
ax1.set_xlabel(col1 if col1 else "X")
ax1.set_ylabel(col2 if col2 else "Y")

# texto k e inercia a la derecha (como en tu captura)
ax1.text(1.02, 0.75, f"k={k}\nInercia = {inertia:.2f}", transform=ax1.transAxes)

st.pyplot(fig1)

# ---------------- Método del codo ----------------
st.subheader("método del codo")
k_vals = list(range(2, 11))
inertias = []
for kk in k_vals:
    km = KMeans(n_clusters=kk, init="k-means++", n_init=10, random_state=42).fit(Xn)
    inertias.append(float(km.inertia_))

fig2, ax2 = plt.subplots()
ax2.plot(k_vals, inertias, marker="o")
ax2.set_xlabel("k")
ax2.set_ylabel("Inercia")
ax2.set_title("Curva del codo")
st.pyplot(fig2)
