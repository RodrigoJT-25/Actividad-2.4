import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import plotly.express as px

# ---------------- Configuración ----------------
st.set_page_config(page_title="K-Means con PCA (Simple)", layout="wide")
st.title("🎯 Clustering Interactivo con K-Means y PCA (Simple)")
st.write(
    "Sube un CSV, selecciona columnas numéricas, aplica **K-Means** y observa los grupos en **PCA (2D o 3D)**."
)

# ---------------- Subir archivo ----------------
st.sidebar.header("📂 Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

if uploaded_file is None:
    st.info("👈 Carga un archivo CSV en la barra lateral para comenzar.")
    st.write(
        """
        **Ejemplo de formato:**
        | Ingreso_Anual | Gasto_Tienda | Edad |
        |---------------|--------------|------|
        | 45000         | 350          | 28   |
        | 72000         | 680          | 35   |
        | 28000         | 210          | 22   |
        """
    )
    st.stop()

# ---------------- Lectura y preview ----------------
data = pd.read_csv(uploaded_file)
st.success("✅ Archivo cargado correctamente.")
st.write("### Vista previa de los datos")
st.dataframe(data.head())

# ---------------- Columnas numéricas ----------------
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) < 2:
    st.warning("⚠️ Se necesitan al menos dos columnas numéricas.")
    st.stop()

st.sidebar.header("⚙️ Configuración del modelo")
selected_cols = st.sidebar.multiselect(
    "Selecciona columnas numéricas para el clustering:",
    options=numeric_cols,
    default=numeric_cols
)
if len(selected_cols) < 2:
    st.warning("⚠️ Selecciona al menos dos columnas.")
    st.stop()

k = st.sidebar.slider("Número de clusters (k)", 2, 10, 3, 1)
pca_dim = st.sidebar.radio("Dimensiones PCA", [2, 3], index=0)

# ---------------- Preparación de datos ----------------
X = data[selected_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- PCA ----------------
pca = PCA(n_components=pca_dim, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_cols = [f"PC{i+1}" for i in range(pca_dim)]
df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=data.index)

# ---------------- Antes de K-Means ----------------
st.subheader("📊 Distribución original (antes de K-Means)")
if pca_dim == 2:
    fig_before = px.scatter(
        df_pca, x="PC1", y="PC2",
        opacity=0.7, title="PCA (2D) sin etiquetas",
        hover_data=data[selected_cols]
    )
else:
    fig_before = px.scatter_3d(
        df_pca, x="PC1", y="PC2", z="PC3",
        opacity=0.7, title="PCA (3D) sin etiquetas",
        hover_data=data[selected_cols]
    )
st.plotly_chart(fig_before, use_container_width=True)

# ---------------- K-Means ----------------
kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
inertia = float(kmeans.inertia_)

# ---------------- Después de K-Means ----------------
st.subheader(f"🎯 Datos agrupados con K-Means (k = {k})")
df_after = df_pca.copy()
df_after["cluster"] = labels.astype(int).astype(str)

if pca_dim == 2:
    fig_after = px.scatter(
        df_after, x="PC1", y="PC2",
        color="cluster", opacity=0.9,
        title=f"Clusters en PCA (2D) | Inercia: {inertia:,.2f}",
        hover_data=data[selected_cols]
    )
else:
    fig_after = px.scatter_3d(
        df_after, x="PC1", y="PC2", z="PC3",
        color="cluster", opacity=0.9,
        title=f"Clusters en PCA (3D) | Inercia: {inertia:,.2f}",
        hover_data=data[selected_cols]
    )
st.plotly_chart(fig_after, use_container_width=True)

# ---------------- Centroides en PCA ----------------
st.subheader("📍 Centroides de los clusters (en espacio PCA)")
centroids_pca = pca.transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_pca, columns=pca_cols)
centroids_df.index = [f"centroide_{i}" for i in range(k)]
st.dataframe(centroids_df)

# ---------------- Método del codo ----------------
st.subheader("📉 Método del Codo (Elbow)")
max_k = st.slider("Explorar k hasta:", 5, 15, 10, 1)
k_range = list(range(2, max_k + 1))
inertias = []
for kk in k_range:
    km = KMeans(n_clusters=kk, init="k-means++", n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(float(km.inertia_))

fig_elbow = px.line(
    x=k_range, y=inertias, markers=True,
    labels={"x": "Número de clusters (k)", "y": "Inercia"},
    title="Curva del codo"
)
st.plotly_chart(fig_elbow, use_container_width=True)

# ---------------- Descargar resultados ----------------
st.subheader("💾 Descargar datos con clusters asignados")
result = data.copy()
result["cluster"] = labels
for col in pca_cols:
    result[f"PCA_{col}"] = df_pca[col]

buf = BytesIO()
result.to_csv(buf, index=False, encoding="utf-8-sig")
st.download_button(
    "⬇️ Descargar CSV con clusters",
    data=buf.getvalue(),
    file_name="resultados_kmeans_pca_simple.csv",
    mime="text/csv"
)
