import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Resultados Electorales ONPE 2021", layout="wide")
st.title("Resultados Electorales ONPE 2021")
st.write("Análisis de la Segunda Elección Presidencial 2021 - Resultados por mesa")
st.write("Alumno: Rolando Roller Veliz")

archivo = "ONPE_2021.csv"

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv(archivo, encoding="utf-8")
    except:
        df = pd.read_csv(archivo, encoding="latin1")

    columnas_votos = ["VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR", "VOTOS_VALIDOS", "VOTOS_NULOS", "VOTOS_BLANCOS"]
    for col in columnas_votos:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = cargar_datos()

# ── PARTE 2: Métricas ─────────────────────────────────────────────────────────
st.subheader("Información general")
col1, col2, col3 = st.columns(3)
col1.metric("Total de mesas", f"{len(df):,}")
col2.metric("Total de columnas", df.shape[1])
col3.metric("Valores nulos", df.isnull().sum().sum())

col4, col5, col6 = st.columns(3)
col4.metric("Votos Perú Libre (Castillo)", f"{df['VOTOS_PERU_LIBRE'].sum():,}")
col5.metric("Votos Fuerza Popular (Fujimori)", f"{df['VOTOS_FUERZA_POPULAR'].sum():,}")
col6.metric("Total votos válidos", f"{df['VOTOS_VALIDOS'].sum():,}")

col7, col8, col9 = st.columns(3)
col7.metric("Votos en blanco", f"{df['VOTOS_BLANCOS'].sum():,}")
col8.metric("Votos nulos", f"{df['VOTOS_NULOS'].sum():,}")
col9.metric("Departamentos", df['DEPARTAMENTO'].nunique())

st.subheader("Vista previa de los datos")
st.dataframe(df.head(20), use_container_width=True)

# ── FILTROS SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.header("Filtros")
columna_filtro = st.sidebar.selectbox("Selecciona una columna para filtrar", df.columns)
valores = df[columna_filtro].dropna().astype(str).unique().tolist()
valores.sort()
valor_seleccionado = st.sidebar.multiselect(
    f"Selecciona valores de {columna_filtro}",
    valores,
    default=valores[:5] if len(valores) > 5 else valores
)

if valor_seleccionado:
    df_filtrado = df[df[columna_filtro].astype(str).isin(valor_seleccionado)]
else:
    df_filtrado = df.copy()

st.subheader("Datos filtrados")
st.dataframe(df_filtrado, use_container_width=True)

# ── PARTE 3: Gráficos dinámicos ───────────────────────────────────────────────
st.subheader("Gráficos")
columna_grafico = st.selectbox("Selecciona una columna para analizar", df_filtrado.columns)
conteo = df_filtrado[columna_grafico].astype(str).value_counts().reset_index()
conteo.columns = [columna_grafico, "Cantidad"]

colA, colB = st.columns(2)
with colA:
    st.write("Gráfico de barras")
    fig_bar = px.bar(conteo.head(10), x=columna_grafico, y="Cantidad", text_auto=True)
    st.plotly_chart(fig_bar, use_container_width=True)

with colB:
    st.write("Gráfico circular")
    fig_pie = px.pie(conteo.head(10), names=columna_grafico, values="Cantidad")
    st.plotly_chart(fig_pie, use_container_width=True)

# ── PARTE 3: Votos por candidato ──────────────────────────────────────────────
st.subheader("Votos por candidato")

df_candidatos = pd.DataFrame({
    "Candidato": ["Pedro Castillo (Perú Libre)", "Keiko Fujimori (Fuerza Popular)"],
    "Votos": [
        df_filtrado["VOTOS_PERU_LIBRE"].sum(),
        df_filtrado["VOTOS_FUERZA_POPULAR"].sum()
    ]
})

fig_candidatos = px.bar(
    df_candidatos,
    x="Candidato",
    y="Votos",
    text_auto=True,
    title="Votos totales por candidato",
    color="Candidato"
)
st.plotly_chart(fig_candidatos, use_container_width=True)

# ── PARTE 3: Distribución por región ─────────────────────────────────────────
st.subheader("Distribución de votos por región")

df_region = df_filtrado.groupby("DEPARTAMENTO")[["VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR"]].sum().reset_index()
df_region = df_region.sort_values("VOTOS_PERU_LIBRE", ascending=False)

fig_region = px.bar(
    df_region,
    x="DEPARTAMENTO",
    y=["VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR"],
    title="Votos por departamento",
    labels={"value": "Votos", "variable": "Candidato"},
    barmode="group",
    color_discrete_map={
        "VOTOS_PERU_LIBRE": "#E63946",
        "VOTOS_FUERZA_POPULAR": "#F4A261"
    }
)
fig_region.update_xaxes(tickangle=45)
st.plotly_chart(fig_region, use_container_width=True)

# ── PARTE 3: Comparación tipos de voto ───────────────────────────────────────
st.subheader("Comparación de resultados")

df_tipos = pd.DataFrame({
    "Tipo": ["Votos Válidos", "Votos Nulos", "Votos Blancos"],
    "Cantidad": [
        df_filtrado["VOTOS_VALIDOS"].sum(),
        df_filtrado["VOTOS_NULOS"].sum(),
        df_filtrado["VOTOS_BLANCOS"].sum()
    ]
})

fig_tipos = px.pie(
    df_tipos,
    names="Tipo",
    values="Cantidad",
    title="Distribución de tipos de voto"
)
st.plotly_chart(fig_tipos, use_container_width=True)

# ── PARTE 3: Histograma ───────────────────────────────────────────────────────
st.subheader("Histograma — Distribución de votos por mesa")

df_votos = df_filtrado.dropna(subset=["VOTOS_VALIDOS"]).copy()
df_votos["VOTOS_VALIDOS"] = pd.to_numeric(df_votos["VOTOS_VALIDOS"], errors="coerce")

fig_hist = px.histogram(
    df_votos,
    x="VOTOS_VALIDOS",
    nbins=20,
    title="Distribución de votos válidos por mesa",
    labels={"VOTOS_VALIDOS": "Votos válidos", "count": "N° de mesas"},
    color_discrete_sequence=["#636EFA"]
)
fig_hist.update_layout(bargap=0.05)
st.plotly_chart(fig_hist, use_container_width=True)

# ── PARTES 4 y 5: Machine Learning ───────────────────────────────────────────
st.subheader("Machine Learning aplicado a datos electorales")

df_ml = df[["N_ELEC_HABIL", "VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR", "VOTOS_VALIDOS"]].dropna()
df_ml = df_ml[df_ml["N_ELEC_HABIL"] > 0].copy()

X = df_ml[["N_ELEC_HABIL"]]
y = df_ml["VOTOS_PERU_LIBRE"]

# ── PARTE 5: Train / Test split ───────────────────────────────────────────────
st.subheader("División del dataset — Entrenamiento y Prueba")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

col1, col2 = st.columns(2)
col1.metric("Datos de entrenamiento (80%)", f"{len(X_train):,} mesas")
col2.metric("Datos de prueba (20%)", f"{len(X_test):,} mesas")

# ── PARTE 4: Regresión Lineal ─────────────────────────────────────────────────
st.subheader("Modelo de Regresión Lineal")
st.write("Predicción de votos para Castillo en base al número de electores hábiles por mesa")

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("MSE", f"{mse:,.2f}")
col2.metric("R²", f"{r2:.4f}")

fig_ml = px.scatter(
    x=y_test,
    y=y_pred,
    labels={"x": "Votos reales", "y": "Votos predichos"},
    title="Regresión Lineal — Votos reales vs predichos",
    opacity=0.3
)
fig_ml.add_shape(
    type="line",
    x0=float(y_test.min()), y0=float(y_test.min()),
    x1=float(y_test.max()), y1=float(y_test.max()),
    line=dict(color="red", dash="dash")
)
st.plotly_chart(fig_ml, use_container_width=True)

# ── PARTE 4: Agrupamiento KMeans ──────────────────────────────────────────────
st.subheader("Agrupamiento de mesas por comportamiento de voto")
st.write("Se agrupan las mesas en 3 clusters según los votos obtenidos por cada candidato")

X_cluster = df_ml[["VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR"]].copy()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_ml["GRUPO"] = kmeans.fit_predict(X_cluster)

fig_cluster = px.scatter(
    df_ml,
    x="VOTOS_PERU_LIBRE",
    y="VOTOS_FUERZA_POPULAR",
    color=df_ml["GRUPO"].astype(str),
    title="Agrupamiento de mesas (KMeans - 3 grupos)",
    labels={
        "VOTOS_PERU_LIBRE": "Votos Castillo",
        "VOTOS_FUERZA_POPULAR": "Votos Fujimori",
        "color": "Grupo"
    },
    opacity=0.4
)
st.plotly_chart(fig_cluster, use_container_width=True)

st.write("Promedio de votos por grupo:")
st.dataframe(
    df_ml.groupby("GRUPO")[["VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR", "N_ELEC_HABIL"]].mean().round(1),
    use_container_width=True
)