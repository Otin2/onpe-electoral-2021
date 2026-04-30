import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Resultados Electorales ONPE 2021", layout="wide")
st.title("Resultados Electorales ONPE 2021")
st.write("Análisis de la Segunda Elección Presidencial 2021 - Resultados por mesa")
st.write("Alumno: Rolando Roller Veliz")

# Cargar datos
archivo = "ONPE_2021.csv"

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv(archivo, encoding="utf-8")
    except:
        df = pd.read_csv(archivo, encoding="latin1")

    # Limpieza básica: convertir columnas de votos a numérico
    columnas_votos = ["VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR", "VOTOS_VALIDOS", "VOTOS_NULOS", "VOTOS_BLANCOS"]
    for col in columnas_votos:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = cargar_datos()

# Mostrar información general
st.subheader("Información general")
col1, col2, col3 = st.columns(3)
col1.metric("Total de mesas", len(df))
col2.metric("Total de columnas", df.shape[1])
col3.metric("Valores nulos", df.isnull().sum().sum())

# Vista previa
st.subheader("Vista previa de los datos")
st.dataframe(df.head(20), use_container_width=True)

# Selección de columna para filtro
st.sidebar.header("Filtros")
columna_filtro = st.sidebar.selectbox("Selecciona una columna para filtrar", df.columns)
valores = df[columna_filtro].dropna().astype(str).unique().tolist()
valores.sort()
valor_seleccionado = st.sidebar.multiselect(
    f"Selecciona valores de {columna_filtro}",
    valores,
    default=valores[:5] if len(valores) > 5 else valores
)

# Aplicar filtro
if valor_seleccionado:
    df_filtrado = df[df[columna_filtro].astype(str).isin(valor_seleccionado)]
else:
    df_filtrado = df.copy()

st.subheader("Datos filtrados")
st.dataframe(df_filtrado, use_container_width=True)

# ── Gráficos por columna categórica ───────────────────────────────────────────
st.subheader("Gráficos")
columna_grafico = st.selectbox("Selecciona una columna para analizar", df_filtrado.columns)
conteo = df_filtrado[columna_grafico].astype(str).value_counts().reset_index()
conteo.columns = [columna_grafico, "Cantidad"]

colA, colB = st.columns(2)

with colA:
    st.write("Gráfico de barras")
    fig_bar = px.bar(
        conteo.head(10),
        x=columna_grafico,
        y="Cantidad",
        text_auto=True
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with colB:
    st.write("Gráfico circular")
    fig_pie = px.pie(
        conteo.head(10),
        names=columna_grafico,
        values="Cantidad"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Comparación de votos por candidato ────────────────────────────────────────
st.subheader("Comparación de votos por candidato")

candidatos = ["VOTOS_PERU_LIBRE", "VOTOS_FUERZA_POPULAR"]
nombres_candidatos = ["Pedro Castillo (Perú Libre)", "Keiko Fujimori (Fuerza Popular)"]
votos_totales = []
for c in candidatos:
    if c in df_filtrado.columns:
        votos_totales.append(df_filtrado[c].sum())
    else:
        votos_totales.append(0)

df_candidatos = pd.DataFrame({
    "Candidato": nombres_candidatos,
    "Votos": votos_totales
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

# ── Distribución votos válidos / nulos / blancos ─────────────────────────────
st.subheader("Distribución de votos válidos, nulos y blancos")

tipos_voto = ["VOTOS_VALIDOS", "VOTOS_NULOS", "VOTOS_BLANCOS"]
totales_voto = []
for t in tipos_voto:
    if t in df_filtrado.columns:
        totales_voto.append(df_filtrado[t].sum())
    else:
        totales_voto.append(0)

df_tipos = pd.DataFrame({
    "Tipo": tipos_voto,
    "Cantidad": totales_voto
})

fig_tipos = px.pie(
    df_tipos,
    names="Tipo",
    values="Cantidad",
    title="Distribución de tipos de voto"
)
st.plotly_chart(fig_tipos, use_container_width=True)

# ── Histograma: distribución de votos por mesa ───────────────────────────────
st.subheader("Histograma — Distribución de votos por mesa")

if "VOTOS_VALIDOS" in df_filtrado.columns:
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