import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Gōku Lab · Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main { background-color: #0f0f0f; }
    .block-container { padding: 2rem 2.5rem; }

    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00e5a0;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #888;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.75rem;
        border-left: 3px solid #00e5a0;
        padding-left: 0.75rem;
    }
    .drill-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #00e5a0;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTO REFRESCO (cada 60 segundos)
# ─────────────────────────────────────────────

st_autorefresh(interval=60_000, key="autorefresh")

# ─────────────────────────────────────────────
# CONEXIÓN MONGODB
# ─────────────────────────────────────────────

@st.cache_resource
def conectar_mongo():
    uri = os.getenv("MONGO_URI") or st.secrets.get("MONGO_URI")
    if not uri:
        st.error("❌ No se encontró MONGO_URI. Agrégala en secrets o como variable de entorno.")
        st.stop()
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client["chatbot_Goku_lab"]

db = conectar_mongo()

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────

@st.cache_data(ttl=60)  # cache de 60 segundos — se refresca solo
def cargar_datos():
    docs = list(db["conversaciones"].find(
        {},
        {"_id": 0, "numero": 1, "mensaje": 1, "intencion": 1,
         "confianza": 1, "sentimiento": 1, "uso_rag": 1,
         "respuesta": 1, "timestamp": 1}
    ))
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hora"] = df["timestamp"].dt.hour
    df["fecha"] = df["timestamp"].dt.date
    df["confianza"] = pd.to_numeric(df["confianza"], errors="coerce")
    return df

df_raw = cargar_datos()

# ─────────────────────────────────────────────
# INTENCIONES A EXCLUIR (ruido)
# ─────────────────────────────────────────────

EXCLUIR = ["Saludo", "Despedida", "captura_numero"]

# ─────────────────────────────────────────────
# COLORES POR INTENCIÓN
# ─────────────────────────────────────────────

COLORES = {
    "Consultar_Cursos":        "#00e5a0",
    "Consultar_Costos":        "#ff6b6b",
    "Consultar_Horarios":      "#4ecdc4",
    "Consultar_Modalidad":     "#ffe66d",
    "Consultar_Certificacion": "#a29bfe",
    "Consultar_ClaseDemo":     "#fd79a8",
    "Consultar_FormasPago":    "#fdcb6e",
    "Consultar_RequisitosEdad":"#81ecec",
    "Consultar_Duracion":      "#6c5ce7",
    "Consultar_Ubicacion":     "#00cec9",
    "Desconocido":             "#636e72",
}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_logo, col_titulo, col_update = st.columns([1, 6, 2])
with col_logo:
    st.markdown("## 🎮")
with col_titulo:
    st.markdown("# Gōku Lab · Dashboard operativo")
    st.markdown("<p style='color:#555; font-size:0.8rem; margin-top:-12px'>Chatbot · Atención académica</p>", unsafe_allow_html=True)
with col_update:
    st.markdown(f"<p style='color:#444; font-size:0.75rem; text-align:right; margin-top:1.5rem'>↻ actualiza cada 60s<br>{datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — FILTROS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎛️ Filtros")

    # Rango de fechas
    if not df_raw.empty:
        fecha_min = df_raw["timestamp"].min().date()
        fecha_max = df_raw["timestamp"].max().date()
    else:
        fecha_min = fecha_max = datetime.now().date()

    rango = st.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    # Sentimiento
    sentimientos = ["Todos", "positivo", "neutral", "negativo"]
    filtro_sentimiento = st.selectbox("Sentimiento", sentimientos)

    # Incluir ruido
    incluir_ruido = st.checkbox("Incluir Saludo / Despedida", value=False)

    st.divider()
    st.markdown("<p style='color:#444; font-size:0.75rem'>Gōku Lab · Prácticas UNAM</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APLICAR FILTROS
# ─────────────────────────────────────────────

if df_raw.empty:
    st.warning("No hay datos en la colección todavía.")
    st.stop()

df = df_raw.copy()

# Fechas
if isinstance(rango, tuple) and len(rango) == 2:
    df = df[(df["fecha"] >= rango[0]) & (df["fecha"] <= rango[1])]

# Sentimiento
if filtro_sentimiento != "Todos":
    df = df[df["sentimiento"] == filtro_sentimiento]

# Ruido
if not incluir_ruido:
    df = df[~df["intencion"].isin(EXCLUIR)]

df_consultas = df[~df["intencion"].isin(EXCLUIR)]

# ─────────────────────────────────────────────
# MÉTRICAS PRINCIPALES
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">Resumen general</p>', unsafe_allow_html=True)

total_msgs     = len(df)
total_leads    = len(df_raw[df_raw["intencion"] == "captura_numero"])
tasa_rag       = df["uso_rag"].mean() * 100 if not df.empty else 0
confianza_prom = df["confianza"].mean() * 100 if not df.empty else 0
usuarios_uniq  = df["numero"].nunique() if not df.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)

for col, valor, label in [
    (c1, total_msgs,              "mensajes totales"),
    (c2, usuarios_uniq,           "usuarios únicos"),
    (c3, total_leads,             "leads capturados"),
    (c4, f"{tasa_rag:.1f}%",      "tasa RAG fallback"),
    (c5, f"{confianza_prom:.1f}%","confianza promedio"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{valor}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILA 1 — INTENCIONES + SENTIMIENTOS
# ─────────────────────────────────────────────

col_izq, col_der = st.columns([3, 2])

with col_izq:
    st.markdown('<p class="section-title">Consultas por intención</p>', unsafe_allow_html=True)

    if not df_consultas.empty:
        conteo = (
            df_consultas["intencion"]
            .value_counts()
            .reset_index()
            .rename(columns={"intencion": "Intención", "count": "Mensajes"})
        )
        conteo["color"] = conteo["Intención"].map(COLORES).fillna("#888")

        fig_bar = px.bar(
            conteo,
            x="Mensajes",
            y="Intención",
            orientation="h",
            color="Intención",
            color_discrete_map=COLORES,
            template="plotly_dark",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=320,
            yaxis=dict(categoryorder="total ascending"),
            font=dict(family="DM Sans", color="#aaa", size=12),
        )
        fig_bar.update_traces(marker_line_width=0)

        # Click en barra → drill down
        evento = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", key="bar_intencion")
        intencion_seleccionada = None
        if evento and evento.get("selection") and evento["selection"].get("points"):
            intencion_seleccionada = evento["selection"]["points"][0].get("label")
    else:
        st.info("Sin datos para mostrar.")
        intencion_seleccionada = None

with col_der:
    st.markdown('<p class="section-title">Distribución de sentimientos</p>', unsafe_allow_html=True)

    if not df.empty:
        sent_conteo = df["sentimiento"].value_counts().reset_index()
        sent_conteo.columns = ["Sentimiento", "Total"]

        colores_sent = {"positivo": "#00e5a0", "neutral": "#4ecdc4", "negativo": "#ff6b6b"}

        fig_donut = px.pie(
            sent_conteo,
            names="Sentimiento",
            values="Total",
            hole=0.6,
            color="Sentimiento",
            color_discrete_map=colores_sent,
            template="plotly_dark",
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#aaa", size=11)),
            margin=dict(l=0, r=0, t=0, b=0),
            height=320,
            font=dict(family="DM Sans", color="#aaa"),
        )
        fig_donut.update_traces(textfont_color="white")
        st.plotly_chart(fig_donut, use_container_width=True)



col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown('<p class="section-title">Mensajes por hora del día</p>', unsafe_allow_html=True)

    if not df.empty:
        por_hora = df.groupby("hora").size().reset_index(name="Mensajes")
        todas_horas = pd.DataFrame({"hora": range(24)})
        por_hora = todas_horas.merge(por_hora, on="hora", how="left").fillna(0)

        fig_hora = px.area(
            por_hora,
            x="hora",
            y="Mensajes",
            template="plotly_dark",
            color_discrete_sequence=["#00e5a0"],
        )
        fig_hora.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=240,
            xaxis=dict(title="Hora", tickmode="linear", dtick=2, color="#555"),
            yaxis=dict(title="", color="#555"),
            font=dict(family="DM Sans", color="#aaa"),
        )
        fig_hora.update_traces(line_color="#00e5a0", fillcolor="rgba(0,229,160,0.1)")
        st.plotly_chart(fig_hora, use_container_width=True)

with col_b:
    st.markdown('<p class="section-title">RAG vs Clasificador SVM</p>', unsafe_allow_html=True)

    if not df.empty:
        rag_conteo = df["uso_rag"].value_counts().reset_index()
        rag_conteo.columns = ["Tipo", "Total"]
        rag_conteo["Tipo"] = rag_conteo["Tipo"].map({True: "RAG fallback", False: "Clasificador SVM"})

        fig_rag = px.pie(
            rag_conteo,
            names="Tipo",
            values="Total",
            hole=0.6,
            color="Tipo",
            color_discrete_map={"RAG fallback": "#ff6b6b", "Clasificador SVM": "#00e5a0"},
            template="plotly_dark",
        )
        fig_rag.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#aaa", size=11)),
            margin=dict(l=0, r=0, t=0, b=0),
            height=240,
            font=dict(family="DM Sans", color="#aaa"),
        )
        fig_rag.update_traces(textfont_color="white")
        st.plotly_chart(fig_rag, use_container_width=True)



st.markdown('<p class="section-title">Confianza promedio por intención</p>', unsafe_allow_html=True)

if not df_consultas.empty:
    conf_prom = (
        df_consultas.groupby("intencion")["confianza"]
        .mean()
        .reset_index()
        .rename(columns={"intencion": "Intención", "confianza": "Confianza"})
        .sort_values("Confianza")
    )
    conf_prom["color"] = conf_prom["Confianza"].apply(
        lambda x: "#ff6b6b" if x < 0.6 else "#ffe66d" if x < 0.8 else "#00e5a0"
    )

    fig_conf = go.Figure(go.Bar(
        x=conf_prom["Confianza"],
        y=conf_prom["Intención"],
        orientation="h",
        marker_color=conf_prom["color"],
        text=conf_prom["Confianza"].apply(lambda x: f"{x:.0%}"),
        textposition="outside",
        textfont=dict(color="#aaa", size=11),
    ))
    fig_conf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=0, b=0),
        height=280,
        xaxis=dict(range=[0, 1.1], tickformat=".0%", color="#555"),
        yaxis=dict(color="#aaa"),
        font=dict(family="DM Sans", color="#aaa"),
    )
    st.plotly_chart(fig_conf, use_container_width=True)
    st.caption("🔴 < 60%  necesita más datos de entrenamiento  ·  🟡 60-80%  aceptable  ·  🟢 > 80%  excelente")


st.divider()

# Si hay selección en la gráfica de barras úsala, sino mostrar selector manual
if intencion_seleccionada:
    intencion_drill = intencion_seleccionada
    st.markdown(f'<p class="drill-header">📋 Mensajes de: {intencion_drill}</p>', unsafe_allow_html=True)
else:
    opciones = ["— Ver todos —"] + sorted(df_consultas["intencion"].unique().tolist()) if not df_consultas.empty else ["— Ver todos —"]
    intencion_drill = st.selectbox("📋 Detalle de mensajes por intención", opciones)
    if intencion_drill == "— Ver todos —":
        intencion_drill = None

# Filtrar tabla
if intencion_drill:
    df_tabla = df[df["intencion"] == intencion_drill].copy()
else:
    df_tabla = df.copy()

if not df_tabla.empty:
    df_tabla = df_tabla.sort_values("timestamp", ascending=False)
    df_tabla["timestamp"] = df_tabla["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_tabla["confianza"] = df_tabla["confianza"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")

    st.dataframe(
        df_tabla[["timestamp", "intencion", "mensaje", "sentimiento", "confianza", "uso_rag"]].rename(columns={
            "timestamp":  "Fecha/Hora",
            "intencion":  "Intención",
            "mensaje":    "Mensaje del usuario",
            "sentimiento":"Sentimiento",
            "confianza":  "Confianza",
            "uso_rag":    "RAG",
            "respuesta":   "Respuesta del bot",
        }),
        use_container_width=True,
        height=350,
        hide_index=True,
    )
    st.caption(f"{len(df_tabla)} mensajes · haz click en una barra arriba para filtrar automáticamente")
else:
    st.info("Sin mensajes para esta selección.")