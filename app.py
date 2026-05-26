import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from datetime import date
from scipy.optimize import minimize
import requests

# 1. Configuración de la plataforma
st.set_page_config(page_title="Portfolio Intelligence Suite", layout="wide")

st.title("📊 Suite Financiera Pro: Optimización y Diagnóstico")
st.markdown("""
Esta plataforma te permite analizar activos de forma modular. 
Los activos locales (.BA) se detectan y **se dolarizan automáticamente usando el Dólar MEP**.
""")

# --- BARRA LATERAL ---
st.sidebar.header("🔧 1. Parámetros de Mercado")
lista_tickers = st.sidebar.text_input("Tickers (separados por coma):", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
fecha_fin = st.sidebar.date_input("Histórico hasta:", pd.to_datetime("2026-05-25"))
activos = [x.strip().upper() for x in lista_tickers.split(",") if x.strip()]

@st.cache_data(ttl=300)
def obtener_mep():
    try:
        r = requests.get("https://criptoya.com/api/dolar", timeout=5)
        if r.status_code == 200:
            return float(r.json()["mep"]["al30"]["ci"])
    except: 
        pass
    return 1431.0

mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=obtener_mep(), step=1.0)

st.sidebar.header("🕹️ 2. Seleccionar Módulo")
modulo = st.sidebar.selectbox(
    "¿Qué herramienta deseas utilizar?", 
    [
        "🔮 Algoritmo Black-Litterman (Opinión + Presets)",
        "⚡ Diagnóstico y Rebalanceo de Cartera Actual",
        "📈 Frontera Eficiente (Markowitz)", 
        "🎯 Cartera por Retorno Objetivo (Multialternativa)", 
        "💰 Simulador de Retiro Automatizado",
        "Neuro-Analisis de Portafolio"
    ]
)

# --- FUNCIONES AUXILIARES ---
def calcular_metricas(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def safe_str(texto):
    remplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N", "%": " por ciento", "•": "-", 
        "🟢": "COMPRAR", "🔴": "VENDER", "🟡": "MANTENER",
        "📊": "", "🎯": "", "💰": "", "📈": "", "📉": ""
    }
    origen = str(texto)
    for k, v in remplazos.items():
        origen = origen.replace(k, v)
    return origen.encode('latin-1', 'ignore').decode('latin-1')

def generar_reporte_pdf(datos_cartera, nombre_plan, retorno, riesgo):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, safe_str(f"Reporte Financiero: {nombre_plan}"), ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, safe_str(f"Fecha de emision: {date.today().strftime('%d/%m/%Y')}"), ln=True)
    pdf.cell(0, 8, safe_str(f"Retorno Anual Esperado (USD): {retorno:.2%}"), ln=True)
    pdf.cell(0, 8, safe_str(f"Volatilidad Anual (Riesgo): {riesgo:.2%}"), ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, safe_str("Composicion del Portafolio:"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    for _, row in datos_cartera.iterrows():
        linea = f" - {row['Activo']}: {row['Peso %']:.2f} por ciento"
        pdf.cell(0, 8, safe_str(linea), ln=True)
    return bytes(pdf.output())

# --- DESCARGA Y PROCESAMIENTO ---
if fecha_inicio >= fecha_fin:
    st.warning("⚠️ La fecha de inicio debe ser anterior a la de fin.")
    st.stop()

datos_raw = yf.download(activos, start=fecha_inicio, end=fecha_fin)
if datos_raw.empty or 'Close' not in datos_raw:
    st.warning("⚠️ No se encontraron precios de cierre.")
    st.stop()

df_close = datos_raw['Close']
if isinstance(df_close, pd.DataFrame):
    df_close.columns = df_close.columns.get_level_values(-1) if isinstance(df_close.columns, pd.MultiIndex) else df_close.columns
else:
    df_close = df_close.to_frame(name=activos[0])

datos_usd = df_close.dropna().copy()
for col in datos_usd.columns:
    if col.endswith(".BA"):
        datos_usd[col] = datos_usd[col] / mep_actual

rendimientos = datos_usd.pct_change().dropna()
ret_anuales = rendimientos.mean() * 252
cov_matrix = rendimientos.cov() * 252
num_activos = len(datos_usd.columns)

# --- MÓDULOS ---
if modulo == "🔮 Algoritmo Black-Litterman (Opinión + Presets)":
    st.header("🔮 Optimización de Black-Litterman")
    # ... (resto de tu lógica de Black-Litterman) ...
    st.write("Configura tus opiniones en la barra lateral.")

elif modulo == "⚡ Diagnóstico y Rebalanceo de Cartera Actual":
    st.header("⚡ Diagnóstico y Rebalanceo")
    capital_total = st.number_input("Capital Total (USD):", value=5000)
    # ... (resto de tu lógica de Rebalanceo) ...

elif modulo == "📈 Frontera Eficiente (Markowitz)":
    st.header("📈 Frontera Eficiente")
    # ... (resto de tu lógica de Markowitz) ...

elif modulo == "🎯 Cartera por Retorno Objetivo (Multialternativa)":
    st.header("🎯 Cartera por Retorno Objetivo")
    # ... (resto de tu lógica de Retorno Objetivo) ...

elif modulo == "💰 Simulador de Retiro Automatizado":
    st.header("💰 Simulador de Retiro")
    # ... (resto de tu lógica de Simulador) ...

elif modulo == "Neuro-Analisis de Portafolio":
    st.header("🧠 Oráculo de Comportamiento")
    vol_diaria = rendimientos.std().mean()
    if vol_diaria > 0.02:
        st.error("ESTADO: ALTA ANSIEDAD. Considera ajustar tu exposición.")
    else:
        st.success("ESTADO: MODO ZEN. Cartera estable.")
    umbral = st.slider("Límite de caída tolerable (%)", 5, 50, 15)
    st.write(f"Tu umbral configurado es {umbral}%. Mantén la calma si el mercado fluctúa.")
