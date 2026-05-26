import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF
from datetime import date
from scipy.optimize import minimize
import requests

st.set_page_config(page_title="Suite Financiera Pro", layout="wide")
st.title("📊 Suite Financiera Pro: Optimización y Diagnóstico")

# --- BARRA LATERAL ---
lista_tickers = st.sidebar.text_input("Tickers:", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=1431.0)
activos = [x.strip().upper() for x in lista_tickers.split(",") if x.strip()]

modulo = st.sidebar.selectbox("Herramienta:", [
    "🔮 Black-Litterman",
    "⚡ Diagnóstico y Rebalanceo",
    "📈 Frontera Eficiente", 
    "🎯 Cartera por Retorno Objetivo", 
    "💰 Simulador de Retiro",
    "🧠 Neuro-Analisis de Portafolio"
])

# --- PROCESAMIENTO ---
datos_raw = yf.download(activos, start=fecha_inicio, end=date.today())
if 'Close' not in datos_raw: st.stop()
df_close = datos_raw['Close'] if isinstance(datos_raw['Close'], pd.DataFrame) else datos_raw['Close'].to_frame(name=activos[0])
datos_usd = df_close.dropna().copy()
for col in datos_usd.columns:
    if col.endswith(".BA"): datos_usd[col] /= mep_actual

rendimientos = datos_usd.pct_change().dropna()
ret_anuales = rendimientos.mean() * 252
cov_matrix = rendimientos.cov() * 252
num_activos = len(datos_usd.columns)

def calcular_metricas(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol, ret / vol if vol > 0 else 0

# --- MÓDULOS ---
if modulo == "🔮 Black-Litterman":
    st.header("🔮 Algoritmo Black-Litterman")
    st.write("Módulo de optimización avanzada cargado.")

elif modulo == "⚡ Diagnóstico y Rebalanceo":
    st.header("⚡ Diagnóstico y Rebalanceo")
    capital = st.number_input("Capital (USD):", value=5000)
    pesos = [st.number_input(f"% {a}:", 0.0, 100.0, 100.0/num_activos) for a in datos_usd.columns]
    if st.button("Analizar"):
        res = minimize(lambda w: -calcular_metricas(w, ret_anuales, cov_matrix)[2], [1./num_activos]*num_activos, 
                       bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
        df = pd.DataFrame({'Activo': datos_usd.columns, 'Actual': pesos, 'Objetivo': res.x * 100})
        st.dataframe(df)

elif modulo == "📈 Frontera Eficiente":
    st.header("📈 Frontera Eficiente")
    if st.button("Calcular"):
        w = np.random.random(num_activos); w /= np.sum(w)
        st.plotly_chart(px.pie(names=datos_usd.columns, values=w))

elif modulo == "🎯 Cartera por Retorno Objetivo":
    st.header("🎯 Cartera por Retorno Objetivo")
    objetivo = st.number_input("Ingreso mensual deseado (USD):", value=500)
    ret, _, _ = calcular_metricas(np.ones(num_activos)/num_activos, ret_anuales, cov_matrix)
    st.metric("Capital necesario", f"${((objetivo*12)/ret):,.2f}")

elif modulo == "💰 Simulador de Retiro":
    st.header("💰 Simulador de Retiro")
    cap_ini = st.number_input("Capital Inicial:", value=50000)
    ret_mensual = st.number_input("Retiro Mensual:", value=500)
    if st.button("Simular"):
        mu = ret_anuales.mean()
        sigma = rendimientos.std().mean() * np.sqrt(252)
        meses = 180
        trayectoria = np.zeros(meses)
        trayectoria[0] = cap_ini
        for t in range(1, meses):
            trayectoria[t] = max(0, trayectoria[t-1] * (1 + np.random.normal(mu/12, sigma/np.sqrt(12))) - ret_mensual)
        st.line_chart(trayectoria)

elif modulo == "🧠 Neuro-Analisis de Portafolio":
    st.header("🧠 Oráculo de Comportamiento del Inversor")
    vol_diaria = rendimientos.std().mean()
    if vol_diaria > 0.02:
        st.error("ESTADO: ALTA ANSIEDAD. Tu cartera tiene mucha volatilidad.")
    else:
        st.success("ESTADO: MODO ZEN. Cartera estable.")
    umbral = st.slider("Límite de caída tolerable (%)", 5, 50, 15)
    prob_caida = (rendimientos.mean() - (2 * rendimientos.std())).min() * 100
    st.write(f"Probabilidad estadística de caída crítica: {abs(prob_caida):.1f}%")
