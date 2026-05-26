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
lista_tickers = st.sidebar.text_input("Tickers (separados por coma):", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=1431.0, step=1.0)
activos = [x.strip().upper() for x in lista_tickers.split(",") if x.strip()]

modulo = st.sidebar.selectbox("Herramienta:", [
    "🔮 Algoritmo Black-Litterman",
    "⚡ Diagnóstico y Rebalanceo",
    "📈 Frontera Eficiente", 
    "🎯 Cartera por Retorno Objetivo", 
    "🧠 Neuro-Análisis de Portafolio"
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
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

# --- MÓDULOS ---
if modulo == "🔮 Algoritmo Black-Litterman":
    st.header("🔮 Algoritmo Black-Litterman")
    st.write("Configura tus opiniones para ajustar el portafolio.")

elif modulo == "⚡ Diagnóstico y Rebalanceo":
    st.header("⚡ Diagnóstico y Rebalanceo")
    capital = st.number_input("Capital Total (USD):", value=5000)
    pesos = [st.number_input(f"% Actual {a}:", 0.0, 100.0, 100.0/num_activos) for a in datos_usd.columns]
    if st.button("Analizar"):
        res = minimize(lambda w: -calcular_metricas(w, ret_anuales, cov_matrix)[2], [1./num_activos]*num_activos, 
                       bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
        df_res = pd.DataFrame({'Activo': datos_usd.columns, 'Peso Actual': pesos, 'Peso Objetivo': res.x * 100})
        st.dataframe(df_res)

elif modulo == "📈 Frontera Eficiente":
    st.header("📈 Frontera Eficiente")
    if st.button("Calcular"):
        w = np.random.random(num_activos); w /= np.sum(w)
        df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso': w * 100})
        st.plotly_chart(px.pie(df_w, values='Peso', names='Activo'))

elif modulo == "🎯 Cartera por Retorno Objetivo":
    st.header("🎯 Cartera por Retorno Objetivo")
    objetivo = st.number_input("Ingreso mensual deseado (USD):", value=500)
    res = minimize(lambda w: -calcular_metricas(w, ret_anuales, cov_matrix)[2], [1./num_activos]*num_activos, 
                   bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
    ret_anual, _, _ = calcular_metricas(res.x, ret_anuales, cov_matrix)
    capital_req = (objetivo * 12) / ret_anual if ret_anual > 0 else 0
    st.metric("Capital necesario", f"${capital_req:,.2f}")

elif modulo == "🧠 Neuro-Análisis de Portafolio":
    st.header("🧠 Neuro-Análisis de Portafolio")
    vol = rendimientos.std().mean() * np.sqrt(252)
    st.write(f"Tu volatilidad anual es: {vol:.2%}")
    if vol > 0.3:
        st.error("ESTADO: ALTA ANSIEDAD. Considera activos de refugio.")
    else:
        st.success("ESTADO: MODO ZEN. Tu cartera es estable.")
    umbral = st.slider("¿Tu límite de caída tolerable (%)?", 5, 50, 15)
    st.write(f"Si tu cartera cae más del {umbral}%, el sistema te sugiere cautela.")import streamlit as st
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
lista_tickers = st.sidebar.text_input("Tickers (separados por coma):", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=1431.0, step=1.0)
activos = [x.strip().upper() for x in lista_tickers.split(",") if x.strip()]

modulo = st.sidebar.selectbox("Herramienta:", [
    "🔮 Algoritmo Black-Litterman",
    "⚡ Diagnóstico y Rebalanceo",
    "📈 Frontera Eficiente", 
    "🎯 Cartera por Retorno Objetivo", 
    "🧠 Neuro-Análisis de Portafolio"
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
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

# --- MÓDULOS ---
if modulo == "🔮 Algoritmo Black-Litterman":
    st.header("🔮 Algoritmo Black-Litterman")
    st.write("Configura tus opiniones para ajustar el portafolio.")

elif modulo == "⚡ Diagnóstico y Rebalanceo":
    st.header("⚡ Diagnóstico y Rebalanceo")
    capital = st.number_input("Capital Total (USD):", value=5000)
    pesos = [st.number_input(f"% Actual {a}:", 0.0, 100.0, 100.0/num_activos) for a in datos_usd.columns]
    if st.button("Analizar"):
        res = minimize(lambda w: -calcular_metricas(w, ret_anuales, cov_matrix)[2], [1./num_activos]*num_activos, 
                       bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
        df_res = pd.DataFrame({'Activo': datos_usd.columns, 'Peso Actual': pesos, 'Peso Objetivo': res.x * 100})
        st.dataframe(df_res)

elif modulo == "📈 Frontera Eficiente":
    st.header("📈 Frontera Eficiente")
    if st.button("Calcular"):
        w = np.random.random(num_activos); w /= np.sum(w)
        df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso': w * 100})
        st.plotly_chart(px.pie(df_w, values='Peso', names='Activo'))

elif modulo == "🎯 Cartera por Retorno Objetivo":
    st.header("🎯 Cartera por Retorno Objetivo")
    objetivo = st.number_input("Ingreso mensual deseado (USD):", value=500)
    res = minimize(lambda w: -calcular_metricas(w, ret_anuales, cov_matrix)[2], [1./num_activos]*num_activos, 
                   bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
    ret_anual, _, _ = calcular_metricas(res.x, ret_anuales, cov_matrix)
    capital_req = (objetivo * 12) / ret_anual if ret_anual > 0 else 0
    st.metric("Capital necesario", f"${capital_req:,.2f}")

elif modulo == "🧠 Neuro-Análisis de Portafolio":
    st.header("🧠 Neuro-Análisis de Portafolio")
    vol = rendimientos.std().mean() * np.sqrt(252)
    st.write(f"Tu volatilidad anual es: {vol:.2%}")
    if vol > 0.3:
        st.error("ESTADO: ALTA ANSIEDAD. Considera activos de refugio.")
    else:
        st.success("ESTADO: MODO ZEN. Tu cartera es estable.")
    umbral = st.slider("¿Tu límite de caída tolerable (%)?", 5, 50, 15)
    st.write(f"Si tu cartera cae más del {umbral}%, el sistema te sugiere cautela.")
