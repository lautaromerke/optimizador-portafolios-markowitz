import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import scipy.stats as stats
from scipy.optimize import minimize
import requests

# 1. Configuración de la página web
st.set_page_config(page_title="Plataforma de Planificación Financiera", layout="wide")

st.title("📊 Planificador Financiero Avanzado: Markowitz + Montecarlo")
st.markdown("""
Esta plataforma integra la **Teoría de Markowitz** con **Optimización por Objetivos** y **Simulaciones de Montecarlo** para proyectar la supervivencia de capital a largo plazo.
""")

st.sidebar.header("🔧 1. Configuración del Portafolio")
lista_tickers = st.sidebar.text_input("Tickers (ej: AAPL, GGAL.BA, MSFT, YPFD.BA):", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
fecha_fin = st.sidebar.date_input("Histórico hasta:", pd.to_datetime("2026-05-25"))
num_portafolios = st.sidebar.slider("Simulaciones Markowitz:", 5000, 25000, 15000, step=5000)

activos = [x.strip().upper() for x in lista_tickers.split(",")]

st.sidebar.header("🎯 2. Objetivo de Inversión")
retorno_objetivo_pct = st.sidebar.slider("Retorno Anual Objetivo (USD %):", 5.0, 30.0, 12.0, step=0.5)

st.sidebar.header("💰 3. Simulador de Retiro")
capital_inicial = st.sidebar.number_input("Capital Inicial (USD):", min_value=1000, value=10000, step=1000)
retiro_mensual = st.sidebar.number_input("Retiro Mensual Deseado (USD):", min_value=0, value=500, step=50)
anios_proyeccion = st.sidebar.slider("Años de Proyección:", 5, 30, 15)

# --- CONSULTA AUTOMÁTICA DE MEP ---
@st.cache_data(ttl=300)
def obtener_mep_en_vivo():
    try:
        r = requests.get("https://criptoya.com/api/dolar", timeout=5)
        if r.status_code == 200:
            return float(r.json()["mep"]["al30"]["ci"])
    except: 
        pass
    return 1431.0

mep_sincronizado = obtener_mep_en_vivo()

st.sidebar.header("💵 Tipo de Cambio Automático")
mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=mep_sincronizado, step=1.0)

# --- FUNCIONES MATEMÁTICAS DE OPTIMIZACIÓN ---
def calcular_metricas_cartera(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return np.array([ret, vol, sharpe])

def minimizar_volatilidad(weights, ret_anuales, cov_matrix):
    return calcular_metricas_cartera(weights, ret_anuales, cov_matrix)[1]

# Botón para ejecutar todo el modelo
if st.sidebar.button("🚀 Ejecutar Planificación Integral"):
    with st.spinner("Procesando modelos cuantitativos y proyecciones..."):
        try:
            if fecha_inicio >= fecha_fin:
                st.warning("⚠️ **Error en el rango:** La fecha de inicio no puede ser posterior o igual a la de fin.")
                st.stop()

            # --- DESCARGA SEGURO CON MULTI-INDEX HANDLING ---
            datos_raw = yf.download(activos, start=fecha_inicio, end=fecha_fin)
            if datos_raw.empty or 'Close' not in datos_raw:
                st.warning("⚠️ No se encontraron precios de cierre para esos tickers.")
                st.stop()
            
            df_close = datos_raw['Close']
            if isinstance(df_close, pd.Series):
                df_close = df_close.to_frame(name=activos[0])
                
            datos_usd = df_close.dropna().copy()
            
            # --- DOLARIZACIÓN DINÁMICA ---
            for col in datos_usd.columns:
                if col.endswith(".BA"):
                    datos_usd[col] = datos_usd[col] / mep_actual
            
            rendimientos = datos_usd.pct_change().dropna()
            if rendimientos.empty:
                st.warning("⚠️ Datos insuficientes tras la limpieza de filas vacías.")
                st.stop()

            ret_anuales = rendimientos.mean() * 252
            cov_matrix = rendimientos.cov() * 252

            # --- SIMULACIÓN MONTECARLO MARKOWITZ ---
            num_activos = len(datos_usd.columns)
            results = np.zeros((3, num_portafolios))
            w_list = []
            
            for i in range(num_portafolios):
                w = np.random.random(num_activos)
                w /= np.sum(w)
                w_list.append(w)
                res = calcular_metricas_cartera(w, ret_anuales, cov_matrix)
                results[0, i] = res[0]
