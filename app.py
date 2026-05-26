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
    except: pass
    return 1431.0

mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=obtener_mep_en_vivo())

# --- FUNCIONES MATEMÁTICAS DE OPTIMIZACIÓN ---
def get_ret_vol_sharpe(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol
    return np.array([ret, vol, sharpe])

def check_sum(weights):
    return np.sum(weights) - 1

# Optimización para Retorno Objetivo
def minimize_volatility(weights, ret_anuales, cov_matrix):
    return get_ret_vol_sharpe(weights, ret_anuales, cov_matrix)[1]

if st.sidebar.button("🚀 Ejecutar Planificación Integral"):
    with st.spinner("Procesando modelos cuantitativos y proyecciones..."):
        try:
            # --- DESCARGA Y DOLARIZACIÓN ---
            datos_raw = yf.download(activos, start=fecha_inicio)['Close'].dropna()
            if isinstance(datos_raw, pd.Series): datos_raw = datos_raw.to_frame(name=activos[0])
            
            datos_usd = datos_raw.copy()
            for col in datos_usd.columns:
                if col.endswith(".BA"): datos_usd[col] = datos_usd[col] / mep_actual
            
            rendimientos = datos_usd.pct_change().dropna()
            ret_anuales = rendimientos.mean() * 252
            cov_matrix = rendimientos.cov() * 252

            # --- MONTECARLO MARKOWITZ ---
            results = np.zeros((3, num_portafolios))
            w_list = []
            for i in range(num_portafolios):
                w = np.random.random(len(activos))
                w /= np.sum(w)
                w_list.append(w)
                res = get_ret_vol_sharpe(w, ret_anuales, cov_matrix)
                results[0,i], results[1,i], results[2,i] = res[0], res[1], res[2]

            idx_sharpe = results[2].argmax()
            best_w = w_list[idx_sharpe]

            # --- OPTIMIZACIÓN POR OBJETIVO (SCIPY) ---
            cons = ({'type':'eq', 'fun': check_sum},
                    {'type':'eq', 'fun': lambda w: np.sum(ret_anuales * w) -
