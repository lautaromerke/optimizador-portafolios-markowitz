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

st.title("📊 Plataforma Integral de Optimización y Planificación")
st.markdown("""
Bienvenido a tu suite financiera adaptada al mercado argentino. Los activos en pesos locales 
se detectan y **se dolarizan automáticamente al MEP** para unificar el análisis.
""")

# --- BARRA LATERAL: CONFIGURACIÓN BASE DE DATOS ---
st.sidebar.header("🔧 1. Datos del Portafolio")
lista_tickers = st.sidebar.text_input("Tickers (ej: AAPL, GGAL.BA, MSFT, YPFD.BA):", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
fecha_fin = st.sidebar.date_input("Histórico hasta:", pd.to_datetime("2026-05-25"))

activos = [x.strip().upper() for x in lista_tickers.split(",")]

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

st.sidebar.header("💵 Tipo de Cambio")
mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=mep_sincronizado, step=1.0)

# --- MENÚ DE SELECCIÓN DE MÓDULO ---
st.sidebar.header("🕹️ 2. Elegir Herramienta")
modulo_seleccionado = st.sidebar.radio(
    "Selecciona qué deseas calcular:",
    ["📈 Optimización Clásica (Markowitz)", "🎯 Cartera por Retorno Objetivo", "💰 Simulador de Retiro (Montecarlo)"]
)

# --- FUNCIONES MATEMÁTICAS AUXILIARES ---
def calcular_metricas_cartera(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return np.array([ret, vol, sharpe])

def minimizar_volatilidad(weights, ret_anuales, cov_matrix):
    return calcular_metricas_cartera(weights, ret_anuales, cov_matrix)[1]

# --- BLOQUE CENTRAL DE PROCESAMIENTO DE DATOS ---
try:
    if fecha_inicio >= fecha_fin:
        st.warning("⚠️ **Error en el rango:** La fecha de inicio no puede ser posterior o igual a la de fin.")
        st.stop()

    # Descarga limpia de Yahoo Finance
    datos_raw = yf.download(activos, start=fecha_inicio, end=fecha_fin)
    if datos_raw.empty or 'Close' not in datos_raw:
        st.warning("⚠️ No se encontraron precios de cierre para los tickers ingresados.")
        st.stop()
    
    df_close = datos_raw['Close']
    
    # Blindaje contra MultiIndex (Evita el error 'Columns must be same length as key')
    if isinstance(df_close, pd.DataFrame):
        if isinstance(df_close.columns, pd.MultiIndex):
            df_close.columns = df_close.columns.get_level_values(-1)
    else:
        df_close = df_close.to_frame(name=activos[0])
        
    datos_usd = df_close.dropna().copy()
    
    # Dolarización dinámica (.BA)
    for col in datos_usd.columns:
        if col.endswith(".BA"):
            datos_usd[col] = datos_usd[col] / mep_actual
    
    rendimientos = datos_usd.pct_change().dropna()
    
    if rendimientos.empty:
        st.warning("⚠️ No hay suficientes datos coincidentes tras limpiar las filas vacías.")
        st.stop()

    ret_anuales = rendimientos.mean() * 252
    cov_matrix = rendimientos.cov() * 252
    num_activos = len(datos_usd.columns)

    # --- MÓDULO 1: MARKOWITZ CLÁSICO ---
    if modulo_seleccionado == "📈 Optimización Clásica (Markowitz)":
        st.header("📈 Optimización Clásica de Portafolios (Markowitz)")
        st.write("Simulación de Montecarlo para detectar las carteras eficientes en USD.")
        
        num_portafolios = st.slider("Número de portafolios a simular:", 5000, 25000, 15000, step=5000)
        
        if st.button("🚀 Calcular Frontera Eficiente"):
            results = np.zeros((3, num_portafolios))
            w_list = []
            
            for i in range(num_portafolios):
                w = np.random.random(num_activos)
                w /= np.sum(w)
                w_list.append(w)
                res = calcular_metricas_cartera(w, ret_anuales, cov_matrix)
                results[0, i] = res[0]
                results[1, i] = res[1]
                results[2, i] = res[2]

            idx_sharpe = results[2].argmax()
            idx_min_vol = results[1].argmin()
            
            c1, c2 = st.columns(2)
            with c1:
                st.success("### 🎯 Cartera Máx. Sharpe (USD)")
                st.metric("Retorno Esperado", f"{results[0, idx_sharpe]:.2%}")
                st.metric("Volatilidad (Riesgo)", f"{results[1, idx_sharpe]:.2%}")
                df_sharpe = pd.DataFrame({'Activo': datos_usd.columns, 'Asignación (%)': w_list[idx_sharpe] * 100})
                st.dataframe(df_sharpe.style.format({'Asignación (%)': '{:.2f}%'}), use_container_width=True)
                
            with c2:
                st.info("### 🛡️ Cartera Mínima Volatilidad (USD)")
                st.metric("Retorno Esperado", f"{results[0, idx_min_vol]:.2%}")
                st.metric("Volatilidad (Riesgo)", f"{results[1, idx_min_vol]:.2%}")
                df_vol = pd.DataFrame({'Activo': datos_usd.columns, 'Asignación (%)': w_list[idx_min_vol] * 100})
                st.dataframe(df_vol.style.format({'Asignación (%)': '{:.2f}%'}), use_container_width=True)
