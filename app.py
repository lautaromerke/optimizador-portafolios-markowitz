import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
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
