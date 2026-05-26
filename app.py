import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
from datetime import date
from scipy.optimize import minimize
import requests

# 1. Configuración de la plataforma
st.set_page_config(page_title="Portfolio Intelligence Suite", layout="wide")

st.title("📊 Suite Financiera: Optimización y Reportes")
st.markdown("""
Esta plataforma te permite analizar activos de forma modular. 
Los activos locales (.BA) se detectan y **se dolarizan automáticamente usando el Dólar MEP**.
""")

# --- BARRA LATERAL ---
st.sidebar.header("🔧 1. Parámetros de Mercado")
lista_tickers = st.sidebar.text_input("Tickers (separados por coma):", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
fecha_fin = st.sidebar.date_input("Histórico hasta:", pd.to_datetime("2026-05-25"))
activos = [x.strip().upper() for x in lista_tickers.split(",")]

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
    ["📈 Frontera Eficiente (Markowitz)", "🎯 Cartera por Retorno Objetivo", "💰 Simulador de Retiro"]
)

# --- FUNCIONES AUXILIARES ---
def calcular_metricas(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def safe_str(texto):
    # Remueve acentos y caracteres especiales no soportados por las fuentes por defecto de FPDF
    return str(texto).encode('latin-1', 'ignore').decode('latin-1')

def generar_reporte_pdf(datos_cartera, nombre_plan, retorno, riesgo):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, safe_str(f"Reporte Financiero: {nombre_plan}"), ln=True, align="C")
    pdf.ln(8)
    
    # Info general
