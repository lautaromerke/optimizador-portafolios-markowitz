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

# --- FUNCIONES AUXAILIARES ---
def calcular_metricas(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def generar_reporte_pdf(datos_cartera, nombre_plan, retorno, riesgo):
    pdf = FPDF()
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, f"Reporte Financiero: {nombre_plan}", ln=True, align="C")
    pdf.ln(8)
    
    # Info general
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Fecha de emision: {date.today().strftime('%d/%m/%Y')}", ln=True)
    pdf.cell(0, 8, f"Retorno Anual Esperado (USD): {retorno:.2%}", ln=True)
    pdf.cell(0, 8, f"Volatilidad Anual (Riesgo): {riesgo:.2%}", ln=True)
    pdf.ln(6)
    
    # Tabla de posiciones
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Composicion Eficiente del Portafolio:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    for _, row in datos_cartera.iterrows():
        pdf.cell(0, 8, f"  • {row['Activo']}: {row['Peso %']:.2f}%", ln=True)
        
    # Retornar los bytes puros de manera segura para Streamlit
    return bytes(pdf.output())

# --- DESCARGA Y PROCESAMIENTO DE DATOS FINANCIEROS ---
if fecha_inicio >= fecha_fin:
    st.warning("⚠️ La fecha de inicio debe ser anterior a la de fin.")
    st.stop()

datos_raw = yf.download(activos, start=fecha_inicio, end=fecha_fin)
if datos_raw.empty or 'Close' not in datos_raw:
    st.warning("⚠️ No se encontraron precios de cierre para los tickers ingresados.")
    st.stop()

df_close = datos_raw['Close']

# Solución definitiva al índice de columnas de Yahoo Finance (Aplanado)
if isinstance(df_close, pd.DataFrame):
    if isinstance(df_close.columns, pd.MultiIndex):
        df_close.columns = df_close.columns.get_level_values(-1)
else:
    df_close = df_close.to_frame(name=activos[0])

datos_usd = df_close.dropna().copy()

# Dolarización de activos argentinos
for col in datos_usd.columns:
    if col.endswith(".BA"):
        datos_usd[col] = datos_usd[col] / mep_actual

rendimientos = datos_usd.pct_change().dropna()
if rendimientos.empty:
    st.warning("⚠️ Datos históricos insuficientes tras realizar la limpieza.")
    st.stop()

ret_anuales = rendimientos.mean() * 252
cov_matrix = rendimientos.cov() * 252
num_activos = len(datos_usd.columns)


# =========================================================================
# MÓDULO 1: MARKOWITZ CLÁSICO
# =========================================================================
if modulo == "📈 Frontera Eficiente (Markowitz)":
    st.header("📈 Optimización Clásica (Frontera de Markowitz)")
    n_sims = st.slider("Cantidad de simulaciones de carteras:", 5000, 20000, 10000, step=5000)
    
    if st.button("🚀 Calcular Portafolios"):
        results = np.zeros((3, n_sims))
        w_list = []
        for i in range(n_sims):
            w = np.random.random(num_activos)
            w /= np.sum(w)
            w_list.append(w)
            r, v, s = calcular_metricas(w, ret_anuales, cov_matrix)
            results[0, i] = r
            results[1, i] = v
            results[2, i] = s
        
        idx_sharpe = results[2].argmax()
        df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': w_list[idx_sharpe] * 100})
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("### 🍰 Ponderación de la Cartera Óptima")
            fig_pie = px.pie(df_w, values='Peso %', names='Activo', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.success("### 🎯 Métricas Máximo Sharpe")
            st.metric("Retorno Esperado (USD)", f"{results[0, idx_sharpe]:.2%}")
            st.metric("Volatilidad (Riesgo)", f"{results[1, idx_sharpe]:.2%}")
            st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)
            
            # Generación segura del reporte PDF
            pdf_data = generar_reporte_pdf(df_w, "Maximo Sharpe (Markowitz)", results[0, idx_sharpe], results[1, idx_sharpe])
            st.download_button(
                label="📩 Descargar Reporte PDF", 
                data=pdf_data, 
                file_name="reporte_markowitz.pdf", 
                mime="application/pdf"
            )


# =========================================================================
# MÓDULO 2: RETORNO OBJETIVO
# =========================================================================
elif modulo == "🎯 Cartera por Retorno Objetivo":
    st.header("🎯 Optimización por Objetivo de Retorno")
    meta = st.slider("Introduce tu Retorno Anual Objetivo (USD %):", 5.0, 30.0, 12.0, step=0.5) / 100
    
    if st.button("🎯 Resolver Combinación Óptima"):
        restricciones = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.sum(ret_anuales * w) - meta}
        )
        limites = tuple((0.0, 1.0) for _ in range(num_activos))
        iniciales = [1.0 / num_activos] * num_activos
        
        res = minimize(
            lambda w: calcular_metricas(w, ret_anuales, cov_matrix)[1], 
            iniciales, 
            bounds=limites, 
            constraints=restricciones
        )
        
        if res.success:
            r, v, s = calcular_metricas(res.x, ret_anuales, cov_matrix)
            df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': res.x * 100})
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("### 🍰 Distribución de Activos")
                fig_pie = px.pie(df_w, values='Peso %', names='Activo', hole=0.4,
                                 color_discrete_sequence=px.colors.qualitative.Dark2)
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.info("### Métricas Alcanzadas")
                st.metric("Retorno Objetivo", f"{r:.2%}")
                st.metric("Riesgo Mínimo", f"{v:.2%}")
                st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)
                
                pdf_data = generar_reporte_pdf(df_w, "Retorno Objetivo Minimizado", r, v)
                st.download_button(
                    label="📩 Descargar Reporte PDF", 
                    data=pdf_data, 
                    file_name="reporte_objetivo.pdf", 
                    mime="application/pdf"
                )
        else:
            st.error("⚠️ Meta inalcanzable con el set de activos actual.")


# =========================================================================
# MÓDULO 3: SIMULADOR DE RETIROS
# =========================================================================
elif modulo == "💰 Simulador de Retiro":
    st.header("💰 Simulador de Supervivencia de Capital")
    c_sim1, c_sim2 = st.columns(2)
    capital_inicial = c_sim1.number_input("Capital Inicial (USD):", min_value=1000, value=10000, step=1000)
    retiro_mensual = c_sim2.number_input("Retiro Mensual (USD):", min_value=0, value=400, step=50)
    
    if st.button("⏳ Correr Análisis"):
        mu, sigma = ret_anuales.mean(), rendimientos.std().mean() * np.sqrt(252)
        simulations = np.zeros((120, 100))
        simulations[0] = capital_inicial
        
        for s in range(100):
            for t in range(1, 120):
                ret_m = np.random.normal(mu / 12, sigma / np.sqrt(12))
                nuevo_c = simulations[t-1, s] * (1 + ret_m) - retiro_mensual
                simulations[t, s] = max(0, nuevo_c)
        
        st.line_chart(simulations)
        prob = np.sum(simulations[-1] > 0) / 100
        st.metric("Probabilidad de éxito del fondo", f"{prob:.1%}")
