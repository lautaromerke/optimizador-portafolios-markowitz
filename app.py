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
import io

# 1. Configuración de la página web
st.set_page_config(page_title="Portfolio Intelligence Suite", layout="wide")

st.title("📊 Suite Financiera: Optimización y Reportes")

# --- BARRA LATERAL ---
st.sidebar.header("🔧 1. Parámetros de Mercado")
lista_tickers = st.sidebar.text_input("Tickers:", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
fecha_fin = st.sidebar.date_input("Histórico hasta:", pd.to_datetime("2026-05-25"))
activos = [x.strip().upper() for x in lista_tickers.split(",")]

@st.cache_data(ttl=300)
def obtener_mep():
    try:
        r = requests.get("https://criptoya.com/api/dolar", timeout=5)
        return float(r.json()["mep"]["al30"]["ci"])
    except: return 1431.0

mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=obtener_mep(), step=1.0)

st.sidebar.header("🕹️ 2. Seleccionar Módulo")
modulo = st.sidebar.selectbox("Herramienta:", ["📈 Markowitz", "🎯 Retorno Objetivo", "💰 Simulador de Retiro"])

# --- FUNCIONES AUXILIARES ---
def calcular_metricas(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def crear_pdf(datos_cartera, nombre_plan, retorno, riesgo):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, f"Reporte de Inversion: {nombre_plan}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Fecha de reporte: {date.today()}", ln=True)
    pdf.cell(0, 10, f"Retorno Anual Esperado: {retorno:.2%}", ln=True)
    pdf.cell(0, 10, f"Riesgo Anual (Volatilidad): {riesgo:.2%}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Composicion del Portafolio:", ln=True)
    pdf.set_font("Arial", "", 12)
    for index, row in datos_cartera.iterrows():
        pdf.cell(0, 10, f"- {row['Activo']}: {row['Peso %']:.2f}%", ln=True)
    return pdf.output()

# --- PROCESAMIENTO DE DATOS ---
datos_raw = yf.download(activos, start=fecha_inicio, end=fecha_fin)['Close']
if isinstance(datos_raw, pd.DataFrame) and isinstance(datos_raw.columns, pd.MultiIndex):
    datos_raw.columns = datos_raw.columns.get_level_values(-1)
datos_usd = datos_raw.dropna().copy()
if isinstance(datos_usd, pd.Series): datos_usd = datos_usd.to_frame(name=activos[0])

for col in datos_usd.columns:
    if col.endswith(".BA"): datos_usd[col] = datos_usd[col] / mep_actual

rendimientos = datos_usd.pct_change().dropna()
ret_anuales = rendimientos.mean() * 252
cov_matrix = rendimientos.cov() * 252
num_activos = len(datos_usd.columns)

# --- MÓDULO 1: MARKOWITZ ---
if modulo == "📈 Markowitz":
    st.header("📈 Frontera Eficiente")
    n_sims = st.slider("Simulaciones:", 5000, 20000, 10000)
    
    if st.button("🚀 Optimizar"):
        results = np.zeros((3, n_sims))
        w_list = []
        for i in range(n_sims):
            w = np.random.random(num_activos)
            w /= np.sum(w)
            w_list.append(w)
            r, v, s = calcular_metricas(w, ret_anuales, cov_matrix)
            results[0,i], results[1,i], results[2,i] = r, v, s
        
        idx = results[2].argmax()
        pesos = w_list[idx]
        df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': pesos * 100})
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("### 🍰 Distribución de Capital")
            fig_pie = px.pie(df_w, values='Peso %', names='Activo', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.success(f"**Retorno:** {results[0,idx]:.2%} | **Riesgo:** {results[1,idx]:.2%}")
            st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}))
            
            pdf_bytes = crear_pdf(df_w, "Cartera Optima Markowitz", results[0,idx], results[1,idx])
            st.download_button("📩 Descargar Reporte PDF", data=pdf_bytes, file_name="reporte_markowitz.pdf", mime="application/pdf")

# --- MÓDULO 2: RETORNO OBJETIVO ---
elif modulo == "🎯 Retorno Objetivo":
    st.header("🎯 Cartera por Objetivo")
    meta = st.slider("Meta Anual (USD %):", 5.0, 30.0, 12.0) / 100
    
    if st.button("🎯 Calcular"):
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(ret_anuales * w) - meta})
        res = minimize(lambda w: calcular_metricas(w, ret_anuales, cov_matrix)[1], [1./num_activos]*num_activos, bounds=[(0,1)]*num_activos, constraints=cons)
        
        if res.success:
            r, v, s = calcular_metricas(res.x, ret_anuales, cov_matrix)
            df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': res.x * 100})
            
            c1, c2 = st.columns(2)
            with c1:
                fig_pie = px.pie(df_w, values='Peso %', names='Activo', hole=0.4)
                st.plotly_chart(fig_pie)
            with c2:
                st.info(f"**Riesgo Mínimo:** {v:.2%}")
                st.dataframe(df_w)
                pdf_bytes = crear_pdf(df_w, "Cartera Retorno Objetivo", r, v)
                st.download_button("📩 Descargar Reporte PDF", data=pdf_bytes, file_name="reporte_objetivo.pdf", mime="application/pdf")
        else: st.error("Meta inalcanzable.")

# --- MÓDULO 3: SIMULADOR ---
elif modulo == "💰 Simulador de Retiro":
    st.header("💰 Supervivencia de Capital")
    cap = st.number_input("Capital (USD):", value=10000)
    ret = st.number_input("Retiro Mensual (USD):", value=400)
    
    if st.button("⏳ Simular"):
        # Lógica simplificada de montecarlo para el simulador
        mu, sigma = ret_anuales.mean(), rendimientos.std().mean() * np.sqrt(252)
        sims = np.zeros((120, 100)) # 10 años
        sims[0] = cap
        for s in range(100):
            for t in range(1, 120):
                sims[t,s] = max(0, sims[t-1,s] * (1 + np.random.normal(mu/12, sigma/np.sqrt(12))) - ret)
        
        st.line_chart(sims)
        prob = np.sum(sims[-1] > 0) / 100
        st.metric("Probabilidad de éxito", f"{prob:.1%}")
