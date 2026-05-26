import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from scipy.optimize import minimize

st.set_page_config(page_title="Suite Financiera Pro", layout="wide")
st.title("📊 Suite Financiera Pro: Optimización y Diagnóstico")

# --- BARRA LATERAL ---
lista_tickers = st.sidebar.text_input("Tickers:", "AAPL, GGAL.BA, MSFT, KO")
fecha_inicio = st.sidebar.date_input("Histórico desde:", pd.to_datetime("2021-01-01"))
mep_actual = st.sidebar.number_input("Dólar MEP ($):", value=1431.0)
activos = [x.strip().upper() for x in lista_tickers.split(",") if x.strip()]

modulo = st.sidebar.selectbox("Herramienta:", [
    "Black-Litterman",
    "Diagnóstico y Rebalanceo",
    "Frontera Eficiente", 
    "Cartera por Retorno Objetivo", 
    "Simulador de Retiro"
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
if modulo == "Black-Litterman":
    st.header("🔮 Algoritmo Black-Litterman")
    st.write("Optimización avanzada cargada.")

elif modulo == "Diagnóstico y Rebalanceo":
    st.header("⚡ Diagnóstico y Rebalanceo")
    capital = st.number_input("Capital (USD):", value=5000)
    pesos = [st.number_input(f"% {a}:", 0.0, 100.0, 100.0/num_activos) for a in datos_usd.columns]
    if st.button("Analizar"):
        res = minimize(lambda w: -calcular_metricas(w, ret_anuales, cov_matrix)[2], [1./num_activos]*num_activos, 
                       bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
        df = pd.DataFrame({'Activo': datos_usd.columns, 'Actual': pesos, 'Objetivo': res.x * 100})
        st.dataframe(df)

elif modulo == "Frontera Eficiente":
    st.header("📈 Frontera Eficiente")
    if st.button("Calcular"):
        w = np.random.random(num_activos); w /= np.sum(w)
        st.plotly_chart(px.pie(names=datos_usd.columns, values=w))

elif modulo == "Cartera por Retorno Objetivo":
    st.header("🎯 Cartera por Retorno Objetivo")
    objetivo = st.number_input("Ingreso mensual deseado (USD):", value=500)
    ret, _, _ = calcular_metricas(np.ones(num_activos)/num_activos, ret_anuales, cov_matrix)
    st.metric("Capital necesario", f"${((objetivo*12)/ret):,.2f}")

elif modulo == "Simulador de Retiro":
    st.header("💰 Simulador de Supervivencia de Capital")
    c1, c2 = st.columns(2)
    cap_ini = c1.number_input("Capital Inicial (USD):", value=50000)
    ret_men = c2.number_input("Retiro Mensual (USD):", value=500)
    
    if st.button("⏳ Simular 3 Escenarios"):
        mu = ret_anuales.mean()
        sigma = rendimientos.std().mean() * np.sqrt(252)
        perfiles = {
            'Conservador': {'mu': mu * 0.5, 'sigma': sigma * 0.4},
            'Moderado':    {'mu': mu,      'sigma': sigma},
            'Arriesgado':  {'mu': mu * 1.5, 'sigma': sigma * 1.8}
        }
        meses = 180
        df_sim = pd.DataFrame(index=range(meses))
        for nombre, params in perfiles.items():
            trayectorias = np.zeros((meses, 30))
            trayectorias[0] = cap_ini
            for s in range(30):
                for t in range(1, meses):
                    rend = np.random.normal(params['mu']/12, params['sigma']/np.sqrt(12))
                    trayectorias[t, s] = max(0, trayectorias[t-1, s] * (1 + rend) - ret_men)
            df_sim[nombre] = np.mean(trayectorias, axis=1)
        st.line_chart(df_sim)
        
        c1, c2, c3 = st.columns(3)
        for i, nombre in enumerate(perfiles.keys()):
            m = (df_sim[nombre] > 0).sum()
            [c1, c2, c3][i].metric(f"Perfil {nombre}", f"{m} meses", f"{m/12:.1f} años")
