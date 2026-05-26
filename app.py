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

        st.markdown("---")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        scatter = ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', s=10, alpha=0.4)
        fig.colorbar(scatter, ax=ax, label='Ratio de Sharpe')
        ax.scatter(results[1, idx_sharpe], results[0, idx_sharpe], color='red', marker='*', s=200, label='Máx Sharpe')
        ax.scatter(results[1, idx_min_vol], results[0, idx_min_vol], color='blue', marker='*', s=200, label='Mín Vol')
        ax.set_xlabel('Volatilidad Anualizada')
        ax.set_ylabel('Retorno Anual Esperado (USD)')
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)


# --- MÓDULO 2: RETORNO OBJETIVO ---
elif modulo_seleccionado == "🎯 Cartera por Retorno Objetivo":
    st.header("🎯 Optimización por Objetivo de Retorno")
    st.write("Encuentra matemáticamente la combinación de menor riesgo posible para alcanzar tu meta.")
    
    retorno_objetivo_pct = st.slider("Introduce tu Retorno Anual Deseado (USD %):", 5.0, 30.0, 12.0, step=0.5)
    
    if st.button("🎯 Calcular Cartera Objetivo"):
        restricciones = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.sum(ret_anuales * w) - (retorno_objetivo_pct / 100.0)}
        )
        limites = tuple((0.0, 1.0) for _ in range(num_activos))
        iniciales = [1.0 / num_activos] * num_activos
        
        opt_res = minimize(minimizar_volatilidad, iniciales, args=(ret_anuales, cov_matrix), method='SLSQP', bounds=limites, constraints=restricciones)
        
        if opt_res.success:
            res_obj = calcular_metricas_cartera(opt_res.x, ret_anuales, cov_matrix)
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("### Métricas de la Cartera Asignada")
                st.metric("Retorno Logrado (USD)", f"{res_obj[0]:.2%}")
                st.metric("Riesgo Mínimo Eficiente", f"{res_obj[1]:.2%}")
            with col2:
                st.write("### Composición Eficiente (%)")
                df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': opt_res.x * 100})
                st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)
        else:
            st.error("⚠️ **Meta inalcanzable:** Ninguna combinación matemática de estos activos logra dar un rendimiento tan alto. Intentá bajando el objetivo o sumando activos con mayor crecimiento.")


# --- MÓDULO 3: SIMULADOR DE RETIROS ---
elif modulo_seleccionado == "💰 Simulador de Retiro (Montecarlo)":
    st.header("💰 Simulador de Supervivencia de Capital")
    st.write("Simulá el comportamiento de tus fondos en el tiempo retirando dinero mensualmente.")
    
    c_sim1, c_sim2, c_sim3 = st.columns(3)
    capital_inicial = c_sim1.number_input("Capital Inicial (USD):", min_value=1000, value=10000, step=1000)
    retiro_mensual = c_sim2.number_input("Retiro Mensual (USD):", min_value=0, value=400, step=50)
    anios_proyeccion = c_sim3.slider("Años a proyectar:", 5, 25, 12)
    
    if st.button("⏳ Correr Simulación de Supervivencia"):
        pesos_base = np.array([1.0 / num_activos] * num_activos)
        res_base = calcular_metricas_cartera(pesos_base, ret_anuales, cov_matrix)
        mu, sigma = res_base[0], res_base[1]
        
        n_sims = 200
        n_meses = anios_proyeccion * 12
        simulaciones = np.zeros((n_meses, n_sims))
        simulaciones[0] = capital_inicial
        
        for s in range(n_sims):
            for t in range(1, n_meses):
                ret_mensual = np.random.normal(mu / 12, sigma / np.sqrt(12))
                nuevo_cap = simulaciones[t-1, s] * (1.0 + ret_mensual) - retiro_mensual
                simulaciones[t, s] = max(0.0, nuevo_cap)

        fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
        ax_sim.plot(simulaciones, color='gray', alpha=0.07)
        ax_sim.plot(np.percentile(simulaciones, 50, axis=1), color='blue', linewidth=2.5, label='Escenario Mediano')
        ax_sim.plot(np.percentile(simulaciones, 10, axis=1), color='red', linestyle='--', linewidth=2, label='Escenario Crítico (10%)')
        ax_sim.set_title("Evolución Esperada del Capital con Retiros Activos")
        ax_sim.set_xlabel("Meses")
        ax_sim.set_ylabel("Saldo disponible (USD)")
        ax_sim.legend(loc="upper left")
        ax_sim.grid(True, alpha=0.2)
        st.pyplot(fig_sim)
        
        exitos = np.sum(simulaciones[-1] > 0)
        prob_exito = exitos / n_sims
        
        st.markdown("---")
        col_met1, col_met2 = st.columns(2)
        col_met1.metric("Probabilidad de éxito del plan", f"{prob_exito:.1%}")
        col_met2.markdown(f"💡 **Análisis:** Hay un **{prob_exito:.1%}** de probabilidades de que tus fondos resistan el ritmo de retiros de **USD {retiro_mensual:,.2f}/mes** a lo largo de {anios_proyeccion} años.")
