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

# --- BOTÓN PRINCIPAL Y LÓGICA DE EJECUCIÓN ---
if st.sidebar.button("🚀 Ejecutar Planificación Integral"):
    with st.spinner("Procesando modelos cuantitativos y proyecciones..."):
        try:
            if fecha_inicio >= fecha_fin:
                st.warning("⚠️ **Error en el rango:** La fecha de inicio no puede ser posterior o igual a la de fin.")
                st.stop()

            # Descarga de datos
            datos_raw = yf.download(activos, start=fecha_inicio, end=fecha_fin)
            if datos_raw.empty or 'Close' not in datos_raw:
                st.warning("⚠️ No se encontraron precios de cierre para esos tickers.")
                st.stop()
            
            df_close = datos_raw['Close']
            if isinstance(df_close, pd.Series):
                df_close = df_close.to_frame(name=activos[0])
                
            datos_usd = df_close.dropna().copy()
            
            # Dolarización de Cedears
            for col in datos_usd.columns:
                if col.endswith(".BA"):
                    datos_usd[col] = datos_usd[col] / mep_actual
            
            rendimientos = datos_usd.pct_change().dropna()
            if rendimientos.empty:
                st.warning("⚠️ Datos insuficientes tras la limpieza de filas vacías.")
                st.stop()

            ret_anuales = rendimientos.mean() * 252
            cov_matrix = rendimientos.cov() * 252

            # Simulación Montecarlo Markowitz
            num_activos = len(datos_usd.columns)
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
            best_w_sharpe = w_list[idx_sharpe]

            # Optimización por objetivo (Scipy)
            restricciones = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sum(ret_anuales * w) - (retorno_objetivo_pct / 100.0)}
            )
            limites = tuple((0.0, 1.0) for _ in range(num_activos))
            iniciales = [1.0 / num_activos] * num_activos
            
            opt_res = minimize(
                minimizar_volatilidad, 
                iniciales, 
                args=(ret_anuales, cov_matrix), 
                method='SLSQP', 
                bounds=limites, 
                constraints=restricciones
            )
            
            # Renderizado de resultados
            st.subheader("🎯 Análisis de Carteras")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.success("### Cartera Máx. Sharpe")
                st.metric("Retorno Esperado", f"{results[0, idx_sharpe]:.2%}")
                st.metric("Volatilidad", f"{results[1, idx_sharpe]:.2%}")
                
            with c2:
                if opt_res.success:
                    st.info("### Cartera Objetivo")
                    res_obj = calcular_metricas_cartera(opt_res.x, ret_anuales, cov_matrix)
                    st.metric("Retorno Logrado", f"{res_obj[0]:.2%}")
                    st.metric("Riesgo Mínimo", f"{res_obj[1]:.2%}")
                    pesos_finales = opt_res.x
                else:
                    st.warning("### Cartera Objetivo")
                    st.write("⚠️ Meta inalcanzable con estos activos. Se usará la de Máx. Sharpe.")
                    pesos_finales = best_w_sharpe

            with c3:
                st.write("### Asignación de Capital (%)")
                df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': pesos_finales * 100})
                st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)

            # Simulador de retiro de fondos
            st.markdown("---")
            st.subheader(f"⏳ Simulador de Supervivencia de Capital ({anios_proyeccion} años)")
            
            res_final = calcular_metricas_cartera(pesos_finales, ret_anuales, cov_matrix)
            mu = res_final[0]
            sigma = res_final[1]
            
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
            ax_sim.set_title("Evolución Esperada del Capital Ajustada por Retiros")
            ax_sim.set_xlabel("Meses")
            ax_sim.set_ylabel("Saldo disponible (USD)")
            ax_sim.legend(loc="upper left")
            ax_sim.grid(True, alpha=0.2)
            st.pyplot(fig_sim)
            
            exitos = np.sum(simulaciones[-1] > 0)
            prob_exito = exitos / n_sims
            
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.metric("Probabilidad de éxito del plan", f"{prob_exito:.1%}")
            with col_met2:
                st.markdown(f"💡 **Análisis Cuantitativo:** Hay un **{prob_exito:.1%}** de chances de que tus fondos resistan el ritmo de retiros de **USD {retiro_mensual:,.2f}/mes** a lo largo de {anios_proyeccion} años.")

        except Exception as e:
            st.error(f"Error en el procesamiento financiero: {e}")import streamlit as st
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

# --- BOTÓN PRINCIPAL Y LÓGICA DE EJECUCIÓN ---
if st.sidebar.button("🚀 Ejecutar Planificación Integral"):
    with st.spinner("Procesando modelos cuantitativos y proyecciones..."):
        try:
            if fecha_inicio >= fecha_fin:
                st.warning("⚠️ **Error en el rango:** La fecha de inicio no puede ser posterior o igual a la de fin.")
                st.stop()

            # Descarga de datos
            datos_raw = yf.download(activos, start=fecha_inicio, end=fecha_fin)
            if datos_raw.empty or 'Close' not in datos_raw:
                st.warning("⚠️ No se encontraron precios de cierre para esos tickers.")
                st.stop()
            
            df_close = datos_raw['Close']
            if isinstance(df_close, pd.Series):
                df_close = df_close.to_frame(name=activos[0])
                
            datos_usd = df_close.dropna().copy()
            
            # Dolarización de Cedears
            for col in datos_usd.columns:
                if col.endswith(".BA"):
                    datos_usd[col] = datos_usd[col] / mep_actual
            
            rendimientos = datos_usd.pct_change().dropna()
            if rendimientos.empty:
                st.warning("⚠️ Datos insuficientes tras la limpieza de filas vacías.")
                st.stop()

            ret_anuales = rendimientos.mean() * 252
            cov_matrix = rendimientos.cov() * 252

            # Simulación Montecarlo Markowitz
            num_activos = len(datos_usd.columns)
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
            best_w_sharpe = w_list[idx_sharpe]

            # Optimización por objetivo (Scipy)
            restricciones = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
                {'type': 'eq', 'fun': lambda w: np.sum(ret_anuales * w) - (retorno_objetivo_pct / 100.0)}
            )
            limites = tuple((0.0, 1.0) for _ in range(num_activos))
            iniciales = [1.0 / num_activos] * num_activos
            
            opt_res = minimize(
                minimizar_volatilidad, 
                iniciales, 
                args=(ret_anuales, cov_matrix), 
                method='SLSQP', 
                bounds=limites, 
                constraints=restricciones
            )
            
            # Renderizado de resultados
            st.subheader("🎯 Análisis de Carteras")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.success("### Cartera Máx. Sharpe")
                st.metric("Retorno Esperado", f"{results[0, idx_sharpe]:.2%}")
                st.metric("Volatilidad", f"{results[1, idx_sharpe]:.2%}")
                
            with c2:
                if opt_res.success:
                    st.info("### Cartera Objetivo")
                    res_obj = calcular_metricas_cartera(opt_res.x, ret_anuales, cov_matrix)
                    st.metric("Retorno Logrado", f"{res_obj[0]:.2%}")
                    st.metric("Riesgo Mínimo", f"{res_obj[1]:.2%}")
                    pesos_finales = opt_res.x
                else:
                    st.warning("### Cartera Objetivo")
                    st.write("⚠️ Meta inalcanzable con estos activos. Se usará la de Máx. Sharpe.")
                    pesos_finales = best_w_sharpe

            with c3:
                st.write("### Asignación de Capital (%)")
                df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': pesos_finales * 100})
                st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)

            # Simulador de retiro de fondos
            st.markdown("---")
            st.subheader(f"⏳ Simulador de Supervivencia de Capital ({anios_proyeccion} años)")
            
            res_final = calcular_metricas_cartera(pesos_finales, ret_anuales, cov_matrix)
            mu = res_final[0]
            sigma = res_final[1]
            
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
            ax_sim.set_title("Evolución Esperada del Capital Ajustada por Retiros")
            ax_sim.set_xlabel("Meses")
            ax_sim.set_ylabel("Saldo disponible (USD)")
            ax_sim.legend(loc="upper left")
            ax_sim.grid(True, alpha=0.2)
            st.pyplot(fig_sim)
            
            exitos = np.sum(simulaciones[-1] > 0)
            prob_exito = exitos / n_sims
            
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.metric("Probabilidad de éxito del plan", f"{prob_exito:.1%}")
            with col_met2:
                st.markdown(f"💡 **Análisis Cuantitativo:** Hay un **{prob_exito:.1%}** de chances de que tus fondos resistan el ritmo de retiros de **USD {retiro_mensual:,.2f}/mes** a lo largo de {anios_proyeccion} años.")

        except Exception as e:
            st.error(f"Error en el procesamiento financiero: {e}")
