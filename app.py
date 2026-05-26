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

st.title("📊 Suite Financiera Pro: Optimización y Diagnóstico")
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
    [
        "🔮 Algoritmo Black-Litterman (Opinión + Presets)",
        "⚡ Diagnóstico y Rebalanceo de Cartera Actual",
        "📈 Frontera Eficiente (Markowitz)", 
        "🎯 Cartera por Retorno Objetivo", 
        "💰 Simulador de Retiro"
    ]
)

# --- FUNCIONES AUXILIARES ---
def calcular_metricas(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def safe_str(texto):
    # Convierte caracteres Unicode problemáticos para evitar colapsos en la generación del PDF
    return str(texto).encode('latin-1', 'ignore').decode('latin-1')

def generar_reporte_pdf(datos_cartera, nombre_plan, retorno, riesgo):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, safe_str(f"Reporte Financiero: {nombre_plan}"), ln=True, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, safe_str(f"Fecha de emision: {date.today().strftime('%d/%m/%Y')}"), ln=True)
    pdf.cell(0, 8, safe_str(f"Retorno Anual Esperado (USD): {retorno:.2%}"), ln=True)
    pdf.cell(0, 8, safe_str(f"Volatilidad Anual (Riesgo): {riesgo:.2%}"), ln=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, safe_str("Composicion Eficiente del Portafolio:"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    for _, row in datos_cartera.iterrows():
        # Evitamos el símbolo % nativo en la celda string de FPDF para prevenir FPDFUnicodeEncodingException
        linea = f"  - {row['Activo']}: {row['Peso %']:.2f} por ciento"
        pdf.cell(0, 8, safe_str(linea), ln=True)
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

if isinstance(df_close, pd.DataFrame):
    if isinstance(df_close.columns, pd.MultiIndex):
        df_close.columns = df_close.columns.get_level_values(-1)
else:
    df_close = df_close.to_frame(name=activos[0])

datos_usd = df_close.dropna().copy()

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
# MÓDULO: BLACK-LITTERMAN + PRESETS DINÁMICOS
# =========================================================================
if modulo == "🔮 Algoritmo Black-Litterman (Opinión + Presets)":
    st.header("🔮 Optimización de Black-Litterman con Presets Macroeconómicos")
    st.markdown("""
    Combina el análisis histórico base con proyecciones macro. Podés usar un **Preset** rápido para 
    no rellenar todo a mano, o ajustar los sliders uno por uno según tus criterios.
    """)
    
    preset = st.selectbox(
        "🎛️ Cargar Preset de Visión de Mercado:",
        ["Personalizado (Manual)", "Bull Market Tecnológico", "Rally de Acciones Argentinas", "Refugio Conservador"]
    )
    
    st.write("### 🧠 Configura tus Opiniones de Rendimiento Extra (Anual USD)")
    views = []
    confianzas = []
    
    columnas_activos = st.columns(min(num_activos, 4))
    for idx, col_name in enumerate(datos_usd.columns):
        default_v = 0.0
        default_c = 50
        
        if preset == "Bull Market Tecnológico":
            if col_name in ["AAPL", "MSFT", "NVDA", "GOOG"]:
                default_v = 8.0
                default_c = 80
            elif col_name.endswith(".BA"):
                default_v = -2.0
        elif preset == "Rally de Acciones Argentinas":
            if col_name.endswith(".BA") or col_name in ["GGAL", "YPF", "BMA", "PAMP"]:
                default_v = 15.0
                default_c = 70
            else:
                default_v = -1.0
        elif preset == "Refugio Conservador":
            if col_name in ["KO", "PEP", "PG", "JNJ", "WMT"]:
                default_v = 4.0
                default_c = 90
            else:
                default_v = -5.0
                default_c = 30

        with columnas_activos[idx % 4]:
            st.subheader(col_name)
            v_val = st.slider(f"Rendimiento Extra:", -20.0, 20.0, default_v, step=1.0, key=f"v_{col_name}") / 100
            c_val = st.slider(f"Confianza (%):", 10, 100, default_c, step=10, key=f"c_{col_name}") / 100
            views.append(v_val)
            confianzas.append(c_val)

    if st.button("🔮 Ejecutar Fusión Bayesiana"):
        Pi = ret_anuales.values
        Sigma = cov_matrix.values
        tau = 0.05
        
        P = np.eye(num_activos)
        Q = Pi + np.array(views)
        
        Omega = np.zeros((num_activos, num_activos))
        for i in range(num_activos):
            incertidumbre = (1.01 - confianzas[i]) * (Sigma[i, i] * 2)
            Omega[i, i] = max(incertidumbre, 1e-6)
            
        inv_tau_Sigma = np.linalg.inv(tau * Sigma)
        inv_Omega = np.linalg.inv(Omega)
        
        Retorno_BL = np.linalg.inv(inv_tau_Sigma + np.dot(P.T, np.dot(inv_Omega, P))) \
                     @ (np.dot(inv_tau_Sigma, Pi) + np.dot(P.T, np.dot(inv_Omega, Q)))
                     
        def max_sharpe_bl(w):
            return -calcular_metricas(w, Retorno_BL, cov_matrix)[2]
            
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        limites = tuple((0.0, 1.0) for _ in range(num_activos))
        iniciales = [1.0 / num_activos] * num_activos
        
        res_bl = minimize(max_sharpe_bl, iniciales, bounds=limites, constraints=cons)
        
        if res_bl.success:
            r_final, v_final, s_final = calcular_metricas(res_bl.x, Retorno_BL, cov_matrix)
            df_bl = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': res_bl.x * 100})
            st.session_state['cartera_optima'] = df_bl
            
            st.markdown("---")
            st.subheader("🏁 Resultado de la Cartera Fusionada (BL)")
            
            c_bl1, c_bl2 = st.columns(2)
            with c_bl1:
                fig_pie = px.pie(df_bl, values='Peso %', names='Activo', hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_bl2:
                st.success("### ✨ Nuevas Métricas Estimadas")
                st.metric("Retorno Esperado Ajustado (USD)", f"{r_final:.2%}")
                st.metric("Volatilidad de Mercado", f"{v_final:.2%}")
                st.dataframe(df_bl.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)
                
                pdf_data = generar_reporte_pdf(df_bl, f"Black-Litterman ({preset})", r_final, v_final)
                st.download_button(label="📩 Descargar Reporte BL (PDF)", data=pdf_data, file_name="reporte_black_litterman.pdf", mime="application/pdf")


# =========================================================================
# MÓDULO: DIAGNÓSTICO Y REBALANCEO AUTOMÁTICO
# =========================================================================
elif modulo == "⚡ Diagnóstico y Rebalanceo de Cartera Actual":
    st.header("⚡ Diagnóstico y Plan de Rebalanceo de Posiciones")
    st.markdown("""
    Ingresá el capital total y el porcentaje real que tenés invertido hoy en cada activo. El sistema lo va a auditar contra 
    la frontera eficiente para decirte exactamente **qué comprar y qué vender** para eliminar riesgos innecesarios.
    """)
    
    c_reb1, c_reb2 = st.columns(2)
    capital_total = c_reb1.number_input("Tu Capital Total Invertido (USD):", min_value=100, value=5000, step=500)
    
    st.write("### 🏦 Composición de tu Cartera Actual")
    pesos_actuales = []
    col_inputs = st.columns(min(num_activos, 4))
    
    for i, col_name in enumerate(datos_usd.columns):
        with col_inputs[i % 4]:
            p_act = st.number_input(f"% Actual en {col_name}:", min_value=0.0, max_value=100.0, value=100.0/num_activos, step=5.0, key=f"reb_{col_name}")
            pesos_actuales.append(p_act)
            
    sum_pesos = sum(pesos_actuales)
    if abs(sum_pesos - 100.0) > 0.1:
        st.warning(f"⚠️ La suma de tus porcentajes actuales es {sum_pesos:.1f}%. Debe sumar exactamente 100% para realizar el análisis.")
    else:
        if st.button("⚡ Analizar y Corregir Cartera"):
            if 'cartera_optima' in st.session_state:
                df_target = st.session_state['cartera_optima']
                origen_target = "Sugerida por Black-Litterman"
            else:
                def max_sharpe_fb(w): return -calcular_metricas(w, ret_anuales, cov_matrix)[2]
                res_fb = minimize(max_sharpe_fb, [1./num_activos]*num_activos, bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
                df_target = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': res_fb.x * 100})
                origen_target = "Óptima Teórica (Markowitz Histórico)"
                
            df_reb = pd.DataFrame({
                'Activo': datos_usd.columns,
                'Peso Actual %': pesos_actuales,
                'Peso Objetivo %': df_target['Peso %'].values
            })
            df_reb['Desvío %'] = df_reb['Peso Objetivo %'] - df_reb['Peso Actual %']
            df_reb['Monto Actual (USD)'] = (df_reb['Peso Actual %'] / 100) * capital_total
            df_reb['Monto Objetivo (USD)'] = (df_reb['Peso Objetivo %'] / 100) * capital_total
            df_reb['Ajuste Necesario (USD)'] = df_reb['Monto Objetivo (USD)'] - df_reb['Monto Actual (USD)']
            
            st.write(f"### 📊 Comparativa: Actual vs Cartera {origen_target}")
            df_melted = df_reb.melt(id_vars=['Activo'], value_vars=['Peso Actual %', 'Peso Objetivo %'], var_name='Tipo', value_name='Porcentaje')
            fig_bar = px.bar(df_melted, x='Activo', y='Porcentaje', color='Tipo', barmode='group', color_discrete_sequence=['#ef4444', '#10b981'])
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.write("### 📑 Plan de Ejecución Inmediata")
            
            def generar_instruccion(val):
                if val > 10.0: return "🟢 COMPRAR FUERTE"
                elif val > 0: return "🟢 Comprar leve"
                elif val < -10.0: return "🔴 VENDER FUERTE"
                else: return "🔴 Vender leve"
                
            df_reb['Acción Recomendada'] = df_reb['Ajuste Necesario (USD)'].apply(generar_instruccion)
            
            st.dataframe(
                df_reb.style.format({
                    'Peso Actual %': '{:.1f}%', 'Peso Objetivo %': '{:.1f}%', 'Desvío %': '{:+.1f}%',
                    'Monto Actual (USD)': '${:.2f}', 'Monto Objetivo (USD)': '${:.2f}', 'Ajuste Necesario (USD)': '${:+.2f}'
                }), use_container_width=True
            )


# =========================================================================
# MÓDULOS TRADICIONALES
# =========================================================================
elif modulo == "📈 Frontera Eficiente (Markowitz)":
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
            results[0, i] = r; results[1, i] = v; results[2, i] = s
        
        idx_sharpe = results[2].argmax()
        df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': w_list[idx_sharpe] * 100})
        st.session_state['cartera_optima'] = df_w
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("### 🍰 Ponderación de la Cartera Óptima")
            fig_pie = px.pie(df_w, values='Peso %', names='Activo', hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.success("### 🎯 Métricas Máximo Sharpe")
            st.metric("Retorno Esperado (USD)", f"{results[0, idx_sharpe]:.2%}")
            st.metric("Volatilidad (Riesgo)", f"{results[1, idx_sharpe]:.2%}")
            st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)
            
            pdf_data = generar_reporte_pdf(df_w, "Maximo Sharpe (Markowitz)", results[0, idx_sharpe], results[1, idx_sharpe])
            st.download_button(label="📩 Descargar Reporte PDF", data=pdf_data, file_name="reporte_markowitz.pdf", mime="application/pdf")

elif modulo == "🎯 Cartera por Retorno Objetivo":
    st.header("🎯 Optimización por Objetivo de Retorno")
    meta = st.slider("Introduce tu Retorno Anual Objetivo (USD %):", 5.0, 30.0, 12.0, step=0.5) / 100
    
    if st.button("🎯 Resolver Combinación Óptima"):
        restricciones = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'eq', 'fun': lambda w: np.sum(ret_anuales * w) - meta})
        res = minimize(lambda w: calcular_metricas(w, ret_anuales, cov_matrix)[1], [1./num_activos]*num_activos, bounds=((0,1),)*num_activos, constraints=restricciones)
        
        if res.success:
            r, v, s = calcular_metricas(res.x, ret_anuales, cov_matrix)
            df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': res.x * 100})
            st.session_state['cartera_optima'] = df_w
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("### 🍰 Distribución de Activos")
                fig_pie = px.pie(df_w, values='Peso %', names='Activo', hole=0.4, color_discrete_sequence=px.colors.qualitative.Dark2)
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                st.info("### Métricas Alcanzadas")
                st.metric("Retorno Objetivo", f"{r:.2%}")
                st.metric("Riesgo Mínimo", f"{v:.2%}")
                st.dataframe(df_w.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)
                
                pdf_data = generar_reporte_pdf(df_w, "Retorno Objetivo Minimizado", r, v)
                st.download_button(label="📩 Descargar Reporte PDF", data=pdf_data, file_name="reporte_objetivo.pdf", mime="application/pdf")
        else:
            st.error("⚠️ Meta inalcanzable con el set de activos actual.")

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
                simulations[t, s] = max(0, simulations[t-1, s] * (1 + np.random.normal(mu/12, sigma/np.sqrt(12))) - retiro_mensual)
        st.line_chart(simulations)
        st.metric("Probabilidad de éxito del fondo", f"{np.sum(simulations[-1] > 0) / 100:.1%}")
