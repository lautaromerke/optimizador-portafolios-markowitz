import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
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
activos = [x.strip().upper() for x in lista_tickers.split(",") if x.strip()]

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
        "🎯 Cartera por Retorno Objetivo (Multialternativa)", 
        "💰 Simulador de Retiro Automatizado"
        "🧠 Neuro-Analisis de Portafolio"
    ]
)

# --- FUNCIONES AUXILIARES ---
def calcular_metricas(weights, ret_anuales, cov_matrix):
    ret = np.sum(ret_anuales * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def safe_str(texto):
    remplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ñ": "n", "Ñ": "N", "%": " por ciento", "•": "-", 
        "🟢": "COMPRAR", "🔴": "VENDER", "🟡": "MANTENER",
        "📊": "", "🎯": "", "💰": "", "📈": "", "📉": ""
    }
    origen = str(texto)
    for k, v in remplazos.items():
        origen = origen.replace(k, v)
    return origen.encode('latin-1', 'ignore').decode('latin-1')

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
    pdf.cell(0, 10, safe_str("Composicion del Portafolio:"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    for _, row in datos_cartera.iterrows():
        linea = f"  - {row['Activo']}: {row['Peso %']:.2f} por ciento"
        pdf.cell(0, 8, safe_str(linea), ln=True)
    return bytes(pdf.output())

# --- DESCARGA Y PROCESAMIENTO DE DATOS ---
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
# MÓDULO: BLACK-LITTERMAN
# =========================================================================
if modulo == "🔮 Algoritmo Black-Litterman (Opinión + Presets)":
    st.header("🔮 Optimización de Black-Litterman con Presets Macroeconómicos")
    
    preset = st.selectbox(
        "🎛️ Cargar Preset de Visión de Mercado:",
        ["Personalizado (Manual)", "Bull Market Tecnológico", "Rally de Acciones Argentinas", "Refugio Conservador"]
    )
    
    views = []; confianzas = []
    columnas_activos = st.columns(min(num_activos, 4))
    for idx, col_name in enumerate(datos_usd.columns):
        default_v = 0.0; default_c = 50
        if preset == "Bull Market Tecnológico":
            if col_name in ["AAPL", "MSFT", "NVDA", "GOOG"]: default_v = 8.0; default_c = 80
            elif col_name.endswith(".BA"): default_v = -2.0
        elif preset == "Rally de Acciones Argentinas":
            if col_name.endswith(".BA") or col_name in ["GGAL", "YPF", "BMA", "PAMP"]: default_v = 15.0; default_c = 70
            else: default_v = -1.0
        elif preset == "Refugio Conservador":
            if col_name in ["KO", "PEP", "PG", "JNJ", "WMT"]: default_v = 4.0; default_c = 90
            else: default_v = -5.0; default_c = 30

        with columnas_activos[idx % 4]:
            st.subheader(col_name)
            v_val = st.slider(f"Rendimiento Extra {col_name}:", -20.0, 20.0, default_v, step=1.0, key=f"v_{col_name}") / 100
            c_val = st.slider(f"Confianza {col_name} (%):", 10, 100, default_c, step=10, key=f"c_{col_name}") / 100
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
                     
        res_bl = minimize(lambda w: -calcular_metricas(w, Retorno_BL, cov_matrix)[2], [1./num_activos]*num_activos, bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
        
        if res_bl.success:
            r_final, v_final, s_final = calcular_metricas(res_bl.x, Retorno_BL, cov_matrix)
            df_bl = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': res_bl.x * 100})
            st.session_state['cartera_optima'] = df_bl
            
            c_bl1, c_bl2 = st.columns(2)
            with c_bl1:
                st.plotly_chart(px.pie(df_bl, values='Peso %', names='Activo', hole=0.4), use_container_width=True)
            with c_bl2:
                st.success("### ✨ Nuevas Métricas Estimadas")
                st.metric("Retorno Esperado Ajustado (USD)", f"{r_final:.2%}")
                st.metric("Volatilidad de Mercado", f"{v_final:.2%}")
                st.dataframe(df_bl.style.format({'Peso %': '{:.2f}%'}), use_container_width=True)


# =========================================================================
# MÓDULO: DIAGNÓSTICO Y REBALANCEO (FIX VALUERROR)
# =========================================================================
elif modulo == "⚡ Diagnóstico y Rebalanceo de Cartera Actual":
    st.header("⚡ Diagnóstico y Plan de Rebalanceo de Posiciones")
    capital_total = st.number_input("Tu Capital Total Invertido (USD):", min_value=100, value=5000, step=500)
    
    pesos_actuales = []
    col_inputs = st.columns(min(num_activos, 4))
    for i, col_name in enumerate(datos_usd.columns):
        with col_inputs[i % 4]:
            p_act = st.number_input(f"% Actual en {col_name}:", min_value=0.0, max_value=100.0, value=100.0/num_activos, step=5.0, key=f"reb_{col_name}")
            pesos_actuales.append(p_act)
            
    if abs(sum(pesos_actuales) - 100.0) > 0.5:
        st.warning("⚠️ La suma de tus porcentajes debe ser exactamente 100%.")
    else:
        if st.button("⚡ Analizar y Corregir Cartera"):
            # Forzamos optimización fresca sobre los activos de la barra lateral para evitar desfasajes de longitud
            res_fb = minimize(lambda w: -calcular_metricas(w, ret_anuales, cov_matrix)[2], [1./num_activos]*num_activos, bounds=((0,1),)*num_activos, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
            pesos_objetivo_ordenados = res_fb.x * 100
                
            activos_lista = list(datos_usd.columns)
            df_reb = pd.DataFrame({
                'Activo': activos_lista, 
                'Peso Actual %': pesos_actuales, 
                'Peso Objetivo %': pesos_objetivo_ordenados
            })
            
            df_reb['Desvío %'] = df_reb['Peso Objetivo %'] - df_reb['Peso Actual %']
            df_reb['Monto Actual (USD)'] = (df_reb['Peso Actual %'] / 100) * capital_total
            df_reb['Monto Objetivo (USD)'] = (df_reb['Peso Objetivo %'] / 100) * capital_total
            df_reb['Ajuste Necesario (USD)'] = df_reb['Monto Objetivo (USD)'] - df_reb['Monto Actual (USD)']
            
            df_melted = df_reb.melt(id_vars=['Activo'], value_vars=['Peso Actual %', 'Peso Objetivo %'], var_name='Tipo', value_name='Porcentaje')
            st.plotly_chart(px.bar(df_melted, x='Activo', y='Porcentaje', color='Tipo', barmode='group', color_discrete_sequence=['#ef4444', '#10b981']), use_container_width=True)
            
            df_reb['Acción Recomendada'] = df_reb['Ajuste Necesario (USD)'].apply(lambda x: "🟢 COMPRAR" if x > 10 else ("🔴 VENDER" if x < -10 else "🟡 Mantener"))
            st.dataframe(df_reb.style.format({'Peso Actual %': '{:.1f}%', 'Peso Objetivo %': '{:.1f}%', 'Desvío %': '{:+.1f}%', 'Monto Actual (USD)': '${:.2f}', 'Monto Objetivo (USD)': '${:.2f}', 'Ajuste Necesario (USD)': '${:+.2f}'}), use_container_width=True)


# =========================================================================
# MÓDULO: FRONTERA EFICIENTE (MARKOWITZ)
# =========================================================================
elif modulo == "📈 Frontera Eficiente (Markowitz)":
    st.header("📈 Optimización Clásica (Frontera de Markowitz)")
    n_sims = st.slider("Cantidad de simulaciones:", 5000, 20000, 10000, step=5000)
    
    if st.button("🚀 Calcular Portafolios"):
        results = np.zeros((3, n_sims))
        w_list = []
        for i in range(n_sims):
            w = np.random.random(num_activos); w /= np.sum(w); w_list.append(w)
            r, v, s = calcular_metricas(w, ret_anuales, cov_matrix)
            results[0, i] = r; results[1, i] = v; results[2, i] = s
        
        idx_sharpe = results[2].argmax()
        df_w = pd.DataFrame({'Activo': datos_usd.columns, 'Peso %': w_list[idx_sharpe] * 100})
        
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.pie(df_w, values='Peso %', names='Activo', hole=0.4), use_container_width=True)
        with c2:
            st.success("### 🎯 Métricas Máximo Sharpe")
            st.metric("Retorno Esperado (USD)", f"{results[0, idx_sharpe]:.2%}")
            st.metric("Volatilidad (Riesgo)", f"{results[1, idx_sharpe]:.2%}")
            pdf_data = generar_reporte_pdf(df_w, "Maximo Sharpe Markowitz", results[0, idx_sharpe], results[1, idx_sharpe])
            st.download_button("📩 Descargar Reporte PDF", data=pdf_data, file_name="reporte_markowitz.pdf", mime="application/pdf")


# =========================================================================
# MÓDULO: CARTERA OBJETIVO MULTIALTERNATIVA
# =========================================================================
elif modulo == "🎯 Cartera por Retorno Objetivo (Multialternativa)":
    st.header("🎯 Planificación Inversa Multi-Perfil: Tu Meta en Efectivo")
    st.markdown("""
    Ingresá cuántos **dólares en efectivo querés meterte en el bolsillo por mes**. El sistema va a calcular la arquitectura 
    financiera requerida evaluando tus activos actuales frente a **3 alternativas recomendadas** según el riesgo.
    """)
    
    renta_deseada_mensual = st.number_input("Monto mensual objetivo (USD):", min_value=10, value=500, step=50)
    renta_anual_necesaria = renta_deseada_mensual * 12
    
    perfil_seleccionado = st.radio(
        "🔍 Elegí qué alternativa de portafolio querés analizar para tu meta:",
        ["Tu Set de Activos Actual", "Alternativa Conservadora (Renta Estable)", "Alternativa Moderada (Balance global)", "Alternativa Arriesgada (Máximo Crecimiento)"],
        horizontal=True
    )
    
    tickers_alternativos = []
    if perfil_seleccionado == "Tu Set de Activos Actual":
        tickers_alternativos = activos
    elif perfil_seleccionado == "Alternativa Conservadora (Renta Estable)":
        tickers_alternativos = ["KO", "PG", "XOM", "JNJ"]
    elif perfil_seleccionado == "Alternativa Moderada (Balance global)":
        tickers_alternativos = ["SPY", "AAPL", "MSFT", "GGAL.BA"]
    elif perfil_seleccionado == "Alternativa Arriesgada (Máximo Crecimiento)":
        tickers_alternativos = ["NVDA", "TSLA", "QQQ", "AMD"]

    with st.spinner(f"Analizando rendimientos históricos para la {perfil_seleccionado}..."):
        raw_alt = yf.download(tickers_alternativos, start=fecha_inicio, end=fecha_fin, progress=False)
        if raw_alt.empty or 'Close' not in raw_alt:
            st.error("Error al conectar con los servidores de mercado para este perfil.")
            st.stop()
            
        df_close_alt = raw_alt['Close']
        if isinstance(df_close_alt, pd.DataFrame):
            if isinstance(df_close_alt.columns, pd.MultiIndex):
                df_close_alt.columns = df_close_alt.columns.get_level_values(-1)
        else:
            df_close_alt = df_close_alt.to_frame(name=tickers_alternativos[0])
            
        df_alt_usd = df_close_alt.dropna().copy()
        for col in df_alt_usd.columns:
            if col.endswith(".BA"):
                df_alt_usd[col] = df_alt_usd[col] / mep_actual
                
        rend_alt = df_alt_usd.pct_change().dropna()
        ret_an_alt = rend_alt.mean() * 252
        cov_mat_alt = rend_alt.cov() * 252
        n_act_alt = len(df_alt_usd.columns)
        
    res_alt = minimize(lambda w: -calcular_metricas(w, ret_an_alt, cov_mat_alt)[2], [1./n_act_alt]*n_act_alt, bounds=((0,1),)*n_act_alt, constraints={'type':'eq','fun':lambda w: np.sum(w)-1.0})
    
    if res_alt.success:
        tasa_ret, vol_ret, _ = calcular_metricas(res_alt.x, ret_an_alt, cov_mat_alt)
        if tasa_ret <= 0.01:
            tasa_ret = 0.04
            
        capital_total_requerido = renta_anual_necesaria / tasa_ret
        
        df_final_alternativo = pd.DataFrame({
            'Activo': df_alt_usd.columns,
            'Peso %': res_alt.x * 100,
            'Monto a Invertir (USD)': res_alt.x * capital_total_requerido
        })
        
        st.markdown("---")
        st.success(f"### Plan de Ingenieria Financiera para {perfil_seleccionado}")
        
        c_an1, c_an2 = st.columns([1, 1])
        with c_an1:
            st.metric(
                label="CAPITAL TOTAL ESTIMADO A INVERTIR", 
                value=f"${capital_total_requerido:,.2f}",
                help="Monto total recomendado a colocar en este perfil para generar la renta mensual deseada."
            )
            st.metric(label="Retorno Compuesto Anual Esperado", value=f"{tasa_ret:.2%}")
            st.metric(label="Nivel de Volatilidad / Riesgo", value=f"{vol_ret:.2%}")
        
        with c_an2:
            st.write("### Fragmentacion Estrategica del Capital:")
            st.plotly_chart(px.pie(df_final_alternativo, values='Monto a Invertir (USD)', names='Activo', hole=0.4), use_container_width=True)
            
        st.write("### Desglose de Compras Requeridas")
        st.dataframe(df_final_alternativo.style.format({'Peso %': '{:.2f}%', 'Monto a Invertir (USD)': '${:,.2f}'}), use_container_width=True)
        
        pdf_data = generar_reporte_pdf(df_final_alternativo, f"Plan Alternativo {perfil_seleccionado}", tasa_ret, vol_ret)
        st.download_button("📩 Descargar Plan de Compra (PDF)", data=pdf_data, file_name="plan_cartera_alternativa.pdf", mime="application/pdf")


# =========================================================================
# MÓDULO: SIMULADOR DE RETIRO AUTOMÁTICO
# =========================================================================
elif modulo == "💰 Simulador de Retiro Automatizado":
    st.header("💰 Simulador de Supervivencia de Capital")
    
    c_sim1, c_sim2 = st.columns(2)
    capital_inicial = c_sim1.number_input("Capital Inicial Disponible (USD):", min_value=1000, value=50000, step=5000)
    retiro_mensual = c_sim2.number_input("Retiro Mensual Deseado (USD):", min_value=10, value=500, step=50)
    
    if st.button("⏳ Simular Perfiles de Riesgo"):
        promedio_ret = ret_anuales.mean()
        promedio_vol = rendimientos.std().mean() * np.sqrt(252)
        
        perfiles = {
            'Conservador (Riesgo Bajo)': {'mu': promedio_ret * 0.6, 'sigma': promedio_vol * 0.5},
            'Moderado (Riesgo Medio)': {'mu': promedio_ret, 'sigma': promedio_vol},
            'Arriesgado (Riesgo Alto)': {'mu': promedio_ret * 1.3, 'sigma': promedio_vol * 1.5}
        }
        
        meses = 180
        df_grafico = pd.DataFrame(index=range(meses))
        
        for nombre, params in perfiles.items():
            trayectorias = np.zeros((meses, 30))
            trayectorias[0] = capital_inicial
            for s in range(30):
                for t in range(1, meses):
                    rendimiento_mes = np.random.normal(params['mu']/12, params['sigma']/np.sqrt(12))
                    trayectorias[t, s] = max(0, trayectorias[t-1, s] * (1 + rendimiento_mes) - retiro_mensual)
            df_grafico[nombre] = np.mean(trayectorias, axis=1)
            
        st.line_chart(df_grafico)
        
        st.write("### 🏁 Diagnóstico de Sostenibilidad")
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            m = (df_grafico['Conservador (Riesgo Bajo)'] > 0).sum()
            st.metric("Duración Conservador", f"{m} meses", f"{m/12:.1f} años")
        with c_p2:
            m = (df_grafico['Moderado (Riesgo Medio)'] > 0).sum()
            st.metric("Duración Moderado", f"{m} meses", f"{m/12:.1f} años")
        with c_p3:
            m = (df_grafico['Arriesgado (Riesgo Alto)'] > 0).sum()
            st.metric("Duración Arriesgado", f"{m} meses", f"{m/12:.1f} años")

# =========================================================================
# MÓDULO: NEURO-ANÁLISIS DE PORTAFOLIO
# =========================================================================
elif modulo == "🧠 Neuro-Analisis de Portafolio":
    st.header("🧠 Oráculo de Comportamiento del Inversor")
    
    st.write("Calculando tu nivel de 'Ansiedad de Mercado' basado en la volatilidad de tus activos...")
    
    # Calculamos volatilidad promedio de la cartera
    vol_diaria = rendimientos.std().mean()
    
    if vol_diaria > 0.02:
        st.error("⚠️ ESTADO ACTUAL: ALTA ANSIEDAD. Tu cartera tiene una volatilidad que puede causar reacciones impulsivas.")
        st.info("💡 Consejo: Considerá reducir la exposición a activos de alta volatilidad (Beta alto) o agregar un activo de refugio.")
    else:
        st.success("✅ ESTADO ACTUAL: MODO ZEN. Tu cartera es estable. Mantené la estrategia.")
        
    st.markdown("---")
    umbral_panico = st.slider("¿Cuánto % de caída te hace dudar de tu inversión?", 5, 50, 15)
    
    # Simulamos probabilidad de caída técnica
    prob_caida = (rendimientos.mean() - (2 * rendimientos.std())).min() * 100
    st.write(f"Según tu histórico, existe una probabilidad técnica de que tu cartera caiga un {abs(prob_caida):.1f}% en un mes crítico.")# =========================================================================
# MÓDULO: NEURO-ANÁLISIS DE PORTAFOLIO
# =========================================================================
elif modulo == "🧠 Neuro-Analisis de Portafolio":
    st.header("🧠 Oráculo de Comportamiento del Inversor")
    
    st.write("Calculando tu nivel de 'Ansiedad de Mercado' basado en la volatilidad de tus activos...")
    
    # Calculamos volatilidad promedio de la cartera
    vol_diaria = rendimientos.std().mean()
    
    if vol_diaria > 0.02:
        st.error("⚠️ ESTADO ACTUAL: ALTA ANSIEDAD. Tu cartera tiene una volatilidad que puede causar reacciones impulsivas.")
        st.info("💡 Consejo: Considerá reducir la exposición a activos de alta volatilidad (Beta alto) o agregar un activo de refugio.")
    else:
        st.success("✅ ESTADO ACTUAL: MODO ZEN. Tu cartera es estable. Mantené la estrategia.")
        
    st.markdown("---")
    umbral_panico = st.slider("¿Cuánto % de caída te hace dudar de tu inversión?", 5, 50, 15)
    
    # Simulamos probabilidad de caída técnica
    prob_caida = (rendimientos.mean() - (2 * rendimientos.std())).min() * 100
    st.write(f"Según tu histórico, existe una probabilidad técnica de que tu cartera caiga un {abs(prob_caida):.1f}% en un mes crítico.")
