
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import scipy.stats as stats

# 1. Configuración de la página web
st.set_page_config(page_title="Plataforma Avanzada de Optimización y Riesgo", layout="wide")

st.title("📊 Plataforma Avanzada de Optimización y Stress Testing")
st.markdown("""
Esta plataforma automatiza la **Teoría Moderna de Portafolios de Markowitz** y la complementa con métricas de **Riesgo Extremo (VaR/CVaR)** y simulaciones de **Crisis Históricas** requeridas en la gestión de fondos profesional.
""")

st.sidebar.header("🔧 1. Configuración del Portafolio")

# Inputs interactivos principales
lista_tickers = st.sidebar.text_input("Introduce los Tickers (separados por coma):", "AAPL, MSFT, GOOGL, KO, TSLA")
fecha_inicio = st.sidebar.date_input("Fecha de inicio (Histórico normal):", pd.to_datetime("2021-01-01"))
fecha_fin = st.sidebar.date_input("Fecha de fin (Histórico normal):", pd.to_datetime("2026-05-25"))
num_portafolios = st.sidebar.slider("Número de portafolios a simular:", 5000, 30000, 15000, step=5000)

activos = [x.strip().upper() for x in lista_tickers.split(",")]

st.sidebar.header("🚨 2. Pruebas de Estrés (Stress Testing)")
crisis_seleccionada = st.sidebar.selectbox(
    "Selecciona una crisis histórica para simular:",
    ["Ninguna", "Pandemia COVID-19 (2020)", "Crisis Financiera Subprime (2008)", "Ajuste Macroeconómico Global (2022)"]
)

# Mapeo de fechas de crisis reales
fechas_crisis = {
    "Pandemia COVID-19 (2020)": ("2020-02-01", "2020-06-01"),
    "Crisis Financiera Subprime (2008)": ("2008-01-01", "2009-06-01"),
    "Ajuste Macroeconómico Global (2022)": ("2022-01-01", "2022-12-31")
}

if st.sidebar.button("🚀 Ejecutar Análisis Completo"):
    with st.spinner("Procesando modelos cuantitativos..."):
        try:
            # --- VALIDACIONES ---
            hoy = date.today()
            if fecha_inicio >= fecha_fin:
                st.warning("⚠️ **Error:** La fecha de inicio no puede ser posterior o igual a la de fin.")
                st.stop()
            if fecha_inicio > hoy:
                st.warning("⚠️ **Error:** La fecha elegida está en el futuro.")
                st.stop()

            # --- 3. DESCARGA DE DATOS ---
            datos_completos = yf.download(activos, start=fecha_inicio, end=fecha_fin)
            if datos_completos.empty or 'Close' not in datos_completos:
                st.warning("⚠️ No se encontraron datos para los tickers ingresados.")
                st.stop()
                
            datos = datos_completos['Close'].dropna()
            rendimientos = datos.pct_change().dropna()
            
            retornos_anuales = rendimientos.mean() * 252
            matriz_covarianza = rendimientos.cov() * 252
            
            # --- 4. SIMULACIÓN DE MONTECARLO ---
            resultados = np.zeros((3, num_portafolios))
            lista_pesos = []

            for i in range(num_portafolios):
                pesos = np.random.random(len(activos))
                pesos /= np.sum(pesos)
                lista_pesos.append(pesos)
                
                retorno_p = np.sum(retornos_anuales * pesos)
                volatilidad_p = np.sqrt(np.dot(pesos.T, np.dot(matriz_covarianza, pesos)))
                
                resultados[0,i] = retorno_p
                resultados[1,i] = volatilidad_p
                resultados[2,i] = retorno_p / volatilidad_p 

            idx_max_sharpe = resultados[2].argmax()
            pesos_optimos = lista_pesos[idx_max_sharpe]
            
            # --- 5. CÁLCULO DE MÉTRICAS DE RIESGO EXTREMO (VaR y CVaR) ---
            # Rendimientos diarios históricos de la cartera óptima aplicada al pasado reciente
            rendimientos_cartera = rendimientos.dot(pesos_optimos)
            
            # VaR Paramétrico (95% confianza diario)
            media_c = rendimientos_cartera.mean()
            sigma_c = rendimientos_cartera.std()
            var_diario_95 = stats.norm.ppf(0.95, media_c, sigma_c)
            
            # CVaR (Pérdida esperada en el peor 5% de los días)
            peores_rendimientos = rendimientos_cartera[rendimientos_cartera <= -var_diario_95]
            cvar_diario_95 = -peores_rendimientos.mean() if len(peores_rendimientos) > 0 else var_diario_95
            
            # --- 6. RENDERIZADO DE RESULTADOS ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success("### 🎯 Cartera Óptima (Sharpe)")
                st.metric("Retorno Anual Esperado", f"{resultados[0, idx_max_sharpe]:.2%}")
                st.metric("Volatilidad Anual", f"{resultados[1, idx_max_sharpe]:.2%}")
                
            with col2:
                st.error("### 🛡️ Métricas de Riesgo Extremo")
                st.metric("Value at Risk (VaR 95% Diario)", f"{var_diario_95:.2%}", 
                          help="Con un 95% de probabilidad, lo máximo que se espera perder en UN solo día es este porcentaje.")
                st.metric("Conditional VaR (CVaR 95%)", f"{cvar_diario_95:.2%}", 
                          help="Si el mercado entra en el 5% de sus peores días históricos, esta sería la pérdida promedio estimada por día.")
                
            with col3:
                st.info("### 📊 Asignación de Capital")
                df_optimo = pd.DataFrame({'Activo': activos, 'Porcentaje (%)': pesos_optimos * 100})
                st.dataframe(df_optimo.style.format({'Porcentaje (%)': '{:.2f}%'}), use_container_width=True)
            
            # Frontera Eficiente
            st.markdown("---")
            col_graf1, col_graf2 = st.columns([1, 1])
            
            with col_graf1:
                st.subheader("📈 Frontera Eficiente de Markowitz")
                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(resultados[1,:], resultados[0,:], c=resultados[2,:], cmap='viridis', s=8, alpha=0.3)
                fig.colorbar(scatter, ax=ax, label='Ratio de Sharpe')
                ax.scatter(resultados[1, idx_max_sharpe], resultados[0, idx_max_sharpe], color='red', marker='*', s=200, label='Cartera Óptima')
                ax.set_xlabel('Volatilidad Anualizada')
                ax.set_ylabel('Retorno Anual Esperado')
                ax.legend()
                ax.grid(True, alpha=0.2)
                st.pyplot(fig)

            # --- 7. MÓDULO DE PRUEBAS DE ESTRÉS (CRISIS HISTÓRICAS) ---
            with col_graf2:
                if crisis_seleccionada != "Ninguna":
                    st.subheader(f"⚡ Simulación de Estrés: {crisis_seleccionada}")
                    inicio_c, fin_c = fechas_crisis[crisis_seleccionada]
                    
                    # Descargar datos específicos del período de crisis
                    datos_crisis_raw = yf.download(activos, start=inicio_c, end=fin_c)
                    
                    if not datos_crisis_raw.empty and 'Close' in datos_crisis_raw:
                        datos_crisis = datos_crisis_raw['Close'].dropna()
                        # Normalizar los precios a base 100 al inicio de la crisis para ver la evolución
                        datos_normalizados = (datos_crisis / datos_crisis.iloc[0]) * 100
                        
                        # Evolución de la cartera simulada en la crisis aplicando nuestros pesos óptimos
                        evolucion_cartera = datos_normalizados.dot(pesos_optimos)
                        
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        ax2.plot(evolucion_cartera.index, evolucion_cartera.values, label="Mi Cartera Óptima", color='red', linewidth=2.5)
                        
                        # Graficar también los activos individuales de fondo para comparar
                        for col in datos_normalizados.columns:
                            ax2.plot(datos_normalizados.index, datos_normalizados[col].values, alpha=0.4, linestyle='--', label=col)
                            
                        ax2.set_ylabel('Evolución del Capital (Base 100)')
                        ax2.set_title('Comportamiento simulado del capital durante la crisis')
                        ax2.legend(loc='lower left', fontsize='small')
                        ax2.grid(True, alpha=0.2)
                        plt.xticks(rotation=45)
                        st.pyplot(fig2)
                    else:
                        st.error("No hay suficientes datos históricos de estos activos para simular este período de crisis.")
                else:
                    st.info("💡 **Prueba de Estrés:** Selecciona una crisis en la barra lateral izquierda y vuelve a optimizar para ver cómo habría respondido tu capital ante un crash financiero real.")
                    
        except Exception as e:
            st.error(f"Error cuantitativo: {e}")
