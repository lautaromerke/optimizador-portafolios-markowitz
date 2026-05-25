import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# 1. Configuración de la página web
st.set_page_config(page_title="Optimizador de Portafolios Inteligente", layout="wide")

st.title("📊 Optimizador Inteligente de Portafolios")
st.markdown("""
Esta aplicación web automatiza la **Teoría Moderna de Portafolios de Harry Markowitz**. 
Permite simular miles de combinaciones de activos para encontrar la distribución óptima entre **Rendimiento** y **Riesgo**.
""")

st.sidebar.header("🔧 Configuración del Portafolio")

# 2. Inputs interactivos
lista_tickers = st.sidebar.text_input("Introduce los Tickers (separados por coma):", "AAPL, MSFT, GOOGL, KO, TSLA")
fecha_inicio = st.sidebar.date_input("Fecha de inicio:", pd.to_datetime("2021-01-01"))
fecha_fin = st.sidebar.date_input("Fecha de fin:", pd.to_datetime("2026-05-25"))
num_portafolios = st.sidebar.slider("Número de portafolios a simular:", 5000, 50000, 20000, step=5000)

activos = [x.strip().upper() for x in lista_tickers.split(",")]

# Botón para ejecutar la optimización
if st.sidebar.button("🚀 Optimizar Portafolio"):
    with st.spinner("Descargando datos del mercado y simulando escenarios..."):
        try:
            # --- VALIDACIÓN DE FECHAS LÓGICAS ---
            hoy = date.today()

            if fecha_inicio >= fecha_fin:
                st.warning("⚠️ **Error en el rango:** La fecha de inicio no puede ser posterior o igual a la fecha de fin. Por favor, corregí las fechas en la barra lateral.")
                st.stop()

            if fecha_inicio > hoy:
                st.warning(f"⚠️ **Fecha en el futuro:** Elegiste una fecha de inicio ({fecha_inicio.strftime('%d/%m/%Y')}) que está en el futuro. Seleccioná una fecha anterior al día de hoy.")
                st.stop()
            # ------------------------------------

            # 3. Descarga de datos
            datos_completos = yf.download(activos, start=fecha_inicio, end=fecha_fin)

            if datos_completos.empty or 'Close' not in datos_completos:
                st.warning("⚠️ **¡Atención!** No se encontraron datos para los tickers ingresados en este período. Revisá que los nombres estén bien escritos.")
                st.stop()

            datos = datos_completos['Close']

            # Verificar consistencia por activo
            datos_por_activo = datos.notna().sum()
            max_datos_posibles = len(datos)

            activos_incompletos = []
            for col in datos.columns:
                if datos_por_activo[col] < (max_datos_posibles * 0.8):
                    activos_incompletos.append(col)

            if activos_incompletos:
                st.warning(f"""
                ⚠️ **Alerta de consistencia en las fechas:** El activo o los activos **{', '.join(activos_incompletos)}** tienen muy pocos datos históricos para el rango seleccionado (desde el {fecha_inicio.strftime('%d/%m/%Y')}). 

                **Sugerencia:** Esto suele pasar si la empresa empezó a cotizar en bolsa *después* de la fecha de inicio que elegiste. Probá achicar el rango poniendo una **Fecha de inicio más reciente** en la barra lateral.
                """)
                st.stop()

            # 4. Cálculos estadísticos
            datos = datos.dropna()
            rendimientos = datos.pct_change().dropna()
            retornos_anuales = rendimientos.mean() * 252
            matriz_covarianza = rendimientos.cov() * 252

            # 5. Simulación de Montecarlo
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
            idx_min_vol = resultados[1].argmin()

            # 6. Mostrar resultados en la Web
            col1, col2 = st.columns(2)

            with col1:
                st.success("### 🎯 Cartera Óptima (Máxima Eficiencia)")
                st.metric("Retorno Anual Esperado", f"{resultados[0, idx_max_sharpe]:.2%}")
                st.metric("Riesgo (Volatilidad)", f"{resultados[1, idx_max_sharpe]:.2%}")

                df_optimo = pd.DataFrame({'Activo': activos, 'Porcentaje (%)': lista_pesos[idx_max_sharpe] * 100})
                st.dataframe(df_optimo.style.format({'Porcentaje (%)': '{:.2f}%'}))

            with col2:
                st.info("### 🛡️ Cartera de Mínimo Riesgo")
                st.metric("Retorno Anual Esperado", f"{resultados[0, idx_min_vol]:.2%}")
                st.metric("Riesgo (Volatilidad)", f"{resultados[1, idx_min_vol]:.2%}")

                df_min_riesgo = pd.DataFrame({'Activo': activos, 'Porcentaje (%)': lista_pesos[idx_min_vol] * 100})
                st.dataframe(df_min_riesgo.style.format({'Porcentaje (%)': '{:.2f}%'}))

            # 7. Gráfico de la Frontera Eficiente
            st.markdown("---")
            st.subheader("📈 Gráfico Interactivo: Frontera Eficiente")

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(resultados[1,:], resultados[0,:], c=resultados[2,:], cmap='viridis', s=10, alpha=0.4)
            fig.colorbar(scatter, ax=ax, label='Ratio de Sharpe')

            ax.scatter(resultados[1, idx_max_sharpe], resultados[0, idx_max_sharpe], color='red', marker='*', s=200, label='Cartera Óptima')
            ax.scatter(resultados[1, idx_min_vol], resultados[0, idx_min_vol], color='blue', marker='*', s=200, label='Mínimo Riesgo')

            ax.set_xlabel('Riesgo Anualizado (Volatilidad)')
            ax.set_ylabel('Retorno Anual Esperado')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar los datos: {e}")
