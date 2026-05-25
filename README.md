# 📊 Optimizador Inteligente de Portafolios (Teoría de Markowitz)

¡Bienvenido! Este proyecto consiste en una **Aplicación Web Interactiva** desarrollada en Python que automatiza la **Teoría Moderna de Portafolios de Harry Markowitz** (Premio Nobel de Economía). 

La herramienta permite a analistas financieros e inversores simular miles de combinaciones de activos en tiempo real a través de una simulación de *Montecarlo*, identificando las fronteras de inversión más eficientes según el perfil de riesgo.

---

## 🎯 ¿Qué resuelve esta aplicación?

En el mercado financiero, maximizar el rendimiento ignorando el riesgo es un error crítico. Esta app resuelve la optimización analizando datos históricos reales directamente desde la API de **Yahoo Finance** para calcular:

1. **La Cartera Óptima (Máxima Eficiencia):** Encuentra la distribución de pesos que maximiza el *Ratio de Sharpe* (el mejor rendimiento por cada unidad de riesgo asumida).
2. **La Cartera de Mínimo Riesgo:** Encuentra la combinación con la menor volatilidad posible, ideal para perfiles marcadamente conservadores.

---

## 🛠️ Tecnologías y Librerías Utilizadas

El proyecto fue desarrollado utilizando el ecosistema de Ciencia de Datos y Finanzas Cuantitativas de Python:

* **Python 3** (Lógica del motor financiero)
* **Streamlit** (Desarrollo de la interfaz web interactiva)
* **YFinance** (Extracción de datos de mercado en tiempo real)
* **Pandas & NumPy** (Procesamiento de matrices de covarianza, rendimientos logarítmicos y vectores de pesos)
* **Matplotlib** (Visualización estática y renderizado de la Frontera Eficiente)

---

## 📐 Conceptos Financieros Automatizados

La aplicación realiza de fondo los siguientes cálculos de economía financiera:

| Concepto Financiero | Implementación en Código |
| :--- | :--- |
| **Rendimientos Anualizados** | Media geométrica de retornos diarios multiplicada por las 252 ruedas del año macroeconómico. |
| **Riesgo del Portafolio** | Raíz cuadrada del producto matricial entre el vector de pesos transpuesto y la matriz de covarianza de los activos ($w^T \cdot \Sigma \cdot w$). |
| **Simulación de Montecarlo** | Modelado estocástico mediante la generación de 20.000 vectores de pesos aleatorios cuya sumatoria es estrictamente igual a 1 ($\sum w_i = 1$). |

---

## 🚀 Validaciones Inteligentes de Usuario (Robustez del Sistema)

Para garantizar una excelente experiencia de usuario (UX) y evitar fallas en la ejecución, el script incluye un sistema de alertas proactivas:
* **Validación Temporal:** Bloquea la ejecución si la fecha de inicio es posterior a la de fin o si se configuran fechas en el futuro.
* **Consistencia de Datos:** Evalúa la densidad de datos históricos de cada ticker y emite una advertencia amarilla con sugerencias si un activo posee menos del 80% del histórico requerido (por ejemplo, empresas de reciente cotización/IPOs).

---

## 📂 Cómo ejecutar el proyecto localmente

Si querés clonar este repositorio y correrlo en tu máquina, ejecutá los siguientes comandos en tu terminal:

git clone https://github.com/tu-usuario/optimizador-portafolios-markowitz.git
pip install streamlit yfinance pandas numpy matplotlib
streamlit run app.py
