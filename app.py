import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from datos import cargar_datos

# Configuración de la página con tema claro y diseño limpio
st.set_page_config(
    page_title="Pronóstico de Pasajeros de Avión",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo personalizado con colores limpios y texto oscuro muy legible
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        color: #111827;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    p, span, label, div {
        color: #374151 !important;
    }
    .stMetric {
        background-color: #f3f4f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #2563eb;
    }
    .stMetric label {
        color: #4b5563 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e5e7eb;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        color: #374151 !important;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    .stTabs [aria-selected="true"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.title("¿Cuántos pasajeros tendremos el próximo año?")

# Cargar datos
df = cargar_datos()

# Asegurar formato de fecha y orden
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month')

# --- Indicadores Superiores ---
total_1949 = df[df['Month'].dt.year == 1949]['Passengers'].sum()
total_1960 = df[df['Month'].dt.year == 1960]['Passengers'].sum()
crecimiento = total_1960 / total_1949

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Pasajeros totales en 1949", value=f"{total_1949:,.0f}")
with col2:
    st.metric(label="Pasajeros totales en 1960", value=f"{total_1960:,.0f}")
with col3:
    st.metric(label="Crecimiento (veces)", value=f"{crecimiento:.2f}x")

st.markdown("---")

# --- Pestañas ---
tab1, tab2, tab3, tab4 = st.tabs(["1. La historia", "2. La temporada", "3. El examen", "4. El pronóstico"])

# Pestaña 1: La historia
with tab1:
    st.subheader("Evolución mensual de pasajeros")
    
    fig1 = px.line(
        df, x='Month', y='Passengers',
        labels={'Month': 'Año / Mes', 'Passengers': 'Número de pasajeros'},
        title="Pasajeros de aerolínea a lo largo del tiempo"
    )
    fig1.update_traces(line=dict(color='#10B981', width=3))
    fig1.update_layout(
        plot_bgcolor='rgba(240,253,244,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="sans-serif", size=12, color="#065F46")
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.info(
        "Este gráfico muestra cómo ha crecido el número de pasajeros mes a mes a lo largo de los años. "
        "Se observa una tendencia clara al alza y que cada año el volumen total aumenta considerablemente, "
        "lo que refleja un negocio en plena expansión y con gran demanda sostenida."
    )

# Pestaña 2: La temporada
with tab2:
    st.subheader("Comportamiento estacional (mes a mes)")
    
    df['Nombre_Mes'] = df['Month'].dt.strftime('%B')
    df['Mes_Num'] = df['Month'].dt.month
    
    promedio_mes = df.groupby(['Mes_Num', 'Nombre_Mes'])['Passengers'].mean().reset_index()
    promedio_mes = promedio_mes.sort_values('Mes_Num')
    
    fig2 = px.bar(
        promedio_mes, x='Nombre_Mes', y='Passengers',
        labels={'Nombre_Mes': 'Mes del año', 'Passengers': 'Promedio de Pasajeros'},
        title="Promedio de pasajeros por mes (Temporada alta y baja)"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info(
        "Aquí comparamos el promedio de pasajeros en cada mes del año para identificar patrones fijos. "
        "Los meses de mediados de año (como verano) muestran picos altos de demanda, mientras que otros meses registran menor actividad, "
        "lo cual ayuda al negocio a planificar personal y flota según la temporada."
    )

# Pestaña 3: El examen
with tab3:
    st.subheader("Evaluación de múltiples modelos con los últimos 24 meses")
    
    train = df.iloc[:-24].copy()
    test = df.iloc[-24:].copy()
    
    # 1. Copiar el año pasado
    copia_anio_pasado = df.iloc[-36:-12]['Passengers'].values
    
    # 2. Suavizamiento Exponencial (Holt-Winters)
    hw_model = ExponentialSmoothing(
        train['Passengers'],
        trend="add",
        seasonal="mul",
        seasonal_periods=12
    ).fit()
    pred_hw = hw_model.forecast(24)
    
    # 3. SARIMA
    try:
        sarima_model = SARIMAX(
            train['Passengers'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        pred_sarima = sarima_model.forecast(24)
    except Exception:
        pred_sarima = pred_hw # fallback si hubiera convergencia
        
    # 4. Regresión Lineal
    train_reg = train.copy()
    train_reg['Time_Index'] = np.arange(len(train_reg))
    lr_model = LinearRegression()
    lr_model.fit(train_reg[['Time_Index']], train_reg['Passengers'])
    
    test_indices = np.arange(len(train), len(train) + 24).reshape(-1, 1)
    pred_lr = lr_model.predict(test_indices)
    
    # 5. Prophet
    df_prophet = train[['Month', 'Passengers']].rename(columns={'Month': 'ds', 'Passengers': 'y'})
    m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m_prophet.fit(df_prophet)
    future_prophet = m_prophet.make_future_dataframe(periods=24, freq='MS')
    forecast_prophet = m_prophet.predict(future_prophet)
    pred_prophet = forecast_prophet.tail(24)['yhat'].values

    # Gráfico de evaluación
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=test['Month'], y=test['Passengers'], mode='lines+markers', name='Real', line=dict(color='black', width=3)))
    fig3.add_trace(go.Scatter(x=test['Month'], y=copia_anio_pasado, mode='lines', name='Copiar año pasado', line=dict(dash='dash')))
    fig3.add_trace(go.Scatter(x=test['Month'], y=pred_hw, mode='lines', name='Suavizamiento Exponencial'))
    fig3.add_trace(go.Scatter(x=test['Month'], y=pred_sarima, mode='lines', name='SARIMA'))
    fig3.add_trace(go.Scatter(x=test['Month'], y=pred_lr, mode='lines', name='Regresión Lineal'))
    fig3.add_trace(go.Scatter(x=test['Month'], y=pred_prophet, mode='lines', name='Prophet'))
    
    fig3.update_layout(title="Comparativa de Modelos vs Real (Últimos 24 meses)", xaxis_title="Mes", yaxis_title="Pasajeros")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Calcular errores (MAPE)
    real_vals = test['Passengers'].values
    err_copia = np.mean(np.abs(real_vals - copia_anio_pasado) / real_vals) * 100
    err_hw = np.mean(np.abs(real_vals - pred_hw.values) / real_vals) * 100
    err_sarima = np.mean(np.abs(real_vals - pred_sarima.values) / real_vals) * 100
    err_lr = np.mean(np.abs(real_vals - pred_lr) / real_vals) * 100
    err_prophet = np.mean(np.abs(real_vals - pred_prophet) / real_vals) * 100
    
    df_errores = pd.DataFrame({
        'Modelo': ['Copiar año pasado', 'Suavizamiento Exponencial', 'SARIMA', 'Regresión Lineal', 'Prophet'],
        'Error promedio (%)': [
            f"{err_copia:.2f}%", 
            f"{err_hw:.2f}%", 
            f"{err_sarima:.2f}%", 
            f"{err_lr:.2f}%", 
            f"{err_prophet:.2f}%"
        ]
    })
    
    st.table(df_errores)
    
    st.info(
        "Aquí comparamos el desempeño de todos los modelos entrenados frente a los datos reales de los últimos dos años. "
        "Esto permite observar qué técnica matemática se ajusta mejor a la estacionalidad y tendencia del negocio."
    )

# Pestaña 4: El pronóstico
with tab4:
    st.subheader("Pronóstico para los 12 meses de 1961 con múltiples modelos")
    
    modelo_elegido = st.selectbox(
        "Selecciona el modelo para proyectar 1961:",
        ["Suavizamiento Exponencial", "SARIMA", "Regresión Lineal", "Prophet", "Copiar año pasado"]
    )
    
    fechas_1961 = pd.date_range(start='1961-01-01', periods=12, freq='MS')
    
    if modelo_elegido == "Suavizamiento Exponencial":
        mod_t = ExponentialSmoothing(df['Passengers'], trend="add", seasonal="mul", seasonal_periods=12).fit()
        pred_1961 = mod_t.forecast(12).values
    elif modelo_elegido == "SARIMA":
        mod_t = SARIMAX(df['Passengers'], order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred_1961 = mod_t.forecast(12).values
    elif modelo_elegido == "Regresión Lineal":
        df_lr = df.copy()
        df_lr['Time_Index'] = np.arange(len(df_lr))
        lr_t = LinearRegression()
        lr_t.fit(df_lr[['Time_Index']], df_lr['Passengers'])
        fut_idx = np.arange(len(df_lr), len(df_lr) + 12).reshape(-1, 1)
        pred_1961 = lr_t.predict(fut_idx)
    elif modelo_elegido == "Prophet":
        df_p = df[['Month', 'Passengers']].rename(columns={'Month': 'ds', 'Passengers': 'y'})
        mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        mp.fit(df_p)
        fut_p = mp.make_future_dataframe(periods=12, freq='MS')
        pred_1961 = mp.predict(fut_p).tail(12)['yhat'].values
    else: # Copiar año pasado
        pred_1961 = df.iloc[-12]['Passengers'].values # o valores de 1960
        pred_1961 = df.iloc[-24:-12]['Passengers'].values

    df_futuro = pd.DataFrame({
        'Month': fechas_1961,
        'Passengers': pred_1961
    })
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df['Month'], y=df['Passengers'], mode='lines', name='Histórico'))
    fig4.add_trace(go.Scatter(x=df_futuro['Month'], y=df_futuro['Passengers'], mode='lines+markers', name=f'Pronóstico 1961 ({modelo_elegido})', line=dict(color='red', dash='dash')))
    
    fig4.update_layout(title=f"Pronóstico para 1961 usando {modelo_elegido}", xaxis_title="Mes", yaxis_title="Pasajeros")
    st.plotly_chart(fig4, use_container_width=True)
    
    df_tabla_futuro = df_futuro.copy()
    df_tabla_futuro['Mes'] = df_tabla_futuro['Month'].dt.strftime('%Y-%m')
    df_tabla_futuro['Pasajeros Esperados'] = df_tabla_futuro['Passengers'].round(0).astype(int)
    df_tabla_futuro = df_tabla_futuro[['Mes', 'Pasajeros Esperados']]
    
    st.table(df_tabla_futuro)
    
    st.info(
        f"Este pronóstico utiliza el modelo **{modelo_elegido}** para estimar la demanda mes a mes del año 1961. "
        "Puedes cambiar de modelo en el selector superior para comparar las distintas proyecciones comerciales."
    )
