import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title('Análisis de Boston Housing')

# Cargar datos
def cargar_datos():
    return pd.read_csv('housing.csv')

df = cargar_datos()
st.write('Vista previa de los datos:')
st.dataframe(df.head())

# Estadísticas descriptivas
st.subheader('Estadísticas descriptivas')
st.write(df.describe())

# Visualización de correlación
st.subheader('Mapa de calor de correlación')
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Visualización interactiva con Plotly
st.subheader('Visualización interactiva')
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox('Variable en X', df.columns)
with col2:
    y_axis = st.selectbox('Variable en Y', df.columns)
fig2 = px.scatter(df, x=x_axis, y=y_axis)
st.plotly_chart(fig2)

# Modelo de regresión lineal
st.subheader('Modelo de Regresión Lineal')
objetivo = st.selectbox('Variable objetivo', df.columns)
caracteristicas = st.multiselect('Variables predictoras', [col for col in df.columns if col != objetivo])

if caracteristicas and objetivo:
    X = df[caracteristicas]
    y = df[objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    st.write('MSE:', mean_squared_error(y_test, y_pred))
    st.write('R2:', r2_score(y_test, y_pred))
    st.write('Coeficientes:', dict(zip(caracteristicas, modelo.coef_)))
