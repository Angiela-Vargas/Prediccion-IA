import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

# Cargar el modelo
modelo = tf.keras.models.load_model('modelo_entrenado.h5')

# Título de la aplicación
st.title('Predicción de Beneficios Educativos - Becas y créditos')

# Agregar la descripción en la barra lateral
st.sidebar.title("***Descripción de las opciones:***")
st.sidebar.write(" **Género**: ")
st.sidebar.write("- 0 FEMENINO")
st.sidebar.write("- 1 MASCULINO")
st.sidebar.write(" **Grupo Étnico**: ")
st.sidebar.write("- 0 NINGUNO")
st.sidebar.write("- 1 INDÍGENA")
st.sidebar.write("- 2 AFROCOLOMBIANO")
st.sidebar.write(" **Víctima del Conflicto Armado**: ")
st.sidebar.write("- 0 NO")
st.sidebar.write("- 1 SI")
st.sidebar.write(" **Tipo de Formación**: ")
st.sidebar.write("- 0 NORMALISTA")
st.sidebar.write("- 1 TÉCNICA PROFESIONAL")
st.sidebar.write("- 2 TECNOLÓGICA")
st.sidebar.write("- 3 UNIVERSITARIA")
st.sidebar.write("- 4 POSTGRADO")


# Entradas del usuario
semestre = st.selectbox('Semestre de Convocatoria', [1, 2])
genero = st.selectbox('Género', [0, 1])  # 0: FEMENINO, 1: MASCULINO
estrato = st.selectbox('Estrato', [1, 2, 3, 4, 5])
grupo_etnico = st.selectbox('Grupo Étnico', [0, 1, 2])  # 0: NINGUNO, 1: INDIGENA, 2: AFROCOLOMBIANO
victima_conflicto = st.selectbox('Víctima del Conflicto Armado', [0, 1])  # 0: NO, 1: SI
tipo_formacion = st.selectbox('Tipo de Formación', [0, 1, 2, 3, 4])  # 0: NORMALISTA, 1: TECNICA PROFESIONAL, 2: TECNOLOGICA, 3: UNIVERSITARIA, 4: POSTGRADO

# Crear el dataframe con las características
input_data = pd.DataFrame({
    'SEMESTRE': [semestre],
    'GÉNERO': [genero],
    'ESTRATO': [estrato],
    'GRUPO ETNICO': [grupo_etnico],
    'VICTIMA DEL CONFLICTO ARMADO': [victima_conflicto],
    'TIPO DE FORMACIÓN': [tipo_formacion]
})

# Convertir las columnas al tipo float
input_data = input_data.astype(float)

# Realizar la predicción
if st.button('Predecir'):
    prediccion = modelo.predict(input_data)
    #resultado = int(np.round(prediccion[0, 0]))  # Redondear y convertir a entero
    resultado = np.argmax(prediccion)  # Obtener el índice del valor máximo
    beneficios = {0: 'MATRICULA', 1: 'SOSTENIMIENTO', 2: 'MATRICULA Y SOSTENIMIENTO'}
    st.write(f'El beneficio otorgado es: {beneficios.get(resultado, "Desconocido")}')


# Escribir en la terminal para ejecutar la vista
# streamlit run d:/Uni/4to/IA/Project_IA/app.py
