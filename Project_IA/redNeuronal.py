import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('Beneficiaros_de_becas_y_creditos_de_programas_de_acceso_a_la_educaci_n_superior_de_Antioquia_20240526.csv')
df.head()

df = df.drop(['DEPARTAMENTO DE NACIMIENTO', 'CONVOCATORIA', 'MUNICIPIO DE NACIMIENTO', 'FECHA DE NACIMIENTO', 'MUNICIPIO DE RESIDENCIA', 'SUBREGIÓN DE RESIDENCIA', 'UNIVERSIDAD', 'MUNICIPIO OFERTA', 'PROGRAMA CURSADO', 'GRADUADO'], axis=1)
df.rename(columns={'BENEFICIO OTORGADO': 'BENEFICIO_OTORGADO'}, inplace=True)
df.rename(columns={'SEMESTRE DE CONVOCATORIA': 'SEMESTRE'}, inplace=True)
df

# Se eliminan las filas que contengan algún valor vacío
df = df.dropna()

# Se cambian los BENEFICIO_OTORGADO
df['BENEFICIO_OTORGADO'] = df['BENEFICIO_OTORGADO'].replace('MATRICULA', 0)
df['BENEFICIO_OTORGADO'] = df['BENEFICIO_OTORGADO'].replace('SOSTENIMIENTO', 1)
df['BENEFICIO_OTORGADO'] = df['BENEFICIO_OTORGADO'].replace('MATRICULA Y SOSTENIMIENTO', 2)

# Se cambian los GÉNERO
df['GÉNERO'] = df['GÉNERO'].replace('FEMENINO', 0) 
df['GÉNERO'] = df['GÉNERO'].replace('MASCULINO', 1)

# Se cambian los ESTRATO
df['ESTRATO'] = df['ESTRATO'].replace('ESTRATO 1', 1) 
df['ESTRATO'] = df['ESTRATO'].replace('Estrato 1', 1) 
df['ESTRATO'] = df['ESTRATO'].replace('ESTRATO 2', 2)
df['ESTRATO'] = df['ESTRATO'].replace('Estrato 2', 2) 
df['ESTRATO'] = df['ESTRATO'].replace('ESTRATO 3', 3)
df['ESTRATO'] = df['ESTRATO'].replace('Estrato 3', 3) 
df['ESTRATO'] = df['ESTRATO'].replace('ESTRATO 4', 4)
df['ESTRATO'] = df['ESTRATO'].replace('ESTRATO 5', 5)

# Se cambian los GRUPO ETNICO
df['GRUPO ETNICO'] = df['GRUPO ETNICO'].replace('NINGUNO', 0)
df['GRUPO ETNICO'] = df['GRUPO ETNICO'].replace('INDIGENA', 1) 
df['GRUPO ETNICO'] = df['GRUPO ETNICO'].replace('AFROCOLOMBIANO', 2) 

# Se cambian las VICTIMA DEL CONFLICTO ARMADO
df['VICTIMA DEL CONFLICTO ARMADO'] = df['VICTIMA DEL CONFLICTO ARMADO'].replace('NO', 0)
df['VICTIMA DEL CONFLICTO ARMADO'] = df['VICTIMA DEL CONFLICTO ARMADO'].replace('SI', 1)

# Se cambian los TIPO DE FORMACIÓN
df['TIPO DE FORMACIÓN'] = df['TIPO DE FORMACIÓN'].replace('NORMALISTA', 0)
df['TIPO DE FORMACIÓN'] = df['TIPO DE FORMACIÓN'].replace('TECNICA PROFESIONAL', 1)
df['TIPO DE FORMACIÓN'] = df['TIPO DE FORMACIÓN'].replace('TECNOLOGICA', 2)
df['TIPO DE FORMACIÓN'] = df['TIPO DE FORMACIÓN'].replace('UNIVERSITARIA', 3)
df['TIPO DE FORMACIÓN'] = df['TIPO DE FORMACIÓN'].replace('Universitaria', 3)
df['TIPO DE FORMACIÓN'] = df['TIPO DE FORMACIÓN'].replace('POSTGRADO', 4)

# Convertir las columnas al tipo float
columns_to_convert = ['SEMESTRE', 'BENEFICIO_OTORGADO', 'GÉNERO', 'ESTRATO', 'GRUPO ETNICO', 'VICTIMA DEL CONFLICTO ARMADO', 'TIPO DE FORMACIÓN']
df[columns_to_convert[1:]] = df[columns_to_convert[1:]].astype(float)

df

features = ['SEMESTRE', 'GÉNERO', 'ESTRATO', 'GRUPO ETNICO', 'VICTIMA DEL CONFLICTO ARMADO', 'TIPO DE FORMACIÓN']
# target = ['BENEFICIO_OTORGADO']

X = df[features]
y = df.BENEFICIO_OTORGADO

# Check the number of samples in X and y
if X.shape[0] != y.shape[0]:
    raise ValueError("The number of samples in X and y must be the same.")

# Divide the data and assign a 70% for training and a 30% for prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print('Dimensiones del X train set: ', X_train.shape)
print('Dimensiones del y train set: ', y_train.shape)
print('Dimensiones del X test set: ',  X_test.shape)
print('Dimensiones del y test set: ',  y_test.shape)

# Crea un objeto con el árbol de decisión
clf = DecisionTreeClassifier()

# Asigna el árbol clasificador del conjunto de datos de entrenamiento
clf = clf.fit(X_train,y_train)

# Predice el árbol de decisión del conjunto de datos de testeo
y_pred = clf.predict(X_test)

# Mediante la métrica se analiza qué tan precisa es la predicción al evaluar lo que se predijo con respecto al testeo
print("Árbol de decisión")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Crear el modelo usando la función de activación relu
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=12, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),  # Añadir dropout para regularización
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Añadir dropout para regularización
    tf.keras.layers.Dense(units=3, activation='softmax')
])

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar el modelo
print('Inicio de entrenamiento...')
historial = modelo.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=True)
print('Modelo entrenado!')

# Guardar el modelo
modelo.save('modelo_entrenado.h5')

# Visualizar las pérdidas de entrenamiento y validación
plt.xlabel('# Época')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.legend()
plt.show()

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = modelo.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')