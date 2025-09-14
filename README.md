# Used car prices Forecast
Plataforma para ayudar a predecir precios justos de autos usados y explicar los resultados de manera transparente.

## Descripción
Esta aplicación permite a los usuarios estimar el precio justo de vehículos usados basado en sus características. La herramienta no solo predice precios, sino que también proporciona explicaciones claras y visualizaciones para que tanto compradores como vendedores puedan entender los factores que influyen en el precio.

## Funcionalidades
- Predicción de precios basada en modelo de machine learning
- Comparación con vehículos similares del mercado
- Explicación visual de los factores que influyen en el precio
- Alertas de precios inusuales (muy altos o muy bajos)
- Exploración de tendencias de precios en el mercado
- Estadísticas por marca, año y otras características

## Dataset
Este conjunto de datos comprende 4009 puntos de datos, cada uno representando un listado único de vehículos, e incluye nueve características distintivas que ofrecen información valiosa sobre el mundo automotriz.
Las características incluidas en el conjunto de datos son las siguientes:
- **Brand & Model**: La marca y el modelo del vehículo.  
- **Model Year**: El año de fabricación del vehículo.  
- **Mileage**: El kilometraje del vehículo.  
- **Fuel Type**: El tipo de combustible que utiliza el vehículo.  
- **Engine Type**: Las especificaciones del motor del vehículo.  
- **Transmission**: El tipo de transmisión del vehículo.  
- **Exterior & Interior Colors**: Los colores exteriores e interiores del vehículo.  
- **Accident History**: El historial de accidentes del vehículo.  
- **Clean Title**: La disponibilidad de un título limpio para el vehículo.  
- **Price**: El precio listado del vehículo.  

[Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data)

## Preprocessing
El conjunto de datos se ha limpiado y preprocesado para asegurar su calidad y relevancia para el análisis. Las siguientes acciones se han llevado a cabo:
- **Eliminación de valores nulos**: Se han eliminado las filas con valores nulos en columnas críticas.
- **Conversión de tipos de datos**: Se han convertido las columnas a los tipos de datos adecuados (por ejemplo, fechas a datetime, categorías a category).
- **Codificación de variables categóricas**: Se han codificado las variables categóricas utilizando técnicas como One-Hot Encoding o Label Encoding.
- **Normalización/Estandarización**: Se han normalizado o estandarizado las características numéricas para mejorar el rendimiento del modelo.
- **Eliminación de outliers**: Se han identificado y eliminado los valores atípicos que podrían sesgar el análisis.


## Model
Como modelo se ha utilizado la libreria sklearn con varios modelos de regresión, entre ellos:
- Linear Regression
- Polynomial Regression (grado 2)
- Ridge Regression (regularización L2)
- Lasso Regression (regularización L1)
- ElasticNet Regression (combinación de L1 y L2)
Estos modelos fueron evaluados utilizando validación cruzada con 5 pliegues (K-Fold Cross-Validation) y la métrica de error cuadrático medio (MSE) para determinar su rendimiento.

Para la seleción del mejor modelo, se compararon los valores promedio de MSE obtenidos en la validación cruzada. El modelo con el menor MSE promedio fue seleccionado como el mejor modelo para predecir los precios de los autos usados.

## Despliegue en Streamlit Cloud

Esta aplicación está configurada para ser desplegada en Streamlit Cloud. Para implementarla, sigue estos pasos:

1. Asegúrate de tener una cuenta en [Streamlit Cloud](https://streamlit.io/cloud)
2. Conecta tu repositorio de GitHub a Streamlit Cloud
3. Selecciona este repositorio y configura:
   - **Main file path**: `streamlit_app.py`
   - No se requieren secrets para esta aplicación

La aplicación utilizará automáticamente:
- El archivo `requirements.txt` para instalar las dependencias
- El modelo preentrenado `best_model.pkl` 
- El conjunto de datos `used_cars.csv`

### Requisitos del sistema

- Python 3.8 o superior
- Bibliotecas específicas listadas en `requirements.txt`
- Streamlit versión 1.30.0 o superior

### Uso en la nube

Una vez desplegada, la aplicación estará disponible en una URL proporcionada por Streamlit Cloud. Los usuarios podrán:
- Predecir precios de autos usados
- Explorar tendencias del mercado
- Ver estadísticas y visualizaciones interactivas
