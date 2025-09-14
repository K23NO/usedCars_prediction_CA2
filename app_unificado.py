# app.py — Evaluador de Autos Usados (Versión Fusionada)

import os
import time
import logging
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Precio Justo de Autos Usados", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("App iniciada")

# Constantes y Configuración
MODEL_PATH = "best_model.pkl"    # Ruta al modelo guardado
DATA_PATH = "used_cars.csv"      # Ruta al dataset

# Función para limpiar datos numéricos (precio, millaje)
def clean_numeric(x):
    if isinstance(x, str):
        # Extraer solo los dígitos de la cadena
        digits = re.findall(r'\d+', x)
        if digits:
            return float(''.join(digits))
        return 0.0  # Si no hay dígitos, devolver 0
    elif pd.isna(x):
        return 0.0  # Si es NaN, devolver 0
    return float(x)  # En caso contrario, convertir a float

# Función para extraer características relacionadas con la edad del vehículo
def extract_age_features(df):
    current_year = 2025  # Año actual
    
    # Edad del vehículo
    df['Vehicle_Age'] = current_year - pd.to_numeric(df['model_year'], errors='coerce')
    
    # Millaje por año
    df['Mileage_per_Year'] = df['milage'] / df['Vehicle_Age'].replace(0, 1)  # Evitar división por cero
    
    return df

# Función para extraer otras características útiles
def extract_other_features(df):
    # Marcas de lujo
    luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                     'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                     'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
    
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    
    return df

# Función para preprocesar datos categóricos
def preprocess_categorical(df):
    # Umbral para valores raros
    threshold = 10
    
    # Columnas categóricas
    cat_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
    
    # Columnas a simplificar (reemplazar valores raros con "rare")
    cols_to_simplify = ['model', 'engine', 'transmission', 'ext_col', 'int_col']
    
    # Reemplazar valores raros con "rare"
    for col in cols_to_simplify:
        if col in df.columns:
            value_counts = df[col].value_counts(dropna=False)
            mask = df[col].map(value_counts) < threshold
            df.loc[mask, col] = "rare"
    
    # Rellenar valores nulos y convertir a categoría
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('missing')
            df[col] = df[col].astype('category')
    
    return df

# Cargar y preprocesar datos
@st.cache_data
def load_and_process_data():
    try:
        # Cargar datos
        df = pd.read_csv(DATA_PATH)
        
        # Limpiar precio y millaje
        df['price'] = df['price'].apply(clean_numeric)
        df['price_numeric'] = df['price'].copy()  # Para compatibilidad con app_minimal
        df['milage'] = df['milage'].apply(clean_numeric)
        
        # Aplicar transformaciones
        df = extract_age_features(df)
        df = extract_other_features(df)
        df = preprocess_categorical(df)
        
        # Verificar si hay valores nulos en columnas críticas
        critical_cols = ['price', 'milage', 'model_year']
        for col in critical_cols:
            if df[col].isnull().sum() > 0:
                logger.warning(f"Hay {df[col].isnull().sum()} valores nulos en {col}. Rellenando con valores predeterminados.")
                if col == 'price':
                    df[col] = df[col].fillna(df[col].median())
                elif col == 'milage':
                    df[col] = df[col].fillna(df[col].median())
                elif col == 'model_year':
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        logger.error(f"Error al cargar y procesar datos: {e}")
        st.error(f"Error al cargar y procesar datos: {e}")
        return None

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Modelo cargado exitosamente")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {e}")
        error_message = str(e)
        
        
        return None

# Función para encontrar vehículos similares
def find_similar_vehicles(df, user_input_dict, n_neighbors=5):
    # Crear un filtro basado en la marca y modelo del vehículo
    similar_df = df[
        (df['brand'] == user_input_dict['brand']) & 
        (df['model'] == user_input_dict['model']) & 
        (df['model_year'] >= user_input_dict['model_year'] - 3) &
        (df['model_year'] <= user_input_dict['model_year'] + 3)
    ]
    
    # Si no hay suficientes vehículos similares, ampliar la búsqueda solo a la marca
    if len(similar_df) < n_neighbors:
        similar_df = df[
            (df['brand'] == user_input_dict['brand']) & 
            (df['model_year'] >= user_input_dict['model_year'] - 5) &
            (df['model_year'] <= user_input_dict['model_year'] + 5)
        ]
    
    # Si aún no hay suficientes, devolver los primeros n_neighbors vehículos
    if len(similar_df) < n_neighbors and len(df) >= n_neighbors:
        # Ordenar por similitud en año y millaje
        df['year_diff'] = abs(df['model_year'] - user_input_dict['model_year'])
        df['milage_diff'] = abs(df['milage'] - user_input_dict['milage'])
        similar_df = df.sort_values(by=['year_diff', 'milage_diff']).head(n_neighbors)
    
    return similar_df.head(n_neighbors) if len(similar_df) >= n_neighbors else similar_df

# Función para calcular la diferencia porcentual
def calculate_percentage_diff(actual, predicted):
    return ((actual - predicted) / predicted) * 100

# Función para determinar si un precio es inusual
def is_unusual_price(price, similar_vehicles):
    mean_price = similar_vehicles['price'].mean()
    std_price = similar_vehicles['price'].std()
    
    # Definir límites de lo que se considera inusual (2 desviaciones estándar)
    lower_bound = mean_price - 2 * std_price
    upper_bound = mean_price + 2 * std_price
    
    if price < lower_bound:
        return True, "bajo", lower_bound
    elif price > upper_bound:
        return True, "alto", upper_bound
    else:
        return False, "normal", None

# Función para graficar la relación entre características y precio
def plot_feature_relationship(df, feature, user_value=None):
    if feature not in df.columns:
        return None
    
    # Crear scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[feature], df['price'], alpha=0.6, color='skyblue')
    
    # Añadir línea de tendencia
    m, b = np.polyfit(df[feature], df['price'], 1)
    ax.plot(df[feature], m*df[feature] + b, color='red', linestyle='-')
    
    # Añadir línea vertical para el valor del usuario
    if user_value is not None:
        ax.axvline(x=user_value, color='red', linestyle='--')
        ax.text(user_value, ax.get_ylim()[1]*0.9, "Su vehículo", 
                rotation=90, verticalalignment='top')
    
    ax.set_title(f'Relación entre {feature} y Precio')
    ax.set_xlabel(feature)
    ax.set_ylabel('Precio ($)')
    
    return fig

# Función para comparar precios de vehículos similares
def plot_similar_vehicles_comparison(similar_vehicles, predicted_price, user_brand, user_model):
    # Crear un DataFrame para la visualización
    df_plot = similar_vehicles[['brand', 'model', 'model_year', 'milage', 'price']].copy()
    
    # Añadir fila para el vehículo del usuario
    user_row = pd.DataFrame([{
        'brand': user_brand,
        'model': user_model,
        'model_year': 0,  # No se mostrará
        'milage': 0,      # No se mostrará
        'price': predicted_price
    }])
    
    df_plot = pd.concat([user_row, df_plot], ignore_index=True)
    
    # Crear etiquetas para la visualización
    df_plot['label'] = df_plot.apply(
        lambda x: f"{x['brand']} {x['model']} ({x['model_year']}, {int(x['milage']):,} km)" 
        if x['model'] != user_model or x['brand'] != user_brand else "Su vehículo (estimado)", 
        axis=1
    )
    
    # Ordenar por precio
    df_plot = df_plot.sort_values('price')
    
    # Gráfico de barras horizontales
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x == "Su vehículo (estimado)" else 'steelblue' for x in df_plot['label']]
    ax.barh(df_plot['label'], df_plot['price'], color=colors)
    ax.set_title('Comparación con vehículos similares')
    ax.set_xlabel('Precio ($)')
    ax.set_ylabel('')
    
    return fig

# Función para visualizar la distribución de precios por segmento
def plot_price_distribution(df, brand=None, model=None):
    if brand and model:
        data = df[(df['brand'] == brand) & (df['model'] == model)]
        title = f'Distribución de precios para {brand} {model}'
    elif brand:
        data = df[df['brand'] == brand]
        title = f'Distribución de precios para {brand}'
    else:
        data = df
        title = 'Distribución general de precios'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data['price'], bins=50, color='skyblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Precio ($)')
    ax.set_ylabel('Cantidad de vehículos')
    
    return fig

# Cargar datos y modelo
df = load_and_process_data()
model = load_model()

if df is None:
    st.error("No se pudieron cargar los datos. Verifica que el archivo used_cars.csv existe.")
    st.stop()

# Interfaz de usuario
st.title("🚗 Precio Justo de Autos Usados")
st.markdown("""
Esta herramienta te ayuda a determinar el precio justo para tu auto usado basado en sus características
y compara con vehículos similares del mercado para darte una perspectiva completa.
""")

# Crear tabs para diferentes secciones
tab1, tab2 = st.tabs(["Calcular Precio", "Explorar Mercado"])

with tab1:
    # Columnas para organizar la interfaz
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Características del Vehículo")
        
        # Obtener opciones de selección
        brands = sorted(df['brand'].unique())
        
        # Formulario para ingreso de datos
        brand = st.selectbox("Marca", brands)
        
        # Filtrar modelos por marca seleccionada
        models = sorted(df[df['brand'] == brand]['model'].unique())
        model = st.selectbox("Modelo", models)
        
        model_year = st.number_input("Año", min_value=1900, max_value=2025, value=2018)
        milage = st.number_input("Kilometraje", min_value=0, value=50000)
        
        # Tipo de combustible
        fuel_types = sorted(df['fuel_type'].unique())
        fuel_type = st.selectbox("Tipo de Combustible", fuel_types)
        
        # Tipo de motor
        if 'engine' in df.columns:
            engines = sorted(df[df['brand'] == brand]['engine'].unique())
            engine = st.selectbox("Motor", engines)
        else:
            engine = "standard"
            
        # Tipo de transmisión
        transmissions = sorted(df['transmission'].unique())
        transmission = st.selectbox("Transmisión", transmissions)
        
        # Historial de accidentes
        if 'accident' in df.columns:
            accidents = sorted(df['accident'].unique())
            accident = st.selectbox("Historial de Accidentes", accidents)
        else:
            accident = "No"
            
        # Título limpio
        if 'clean_title' in df.columns:
            clean_titles = sorted(df['clean_title'].unique())
            clean_title = st.selectbox("Título Limpio", clean_titles)
        else:
            clean_title = "Yes"
        
        # Botón para buscar
        if st.button("Calcular Precio Justo"):
            # Crear diccionario con los datos del usuario
            user_input = {
                'brand': brand,
                'model': model,
                'model_year': model_year,
                'milage': milage,
                'fuel_type': fuel_type,
                'engine': engine,
                'transmission': transmission,
                'accident': accident,
                'clean_title': clean_title
            }
            
            # Buscar vehículos similares (enfoque de app_minimal)
            similar_cars = find_similar_vehicles(df, user_input, n_neighbors=5)
            
            # Mostrar resultados
            with col2:
                st.subheader("Resultados")
                
                if len(similar_cars) > 0:
                    # Calcular estadísticas
                    avg_price = similar_cars['price'].mean()
                    median_price = similar_cars['price'].median()
                    min_price = similar_cars['price'].min()
                    max_price = similar_cars['price'].max()
                    
                    # Calcular precio estimado basado en vehículos similares
                    avg_milage = similar_cars['milage'].mean()
                    milage_factor = 1.0
                    
                    # Ajustar por kilometraje (deducción simple del 5% por cada 10,000 km por encima del promedio)
                    if milage > avg_milage:
                        extra_milage = milage - avg_milage
                        milage_factor = max(0.7, 1.0 - (extra_milage / 200000))
                    elif milage < avg_milage:
                        saved_milage = avg_milage - milage
                        milage_factor = min(1.3, 1.0 + (saved_milage / 200000))
                    
                    estimated_price = median_price * milage_factor
                    
                    # Intentar usar el modelo si está disponible
                    ml_prediction = None
                    if model is not None:
                        try:
                            # Crear un DataFrame para la predicción
                            user_df = pd.DataFrame([user_input])
                            user_df['milage'] = user_df['milage'].apply(clean_numeric)
                            user_df = extract_age_features(user_df)
                            user_df = extract_other_features(user_df)
                            user_df = preprocess_categorical(user_df)
                            
                            # Hacer la predicción
                            ml_prediction = model.predict(user_df)[0]
                            
                            # Mostrar ambas predicciones
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Precio por ML", f"${ml_prediction:,.2f}")
                            with col_b:
                                st.metric("Precio por Comparables", f"${estimated_price:,.2f}", 
                                      delta=f"{(milage_factor-1)*100:.1f}% ajuste por kilometraje")
                            
                            # Precio final (promedio de ambos)
                            final_price = (ml_prediction + estimated_price) / 2
                            st.metric("Precio Recomendado", f"${final_price:,.2f}")
                            
                            # Determinar si el precio es inusual
                            is_unusual, condition, threshold = is_unusual_price(final_price, similar_cars)
                            if is_unusual:
                                if condition == "alto":
                                    st.warning(f"⚠️ El precio estimado es inusualmente alto comparado con vehículos similares. La mayoría están por debajo de ${threshold:,.2f}.")
                                else:
                                    st.warning(f"⚠️ El precio estimado es inusualmente bajo comparado con vehículos similares. La mayoría están por encima de ${threshold:,.2f}.")
                        except Exception as e:
                            logger.error(f"Error al usar el modelo ML: {e}")
                            #st.warning(f"No se pudo usar el modelo de ML para predicción. Usando solo comparables.")
                            st.metric("Precio Estimado", f"${estimated_price:,.2f}", 
                                    delta=f"{(milage_factor-1)*100:.1f}% ajuste por kilometraje")
                    else:
                        # Si no hay modelo, usar solo el enfoque basado en comparables
                        st.metric("Precio Estimado", f"${estimated_price:,.2f}", 
                                delta=f"{(milage_factor-1)*100:.1f}% ajuste por kilometraje")
                    
                    # Mostrar estadísticas
                    st.markdown(f"### Estadísticas de precios para {brand} {model} (±3 años)")
                    
                    col1_stats, col2_stats, col3_stats = st.columns(3)
                    with col1_stats:
                        st.metric("Precio Promedio", f"${avg_price:,.2f}")
                    with col2_stats:
                        st.metric("Precio Mediano", f"${median_price:,.2f}")
                    with col3_stats:
                        st.metric("Rango de Precios", f"${min_price:,.2f} - ${max_price:,.2f}")
                    
                    # Visualizaciones
                    st.subheader("Comparación con vehículos similares")
                    
                    # Gráfico de comparación
                    final_price_to_use = ml_prediction if ml_prediction is not None else estimated_price
                    comparison_chart = plot_similar_vehicles_comparison(similar_cars, final_price_to_use, brand, model)
                    st.pyplot(comparison_chart)
                    
                    # Mostrar tabla con detalles de vehículos similares
                    st.markdown("### Detalles de vehículos similares")
                    st.dataframe(similar_cars[['brand', 'model', 'model_year', 'milage', 'price']].sort_values('model_year', ascending=False))
                    
                    # Graficar relación entre kilometraje y precio
                    st.subheader("Impacto del kilometraje en el precio")
                    mileage_chart = plot_feature_relationship(df[df['brand'] == brand], 'milage', milage)
                    if mileage_chart is not None:
                        st.pyplot(mileage_chart)
                    
                    # Graficar relación entre año del modelo y precio
                    st.subheader("Impacto de la edad del vehículo en el precio")
                    year_chart = plot_feature_relationship(df[df['brand'] == brand], 'model_year', model_year)
                    if year_chart is not None:
                        st.pyplot(year_chart)
                    
                else:
                    st.warning(f"No se encontraron vehículos similares a {brand} {model} en el rango de años {model_year-3} - {model_year+3}.")
                    st.markdown("""
                    Recomendaciones:
                    - Prueba con un modelo diferente
                    - Amplía el rango de años
                    - Elige una marca con más datos disponibles
                    """)

with tab2:
    st.header("Explorador del Mercado de Autos Usados")
    
    # Filtros para explorar el mercado
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_brand = st.selectbox("Filtrar por Marca", ["Todas"] + list(brands), key="explore_brand")
    
    with col2:
        min_year = st.number_input("Año Mínimo", min_value=int(df['model_year'].min()), max_value=int(df['model_year'].max()), value=int(df['model_year'].min()), key="min_year")
    
    with col3:
        max_year = st.number_input("Año Máximo", min_value=int(df['model_year'].min()), max_value=int(df['model_year'].max()), value=int(df['model_year'].max()), key="max_year")
    
    # Filtrar datos según selección
    filtered_df = df.copy()
    
    if selected_brand != "Todas":
        filtered_df = filtered_df[filtered_df['brand'] == selected_brand]
    
    filtered_df = filtered_df[(filtered_df['model_year'] >= min_year) & (filtered_df['model_year'] <= max_year)]
    
    # Mostrar estadísticas del mercado
    st.subheader("Estadísticas del Mercado")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precio Promedio", f"${filtered_df['price'].mean():,.2f}")
    
    with col2:
        st.metric("Precio Mediano", f"${filtered_df['price'].median():,.2f}")
    
    with col3:
        st.metric("Precio Mínimo", f"${filtered_df['price'].min():,.2f}")
    
    with col4:
        st.metric("Precio Máximo", f"${filtered_df['price'].max():,.2f}")
    
    # Visualizaciones de mercado
    st.subheader("Distribución de Precios")
    
    price_dist_chart = plot_price_distribution(filtered_df, selected_brand if selected_brand != "Todas" else None)
    st.pyplot(price_dist_chart)
    
    # Relación entre año y precio
    st.subheader("Tendencia de Precios por Año")
    
    # Usar matplotlib en lugar de seaborn para el boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Agrupar por año
    year_groups = filtered_df.groupby('model_year')
    
    # Crear datos para el boxplot
    years = []
    prices = []
    
    for year, group in year_groups:
        for price in group['price']:
            years.append(year)
            prices.append(price)
    
    # Crear un diccionario para el boxplot
    data = {'year': years, 'price': prices}
    
    # Convertir a DataFrame
    box_df = pd.DataFrame(data)
    
    # Crear un boxplot usando plt.boxplot
    years_unique = sorted(box_df['year'].unique())
    prices_by_year = [box_df[box_df['year'] == year]['price'] for year in years_unique]
    
    ax.boxplot(prices_by_year, labels=[str(int(year)) for year in years_unique])
    ax.set_title('Distribución de precios por año')
    ax.set_xlabel('Año')
    ax.set_ylabel('Precio ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Relación entre marca y precio
    if selected_brand == "Todas":
        st.subheader("Comparación de Precios por Marca")
        
        # Agrupar por marca y calcular estadísticas
        brand_stats = filtered_df.groupby('brand')['price'].agg(['mean', 'median', 'count']).reset_index()
        brand_stats = brand_stats.sort_values('mean', ascending=False)
        brand_stats = brand_stats[brand_stats['count'] >= 5]  # Mostrar solo marcas con al menos 5 vehículos
        
        # Tomar top 15 marcas
        top_brands = brand_stats.head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(top_brands['brand'], top_brands['mean'], color=plt.cm.viridis(np.linspace(0, 1, len(top_brands))))
        
        # Añadir etiquetas con la cuenta de vehículos
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(top_brands['mean']),
                   f"{int(top_brands['count'].iloc[i])} vehículos",
                   ha='center', va='bottom', rotation=90)
        
        ax.set_title('Precio promedio por marca (Top 15)')
        ax.set_xlabel('Marca')
        ax.set_ylabel('Precio Promedio ($)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)

# Barra lateral con estadísticas de la app
with st.sidebar:
    st.header("Estadísticas del Dataset")
    
    # Mostrar información sobre el dataset
    st.subheader("Datos del Mercado")
    st.write(f"Total de vehículos: {len(df):,}")
    st.write(f"Marcas disponibles: {df['brand'].nunique()}")
    st.write(f"Modelos disponibles: {df['model'].nunique()}")
    st.write(f"Rango de años: {int(df['model_year'].min())} - {int(df['model_year'].max())}")
    
    st.markdown("---")
    
    # Top 5 marcas por cantidad
    top_brands = df['brand'].value_counts().head(5)
    st.subheader("Marcas más populares")
    for brand, count in top_brands.items():
        st.write(f"{brand}: {count} vehículos")
    
    st.markdown("---")
    
    # Contar predicciones realizadas
    def app_stats():
        try:
            total = 0
            if os.path.exists("logs/app.log"):
                with open("logs/app.log", "r", encoding="utf-8") as f:
                    for line in f:
                        if "App iniciada" in line:
                            total += 1
            return total
        except Exception as e:
            logging.error(f"Error en estadísticas: {e}")
            return 0
    
    total_preds = app_stats()
    st.metric("Sesiones de la App", total_preds)
    
    st.markdown("---")
    
    st.caption("Desarrollado para el proyecto de Machine Learning - 2025")