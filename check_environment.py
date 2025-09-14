import streamlit as st
import sys
import os

st.set_page_config(page_title="Verificador de Entorno", page_icon="🔍")

st.title("Verificador de Entorno Python")

# Información básica del sistema
st.write("## Información del Sistema")
st.write(f"Python version: {sys.version}")
st.write(f"Executable: {sys.executable}")
st.write(f"Current directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir('.')}")

# Verificación de instalación de paquetes
st.write("## Paquetes Instalados")

try:
    import pandas as pd
    st.success(f"✓ pandas {pd.__version__}")
except ImportError as e:
    st.error(f"✗ pandas: {e}")

try:
    import numpy as np
    st.success(f"✓ numpy {np.__version__}")
except ImportError as e:
    st.error(f"✗ numpy: {e}")

# Verificación específica de matplotlib
st.write("## Diagnóstico de Matplotlib")
try:
    import matplotlib
    st.success(f"✓ matplotlib {matplotlib.__version__}")
    st.write(f"Backend usado: {matplotlib.get_backend()}")
    
    # Intentar cambiar al backend Agg
    try:
        matplotlib.use('Agg')
        st.success("✓ Cambio a backend 'Agg' exitoso")
    except Exception as e:
        st.error(f"✗ Error al cambiar backend: {e}")
    
    # Intentar importar pyplot
    try:
        import matplotlib.pyplot as plt
        st.success("✓ matplotlib.pyplot importado correctamente")
        
        # Intentar crear una figura simple
        try:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot([1, 2, 3], [1, 4, 9])
            ax.set_title("Prueba Matplotlib")
            st.pyplot(fig)
            st.success("✓ Creación y visualización de gráfico exitosa")
        except Exception as e:
            st.error(f"✗ Error al crear/mostrar gráfico: {e}")
    except Exception as e:
        st.error(f"✗ Error al importar pyplot: {e}")
        
except ImportError as e:
    st.error(f"✗ matplotlib: {e}")

# Verificar otras dependencias
st.write("## Otras Dependencias")
for package in ['joblib', 'scikit-learn', 'seaborn', 'python-dateutil', 
                'cycler', 'kiwisolver', 'pyparsing', 'pillow', 
                'contourpy', 'fonttools', 'packaging']:
    try:
        module = __import__(package)
        if hasattr(module, '__version__'):
            st.success(f"✓ {package} {module.__version__}")
        else:
            st.success(f"✓ {package} (versión desconocida)")
    except ImportError as e:
        st.warning(f"✗ {package}: {e}")

st.write("## Entorno del Sistema")
for var in ['PYTHONPATH', 'MPLBACKEND', 'PATH']:
    if var in os.environ:
        st.write(f"**{var}**: {os.environ[var]}")
    else:
        st.write(f"**{var}**: No definido")