import streamlit as st
import os
import sys

# Esta es una redirección para Streamlit Cloud
# Ejecutará app_unificado.py automáticamente

# Imprimir información de sistema para depuración
st.sidebar.write("Información del sistema:")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {os.listdir('.')}")

# Ejecutar la aplicación principal
import app_unificado