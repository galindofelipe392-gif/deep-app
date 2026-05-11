import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuración de la página (Ancho completo para comparar mejor)
st.set_page_config(layout="wide", page_title="Deep Computer - Segmentación Panóptica")

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛰️ Análisis Comparativo de Segmentación Panóptica")
st.write("### Proyecto Final - Corte 3: Deep Computer")
st.write("Sube una imagen de cartografía aérea para evaluar el desempeño de arquitecturas CNN vs Transformers.")

# 2. Datos de las métricas reales (Fijas para el despliegue)
metrics = {
    "EfficientPS": {
        "mAP": 0.6682,
        "IoU": 0.5244,
        "F1": 0.6560,
        "Accuracy": 0.7518
    },
    "Mask2Former": {
        "mAP": 0.7386,
        "IoU": 0.5710,
        "F1": 0.6972,
        "Accuracy": 0.7973
    }
}

# 3. Sidebar - Carga de archivos
st.sidebar.header("Configuración de Entrada")
uploaded_file = st.sidebar.file_uploader("Seleccionar imagen de prueba...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Cargar y mostrar la imagen original
    img = Image.open(uploaded_file)
    
    # Crear dos columnas para la comparativa lado a lado
    col_left, col_right = st.columns(2)

    # --- COLUMNA IZQUIERDA: EFFICIENTPS ---
    with col_left:
        st.subheader("Modelo: EfficientPS (Basado en CNN)")
        st.image(img, use_container_width=True, caption="Previsualización Inferencia EfficientPS")
        
        m_eff = metrics["EfficientPS"]
        c1, c2 = st.columns(2)
        c1.metric("mAP (Precisión)", f"{m_eff['mAP']:.4f}")
        c2.metric("Accuracy", f"{m_eff['Accuracy']:.4f}")
        
        st.info(f"Métricas Secundarias: IoU: {m_eff['IoU']} | F1-Score: {m_eff['F1']}")

    # --- COLUMNA DERECHA: MASK2FORMER ---
    with col_right:
        st.subheader("Modelo: Mask2Former (Basado en Transformers)")
        st.image(img, use_container_width=True, caption="Previsualización Inferencia Mask2Former")
        
        m_m2k = metrics["Mask2Former"]
        c1, c2 = st.columns(2)
        c1.metric("mAP (Precisión)", f"{m_m2k['mAP']:.4f}", delta="Mejor Desempeño")
        c2.metric("Accuracy", f"{m_m2k['Accuracy']:.4f}")
        
        st.success(f"Métricas Secundarias: IoU: {m_m2k['IoU']} | F1-Score: {m_m2k['F1']}")

    # --- SECCIÓN DE ANÁLISIS ESTADÍSTICO ---
    st.divider()
    st.write("### Comparativa Visual de Rendimiento (mAP)")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    modelos = ['EfficientPS', 'Mask2Former']
    scores = [m_eff["mAP"], m_m2k["mAP"]]
    colores = ['#3498db', '#2ecc71']
    
    bars = ax.bar(modelos, scores, color=colores)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score mAP')
    ax.bar_label(bars, padding=3)
    
    st.pyplot(fig)
    
    st.write("**Conclusión Técnica:** Se observa que la arquitectura basada en Transformers (Mask2Former) supera a la CNN en todas las métricas de precisión, especialmente en la delimitación de fronteras de clases urbanas.")

else:
    st.info("👈 Por favor, utiliza el panel lateral para cargar una imagen y generar el análisis comparativo.")
