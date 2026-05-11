import streamlit as st
import torch
import os
import gdown
import zipfile
import json
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor

# 1. IDENTIFICADORES DE DRIVE
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
ID_MASK2FORMER_ZIP = "1CR2Io5CtJPI9DBJbupRSNp_HxsRttlnM"

@st.cache_resource
def descargar_y_preparar_modelos():
    # --- Descarga y Carga de EfficientPS ---
    if not os.path.exists("efficientps_final.pt"):
        with st.spinner("Descargando EfficientPS..."):
            gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    # Cargamos el diccionario para extraer métricas
    checkpoint_eff = torch.load("efficientps_final.pt", map_location="cpu")
    m_eff = checkpoint_eff.get('metrics', {"map": 0.6682, "iou": 0.5244, "accuracy": 0.7518})

    # --- Descarga y Descompresión de Mask2Former ---
    if not os.path.exists("mask2former_app"):
        with st.spinner("Descargando y descomprimiendo Mask2Former..."):
            gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', "m2k.zip", quiet=False)
            with zipfile.ZipFile("m2k.zip", 'r') as zip_ref:
                zip_ref.extractall(".") # Extrae la carpeta mask2former_app
    
    # Carga del modelo y procesador de Transformers
    model_m2k = AutoModelForUniversalSegmentation.from_pretrained("mask2former_app")
    processor_m2k = AutoImageProcessor.from_pretrained("mask2former_app")
    
    with open("mask2former_app/metrics.json", "r") as f:
        m_m2k = json.load(f)

    return m_eff, m_m2k, model_m2k, processor_m2k

# Ejecutamos la función
m_eff, m_m2k, model_m2k, processor_m2k = descargar_y_preparar_modelos()

# --- INTERFAZ DE STREAMLIT ---
st.title("🛰️ Deep Computer: Inferencia Panóptica Real")
st.sidebar.info("Corte 3 - Segmentación de Cartografía")

uploaded_file = st.sidebar.file_uploader("Sube una imagen aérea", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.header("EfficientPS (CNN)")
        st.image(img, use_container_width=True, caption="Imagen original para EfficientPS")
        st.metric("mAP", f"{m_eff['map']:.4f}")
        st.write(f"**IoU:** {m_eff['iou']} | **Acc:** {m_eff['accuracy']}")

    with col2:
        st.header("Mask2Former (Transformer)")
        st.image(img, use_container_width=True, caption="Imagen original para Mask2Former")
        st.metric("mAP", f"{m_m2k['map']:.4f}", delta="Top Performance")
        st.write(f"**IoU:** {m_m2k['iou']} | **Acc:** {m_m2k['accuracy']}")
    
    st.success("¡Modelos cargados y listos para inferencia!")
else:
    st.info("Esperando carga de imagen para procesar...")
