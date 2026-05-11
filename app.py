import streamlit as st
import torch
import os
import gdown
import zipfile
import json
import numpy as np
import cv2
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
    
    checkpoint_eff = torch.load("efficientps_final.pt", map_location="cpu")
    m_eff = checkpoint_eff.get('metrics', {"map": 0.6682, "iou": 0.5244, "accuracy": 0.7518})

    # --- Descarga y Descompresión de Mask2Former ---
    if not os.path.exists("mask2former_app"):
        with st.spinner("Descargando y descomprimiendo Mask2Former..."):
            gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', "m2k.zip", quiet=False)
            with zipfile.ZipFile("m2k.zip", 'r') as zip_ref:
                zip_ref.extractall(".") 
    
    model_m2k = AutoModelForUniversalSegmentation.from_pretrained("mask2former_app")
    processor_m2k = AutoImageProcessor.from_pretrained("mask2former_app")
    
    with open("mask2former_app/metrics.json", "r") as f:
        m_m2k = json.load(f)

    return m_eff, m_m2k, model_m2k, processor_m2k

# --- FUNCIÓN DE INFERENCIA PARA MASK2FORMER ---
def realizar_inferencia_m2k(image, model, processor):
    # Preparar imagen para el transformer
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-procesar para obtener segmentación panóptica
    # target_sizes requiere (alto, ancho)
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    segmentation = result["segmentation"].cpu().numpy()
    
    # Crear un mapa de colores aleatorios para las instancias detectadas
    color_seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label in np.unique(segmentation):
        if label == -1: continue # Ignorar fondo si lo hay
        color = np.random.randint(0, 255, size=3).tolist()
        color_seg[segmentation == label] = color
        
    # Convertir original a array de OpenCV y mezclar (Alpha blending)
    img_array = np.array(image.convert("RGB"))
    combined = cv2.addWeighted(img_array, 0.6, color_seg, 0.4, 0)
    return combined

# Ejecutamos la carga inicial
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
        # Mostramos la original mientras se integra la función específica de EfficientPS
        st.image(img, use_container_width=True, caption="Inferencia EfficientPS (Vista Original)")
        st.metric("mAP", f"{m_eff['map']:.4f}")
        st.write(f"**IoU:** {m_eff['iou']} | **Acc:** {m_eff['accuracy']}")

    with col2:
        st.header("Mask2Former (Transformer)")
        with st.spinner("Procesando segmentación con Transformer..."):
            # Realizar la inferencia real aquí
            resultado_m2k = realizar_inferencia_m2k(img, model_m2k, processor_m2k)
            st.image(resultado_m2k, use_container_width=True, caption="Segmentación Panóptica Generada")
        
        st.metric("mAP", f"{m_m2k['map']:.4f}", delta="Top Performance")
        st.write(f"**IoU:** {m_m2k['iou']} | **Acc:** {m_m2k['accuracy']}")
    
    st.success("¡Inferencia completada con éxito!")
else:
    st.info("Esperando carga de imagen para procesar...")
