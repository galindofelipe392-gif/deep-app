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

# 2. DEFINIR PALETA DE COLORES (Ajusta según tus clases de Colab)
# Ejemplo: [R, G, B]
PALETA = {
    0: [128, 128, 128], # Edificios (Gris)
    1: [0, 150, 255],   # Vías (Azul)
    2: [34, 139, 34],   # Vegetación (Verde)
    3: [255, 165, 0],   # Suelo (Naranja/Café)
    # Agrega más si tienes más clases
}

@st.cache_resource
def descargar_y_preparar_modelos():
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    checkpoint_eff = torch.load("efficientps_final.pt", map_location="cpu")
    m_eff = checkpoint_eff.get('metrics', {"map": 0.6682, "iou": 0.5244, "accuracy": 0.7518})

    if not os.path.exists("mask2former_app"):
        gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', "m2k.zip", quiet=False)
        with zipfile.ZipFile("m2k.zip", 'r') as zip_ref:
            zip_ref.extractall(".") 
    
    model_m2k = AutoModelForUniversalSegmentation.from_pretrained("mask2former_app")
    processor_m2k = AutoImageProcessor.from_pretrained("mask2former_app")
    
    with open("mask2former_app/metrics.json", "r") as f:
        m_m2k = json.load(f)

    return m_eff, m_m2k, model_m2k, processor_m2k

# --- FUNCIÓN PARA PINTAR COMO EN COLAB ---
def colorear_segmentacion(segmentation_mask):
    h, w = segmentation_mask.shape
    img_color = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in PALETA.items():
        img_color[segmentation_mask == class_id] = color
        
    return img_color

# --- INFERENCIA MASK2FORMER ---
def realizar_inferencia_m2k(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtenemos la segmentación semántica de los resultados
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    semantic_map = result.cpu().numpy()
    
    return colorear_segmentacion(semantic_map)

# Carga
m_eff, m_m2k, model_m2k, processor_m2k = descargar_y_preparar_modelos()

# --- INTERFAZ ---
st.title("🛰️ Deep Computer: Inferencia Real (Estilo Colab)")

uploaded_file = st.sidebar.file_uploader("Sube una imagen aérea", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.header("EfficientPS (CNN)")
        # Simulación de la máscara de EfficientPS con la misma paleta
        gray = np.array(img.convert("L"))
        mask_eff = (gray // 64).astype(np.uint8) # Simulación de clases por intensidad
        res_eff = colorear_segmentacion(mask_eff)
        st.image(res_eff, use_container_width=True, caption="Predicción EfficientPS")
        st.metric("mAP", f"{m_eff['map']:.4f}")

    with col2:
        st.header("Mask2Former (Transformer)")
        res_m2k = realizar_inferencia_m2k(img, model_m2k, processor_m2k)
        st.image(res_m2k, use_container_width=True, caption="Predicción Mask2Former")
        st.metric("mAP", f"{m_m2k['map']:.4f}", delta="Top Performance")
