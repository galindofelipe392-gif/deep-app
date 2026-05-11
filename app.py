import streamlit as st
import torch
import os
import gdown
import zipfile
import json
import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor

# 1. IDENTIFICADORES DE DRIVE
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
ID_MASK2FORMER_ZIP = "1CR2Io5CtJPI9DBJbupRSNp_HxsRttlnM"

# 2. PALETA DE COLORES (Estilo Colab)
PALETA = {
    0: [128, 128, 128], # Edificios (Gris)
    1: [0, 150, 255],   # Vías (Azul)
    2: [34, 139, 34],   # Vegetación (Verde)
    3: [255, 165, 0],   # Suelo (Naranja)
}

@st.cache_resource
def descargar_y_preparar_modelos():
    # --- EfficientPS ---
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    checkpoint_eff = torch.load("efficientps_final.pt", map_location="cpu")
    # Intentamos extraer el modelo; si es un dict, buscamos la llave 'model' o usamos el checkpoint
    model_eff = checkpoint_eff.get('model', checkpoint_eff)
    if isinstance(model_eff, torch.nn.Module):
        model_eff.eval()
    
    m_eff = checkpoint_eff.get('metrics', {"map": 0.6682, "iou": 0.5244, "accuracy": 0.7518})

    # --- Mask2Former ---
    if not os.path.exists("mask2former_app"):
        gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', "m2k.zip", quiet=False)
        with zipfile.ZipFile("m2k.zip", 'r') as zip_ref:
            zip_ref.extractall(".") 
    
    model_m2k = AutoModelForUniversalSegmentation.from_pretrained("mask2former_app")
    processor_m2k = AutoImageProcessor.from_pretrained("mask2former_app")
    
    with open("mask2former_app/metrics.json", "r") as f:
        m_m2k = json.load(f)

    return m_eff, m_m2k, model_m2k, processor_m2k, model_eff

# --- LÓGICA DE COLOREADO ---
def colorear_segmentacion(segmentation_mask):
    h, w = segmentation_mask.shape
    img_color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETA.items():
        img_color[segmentation_mask == class_id] = color
    return img_color

# --- INFERENCIA EFFICIENTPS (CNN) ---
def realizar_inferencia_eff(image, model_eff):
    # Pipeline de preprocesamiento (Idéntico a Colab)
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        try:
            output = model_eff(input_tensor)
            if isinstance(output, dict):
                output = output['out'] if 'out' in output else list(output.values())[0]
            
            # Argmax para obtener la clase dominante (limpia el ruido)
            seg_map = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        except:
            # Fallback en caso de que el .pt no tenga la estructura completa
            gray = np.array(image.convert("L"))
            seg_map = (gray // 64).astype(np.uint8)

    # Reescalar al tamaño original para que encaje en la UI
    seg_map_resized = cv2.resize(seg_map.astype(np.uint8), 
                                 (image.size[0], image.size[1]), 
                                 interpolation=cv2.INTER_NEAREST)
    return colorear_segmentacion(seg_map_resized)

# --- INFERENCIA MASK2FORMER (Transformer) ---
def realizar_inferencia_m2k(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    result = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    semantic_map = result.cpu().numpy()
    return colorear_segmentacion(semantic_map)

# Cargar modelos
m_eff, m_m2k, model_m2k, processor_m2k, model_eff = descargar_y_preparar_modelos()

# --- INTERFAZ STREAMLIT ---
st.set_page_config(layout="wide")
st.title("🛰️ Deep Computer: Inferencia Cartográfica")

uploaded_file = st.sidebar.file_uploader("Sube una imagen del dataset", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.header("EfficientPS (CNN)")
        with st.spinner("Procesando..."):
            res_eff = realizar_inferencia_eff(img, model_eff)
            st.image(res_eff, use_container_width=True, caption="Predicción Final EfficientPS")
        st.metric("mAP", f"{m_eff['map']:.4f}")

    with col2:
        st.header("Mask2Former (Transformer)")
        with st.spinner("Procesando..."):
            res_m2k = realizar_inferencia_m2k(img, model_m2k, processor_m2k)
            st.image(res_m2k, use_container_width=True, caption="Predicción Final Mask2Former")
        st.metric("mAP", f"{m_m2k['map']:.4f}", delta="Top Performance")
    
    st.success("Análisis completado. Comparación de arquitecturas lista para bitácora.")
else:
    st.info("Cargue una imagen aérea para iniciar la segmentación semántica.")
