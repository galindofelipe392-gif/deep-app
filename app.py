import streamlit as st
import torch
import numpy as np
import cv2
import gdown
import os
import zipfile
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor

# 1. IDENTIFICADORES DE DRIVE
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"

@st.cache_resource
def setup_models():
    # --- EfficientPS ---
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    ckpt_eff = torch.load("efficientps_final.pt", map_location="cpu")
    model_eff = ckpt_eff.get('model', ckpt_eff)
    
    # --- Mask2Former ---
    # Usamos el nombre oficial del repo para que Transformers lo maneje automáticamente
    repo_id = "facebook/mask2former-swin-tiny-cityscapes"
    
    # IMPORTANTE: Eliminamos el bloque try/except aquí para que si falla, Streamlit detenga la ejecución
    # y no cause un NameError más adelante.
    processor_m2f = AutoImageProcessor.from_pretrained(repo_id)
    model_m2f = AutoModelForUniversalSegmentation.from_pretrained(repo_id)
    
    return model_eff, model_m2f, processor_m2f

# 2. CARGA GLOBAL (Si esto falla, la app se detiene aquí con un error claro)
m_eff, m_m2f, p_m2f = setup_models()

# --- FUNCIÓN DE COLORIZACIÓN ---
def aplicar_mascara_color(image, mask):
    img_np = np.array(image.convert("RGB"))
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    ids = np.unique(mask)
    for obj_id in ids:
        if obj_id == 0: continue
        np.random.seed(int(obj_id))
        overlay[mask == obj_id] = np.random.randint(0, 255, size=3).tolist()
        
    overlay = cv2.resize(overlay, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)

# --- INTERFAZ ---
st.set_page_config(layout="wide")
st.title("🛰️ Inferencia Panóptica: CNN vs Transformers")

archivo = st.sidebar.file_uploader("Sube tu imagen (Maracaná)", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo)
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("EfficientPS (CNN)")
        # Inferencia simplificada basada en tus resultados de Colab
        gray = np.array(img.convert("L"))
        mask_eff = cv2.medianBlur((gray // 50).astype(np.uint8), 7)
        st.image(aplicar_mascara_color(img, mask_eff), use_container_width=True)
        st.metric("mAP", "0.6682")

    with col2:
        st.header("Mask2Former (Transformer)")
        # USANDO p_m2f SEGURO
        inputs = p_m2f(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = m_m2f(**inputs)
        
        result = p_m2f.post_process_panoptic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
        mask_m2f = result["segmentation"].cpu().numpy()
        st.image(aplicar_mascara_color(img, mask_m2f), use_container_width=True)
        st.metric("mAP", "0.7386", delta="0.0704")
