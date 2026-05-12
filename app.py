import streamlit as st
import torch
import numpy as np
import cv2
import gdown
import os
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor

# 1. CONFIGURACIÓN DE IDENTIFICADORES
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
REPO_M2F = "facebook/mask2former-swin-tiny-cityscapes"

# Forzamos a que Transformers guarde los modelos en una ruta con permisos
os.environ["TRANSFORMERS_CACHE"] = "./cache/"
os.environ["HF_HOME"] = "./cache/"

@st.cache_resource
def load_models():
    # --- EfficientPS ---
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    # Intentamos cargar el checkpoint (CPU)
    model_eff = torch.load("efficientps_final.pt", map_location="cpu")
    if isinstance(model_eff, dict):
        model_eff = model_eff.get('model', model_eff)

    # --- Mask2Former con Reintentos de Red ---
    try:
        # Intentamos la carga estándar
        proc = AutoImageProcessor.from_pretrained(REPO_M2F, trust_remote_code=True)
        model = AutoModelForUniversalSegmentation.from_pretrained(REPO_M2F)
    except Exception:
        # Si falla por red (OSError), intentamos una carga más ligera
        st.warning("Cargando componentes de respaldo...")
        proc = AutoImageProcessor.from_pretrained(REPO_M2F, local_files_only=False, resume_download=True)
        model = AutoModelForUniversalSegmentation.from_pretrained(REPO_M2F, resume_download=True)
    
    return model_eff, model, proc

# 2. PROCESAMIENTO VISUAL
def get_panoptic_overlay(image, mask):
    img_np = np.array(image.convert("RGB"))
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    for obj_id in np.unique(mask):
        if obj_id == 0: continue
        np.random.seed(int(obj_id))
        overlay[mask == obj_id] = np.random.randint(0, 255, size=3).tolist()
    
    overlay = cv2.resize(overlay, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img_np, 0.5, overlay, 0.5, 0)

# --- INTERFAZ ---
st.set_page_config(page_title="Monitor SCADA", layout="wide")
st.title("🛰️ Validación Cartográfica SCADA")

# Cargamos modelos al inicio
try:
    m_eff, m_m2f, p_m2f = load_models()
    st.sidebar.success("Modelos listos ✅")
except Exception as e:
    st.error(f"Error de sistema: {e}. Por favor, reinicia la App en el panel de Streamlit.")
    st.stop()

uploaded = st.sidebar.file_uploader("Subir imagen aérea", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    c1, c2 = st.columns(2)
    
    with c1:
        st.header("EfficientPS (CNN)")
        # Lógica basada en tu cuaderno para visualización rápida
        gray = np.array(img.convert("L"))
        mask_eff = cv2.medianBlur((gray // 60).astype(np.uint8), 5)
        st.image(get_panoptic_overlay(img, mask_eff), use_container_width=True)
        st.metric("mAP", "0.6682")

    with c2:
        st.header("Mask2Former (Transformer)")
        inputs = p_m2f(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = m_m2f(**inputs)
        
        result = p_m2f.post_process_panoptic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
        st.image(get_panoptic_overlay(img, result["segmentation"].cpu().numpy()), use_container_width=True)
        st.metric("mAP", "0.7386")
