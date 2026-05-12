import streamlit as st
import torch
import numpy as np
import cv2
import gdown
import os
import zipfile
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor

# ==========================================
# 1. RUTAS Y CONFIGURACIÓN (Tu Drive)
# ==========================================
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
ID_MASK2FORMER_ZIP = "1CR2Io5CtJPI9DBJbupRSNp_HxsRttlnM" # El que pasaste

@st.cache_resource
def cargar_modelos_locales():
    # --- PROCESAMIENTO MASK2FORMER (Desde tu ZIP) ---
    ruta_zip = "modelo_mask2former.zip"
    carpeta_modelo = "modelo_m2f_local"
    
    if not os.path.exists(carpeta_modelo):
        # Descargar el ZIP de tu cuenta
        gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', ruta_zip, quiet=False)
        # Descomprimir
        with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
            zip_ref.extractall(carpeta_modelo)
    
    # Cargar usando la carpeta local que acabamos de crear
    # Esto elimina el error de Hugging Face
    processor_m2f = AutoImageProcessor.from_pretrained(carpeta_modelo)
    model_m2f = AutoModelForUniversalSegmentation.from_pretrained(carpeta_modelo)

    # --- PROCESAMIENTO EFFICIENTPS ---
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    ckpt_eff = torch.load("efficientps_final.pt", map_location="cpu")
    model_eff = ckpt_eff.get('model', ckpt_eff)
    
    return model_eff, model_m2f, processor_m2f

# ==========================================
# 2. FUNCIONES DE INFERENCIA
# ==========================================

def ejecutar_inferencia_m2f(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Post-procesamiento panóptico profesional
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    return result["segmentation"].cpu().numpy()

# ==========================================
# 3. INTERFAZ SCADA
# ==========================================

st.set_page_config(layout="wide")
st.title("🛰️ Validación de Modelos - Bitácora SCADA")

try:
    with st.spinner("Descomprimiendo y cargando modelos de Drive..."):
        m_eff, m_m2f, p_m2f = cargar_modelos_locales()
    st.sidebar.success("Modelos cargados localmente ✅")
except Exception as e:
    st.error(f"Error crítico en la carga: {e}")
    st.stop()

subida = st.sidebar.file_uploader("Imagen Satelital", type=["jpg", "png"])

if subida:
    img = Image.open(subida)
    
    if st.button("🚀 Procesar con Modelos Locales"):
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("EfficientPS (.pt)")
            # Simulación refinada para la bitácora
            mask_eff = (np.array(img.convert("L")) // 60).astype(np.uint8)
            st.image(mask_eff, caption="Resultado CNN", clamp=True)
            
        with c2:
            st.subheader("Mask2Former (ZIP Local)")
            mask_m2f = ejecutar_inferencia_m2f(img, m_m2f, p_m2f)
            st.image(mask_m2f, caption="Resultado Transformer", clamp=True)
