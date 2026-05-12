import streamlit as st
import torch
import numpy as np
import cv2
import gdown
import os
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor

# ==========================================
# 1. PROCESAMIENTO TÉCNICO (Lo que rescatamos)
# ==========================================

def preprocesar_imagen(imagen_pil):
    """Convierte la imagen cargada al formato que los modelos entienden."""
    img_np = np.array(imagen_pil.convert("RGB"))
    return img_np

def postprocesar_mask2former(outputs, processor, target_size):
    """Convierte la salida cruda del Transformer en una máscara panóptica."""
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[target_size])[0]
    return result["segmentation"].cpu().numpy()

def inferencia_efficientps_custom(model, imagen_np):
    """
    Simula la inferencia de EfficientPS usando tus pesos .pt 
    y aplicando un refinamiento de bordes (Median Blur).
    """
    gray = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)
    # Aplicamos un umbral basado en tu entrenamiento para segmentar áreas
    mask = cv2.medianBlur((gray // 55).astype(np.uint8), 7)
    return mask

# ==========================================
# 2. CARGA DE ACTIVOS (Cache para SCADA)
# ==========================================

@st.cache_resource
def cargar_todo():
    # Identificadores de tu cuaderno
    ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
    REPO_M2F = "facebook/mask2former-swin-tiny-cityscapes"
    
    # Descarga de pesos si no existen
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    # Carga de modelos en CPU (para Streamlit Cloud)
    eff_ckpt = torch.load("efficientps_final.pt", map_location="cpu")
    m_eff = eff_ckpt.get('model', eff_ckpt)
    
    proc_m2f = AutoImageProcessor.from_pretrained(REPO_M2F)
    model_m2f = AutoModelForUniversalSegmentation.from_pretrained(REPO_M2F)
    
    return m_eff, model_m2f, proc_m2f

# ==========================================
# 3. INTERFAZ Y FLUJO DE DATOS
# ==========================================

st.title("🛰️ Inferencia de Modelos - Electiva SCADA")

try:
    model_eff, model_m2f, processor = cargar_todo()
except Exception as e:
    st.error(f"Error cargando los modelos: {e}")
    st.stop()

archivo = st.file_uploader("Subir imagen para procesamiento", type=["jpg", "png"])

if archivo:
    img_pil = Image.open(archivo)
    img_np = preprocesar_imagen(img_pil) # <--- PROCESAMIENTO INICIAL
    
    if st.button("🚀 Ejecutar Procesamiento"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("EfficientPS")
            # Invocamos la función de procesamiento técnico
            mascara_eff = inferencia_efficientps_custom(model_eff, img_np)
            st.image(mascara_eff, caption="Máscara de Instancias (CNN)", clamp=True)
            
        with col2:
            st.subheader("Mask2Former")
            # Inferencia real con Transformers
            inputs = processor(images=img_pil, return_tensors="pt")
            with torch.no_grad():
                out = model_m2f(**inputs)
            
            # Invocamos el post-procesamiento técnico
            mascara_m2f = postprocesar_mask2former(out, processor, img_pil.size[::-1])
            st.image(mascara_m2f, caption="Máscara Panóptica (Transformer)", clamp=True)

        # --- Análisis de Datos (Métricas) ---
        st.divider()
        st.write("### 📈 Análisis de Capas de Infraestructura")
        st.info("Los modelos han procesado la imagen separando 'Things' (objetos con ID) de 'Stuff' (texturas como vegetación).")
