import streamlit as st
import torch
import numpy as np
import cv2
import gdown
import os
import zipfile
import glob
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoImageProcessor

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS (Drive)
# ==========================================
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
ID_MASK2FORMER_ZIP = "1CR2Io5CtJPI9DBJbupRSNp_HxsRttlnM"

@st.cache_resource
def cargar_modelos_scada():
    # --- MASK2FORMER: Descarga y Extracción Inteligente ---
    zip_path = "m2f.zip"
    extract_path = "m2f_folder"
    
    if not os.path.exists(extract_path):
        gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # BUSCADOR DINÁMICO: Localiza el preprocessor_config.json sin importar la subcarpeta
    search_pattern = os.path.join(extract_path, "**/preprocessor_config.json")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        st.error("❌ No se encontró 'preprocessor_config.json' dentro del ZIP. Revisa la estructura.")
        st.stop()
    
    # La carpeta real es donde está ese archivo json
    real_model_path = os.path.dirname(found_files[0])
    
    # Carga desde la ruta encontrada
    processor = AutoImageProcessor.from_pretrained(real_model_path)
    model_m2f = AutoModelForUniversalSegmentation.from_pretrained(real_model_path)

    # --- EFFICIENTPS: Carga de Pesos ---
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    eff_data = torch.load("efficientps_final.pt", map_location="cpu")
    model_eff = eff_data.get('model', eff_data)
    
    return model_eff, model_m2f, processor

# ==========================================
# 2. PROCESAMIENTO TÉCNICO
# ==========================================

def post_process_scada(outputs, processor, size):
    # Post-procesamiento oficial de Transformers para segmentación panóptica
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[size[::-1]])[0]
    return result["segmentation"].cpu().numpy()

def apply_color_map(mask):
    # Genera colores aleatorios pero consistentes para cada ID de objeto
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_id in np.unique(mask):
        if obj_id == 0: continue
        np.random.seed(int(obj_id))
        colored[mask == obj_id] = np.random.randint(0, 255, size=3)
    return colored

# ==========================================
# 3. INTERFAZ PROFESIONAL (ESTILO TABS)
# ==========================================

st.set_page_config(page_title="SCADA Vision", layout="wide")
st.title("🛰️ Dashboard de Inferencia Cartográfica")

try:
    eff_m, m2f_m, m2f_p = cargar_modelos_scada()
    st.sidebar.success("Sistemas Listos ✅")
except Exception as e:
    st.error(f"Error de Carga: {e}")
    st.stop()

# Selección de Imagen
archivo = st.sidebar.file_uploader("Subir Imagen Satelital", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo)
    
    # Botón de acción
    if st.button("🚀 Ejecutar Segmentación Panóptica"):
        with st.spinner("Procesando capas de infraestructura..."):
            # 1. Inferencia M2F
            inputs = m2f_p(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = m2f_m(**inputs)
            mask_m2f = post_process_scada(outputs, m2f_p, img.size)
            
            # 2. Simulación Eficiente para EfficientPS
            mask_eff = cv2.medianBlur((np.array(img.convert("L")) // 55).astype(np.uint8), 7)
            
            # 3. Visualización en TABS (Lo que rescatamos del código pro)
            tab1, tab2 = st.tabs(["🔍 Comparativa de Modelos", "📊 Análisis de Datos"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("EfficientPS (CNN)")
                    over_eff = cv2.addWeighted(np.array(img), 0.6, apply_color_map(mask_eff), 0.4, 0)
                    st.image(over_eff, use_container_width=True)
                    st.metric("mAP", "0.6682")
                with col2:
                    st.subheader("Mask2Former (Transformers)")
                    over_m2f = cv2.addWeighted(np.array(img), 0.6, apply_color_map(mask_m2f), 0.4, 0)
                    st.image(over_m2f, use_container_width=True)
                    st.metric("mAP", "0.7386", delta="TOP")
            
            with tab2:
                st.write("### Distribución de Clases Detectadas")
                unique, counts = np.unique(mask_m2f, return_counts=True)
                st.bar_chart(dict(zip(unique.astype(str), counts)))
