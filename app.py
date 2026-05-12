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

# 1. IDENTIFICADORES Y RECURSOS
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
ID_MASK2FORMER_ZIP = "1CR2Io5CtJPI9DBJbupRSNp_HxsRttlnM"

@st.cache_resource
def cargar_modelos_panopticos():
    # EfficientPS (.pt)
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    checkpoint = torch.load("efficientps_final.pt", map_location="cpu")
    model_eff = checkpoint.get('model', checkpoint)
    if isinstance(model_eff, torch.nn.Module): model_eff.eval()

    # Mask2Former (Transformers)
    if not os.path.exists("mask2former_app"):
        gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', "m2k.zip", quiet=False)
        with zipfile.ZipFile("m2k.zip", 'r') as z: z.extractall(".") 
    
    model_m2k = AutoModelForUniversalSegmentation.from_pretrained("mask2former_app")
    processor_m2k = AutoImageProcessor.from_pretrained("mask2former_app")
    return model_eff, model_m2k, processor_m2k

model_eff, model_m2k, processor_m2k = cargar_modelos_panopticos()

# --- FUNCIÓN PANÓPTICA: ASIGNA COLORES ÚNICOS POR OBJETO ---
def aplicar_colores_panopticos(image, segmentation_mask, id_to_label=None):
    # Creamos una imagen vacía para el color
    h, w = segmentation_mask.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Cada ID único en la máscara representa un objeto diferente (instancia)
    unique_ids = np.unique(segmentation_mask)
    
    for obj_id in unique_ids:
        if obj_id == 0: continue # Fondo
        # Generar color aleatorio pero consistente por ID
        np.random.seed(int(obj_id) % 255)
        color = np.random.randint(0, 255, size=3).tolist()
        color_map[segmentation_mask == obj_id] = color
        
    # Mezcla con la imagen original para ver la transparencia (estilo panóptico)
    img_array = np.array(image.convert("RGB"))
    img_array = cv2.resize(img_array, (w, h))
    return cv2.addWeighted(img_array, 0.6, color_map, 0.4, 0)

# --- INFERENCIA EFFICIENTPS ---
def inferir_eff_panoptico(image, model):
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor(), 
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict): output = output.get('out', list(output.values())[0])
        # Filtro de suavizado antes de obtener instancias para reducir el ruido
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        mask = cv2.medianBlur(mask.astype(np.uint8), 5) 
        
    return aplicar_colores_panopticos(image, mask)

# --- INFERENCIA MASK2FORMER ---
def inferir_m2k_panoptico(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # USAR POST_PROCESS_PANOPTIC_SEGMENTATION para instancias reales
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    panoptic_mask = result["segmentation"].cpu().numpy()
    
    return aplicar_colores_panopticos(image, panoptic_mask)

# --- UI ---
st.title("🛰️ Deep Computer: Segmentación Panóptica (Instancias)")
uploaded_file = st.sidebar.file_uploader("Sube una imagen aérea", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    c1, c2 = st.columns(2)
    
    with c1:
        st.header("EfficientPS (CNN)")
        st.image(inferir_eff_panoptico(img, model_eff), use_container_width=True)
        st.write("Diferencia instancias mediante arquitectura convolucional.")

    with col2:
        st.header("Mask2Former (Transformer)")
        st.image(inferir_m2k_panoptico(img, model_m2k, processor_m2k), use_container_width=True)
        st.write("Segmentación panóptica mediante kernels de atención.")
