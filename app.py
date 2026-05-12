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

# 1. RECURSOS
ID_EFFICIENTPS = "1u4Of7RwrI-EAyszZqSQY5wv_cbZ1uJsN"
ID_MASK2FORMER_ZIP = "1CR2Io5CtJPI9DBJbupRSNp_HxsRttlnM"

@st.cache_resource
def cargar_modelos_panopticos():
    # --- EfficientPS ---
    if not os.path.exists("efficientps_final.pt"):
        gdown.download(f'https://drive.google.com/uc?id={ID_EFFICIENTPS}', "efficientps_final.pt", quiet=False)
    
    checkpoint = torch.load("efficientps_final.pt", map_location="cpu")
    
    # CORRECCIÓN CRÍTICA: Extraer el modelo si el .pt es un diccionario
    if isinstance(checkpoint, dict):
        # Intentamos obtener la llave donde suele guardarse el modelo
        model_eff = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        model_eff = checkpoint

    # Si model_eff sigue siendo un state_dict (solo pesos), necesitamos la clase original.
    # Por ahora, asumiremos que el .pt incluye la estructura (objeto completo).
    if hasattr(model_eff, 'eval'):
        model_eff.eval()
    
    # --- Mask2Former ---
    if not os.path.exists("mask2former_app"):
        gdown.download(f'https://drive.google.com/uc?id={ID_MASK2FORMER_ZIP}', "m2k.zip", quiet=False)
        with zipfile.ZipFile("m2k.zip", 'r') as z: z.extractall(".") 
    
    model_m2k = AutoModelForUniversalSegmentation.from_pretrained("mask2former_app")
    processor_m2k = AutoImageProcessor.from_pretrained("mask2former_app")
    
    return model_eff, model_m2k, processor_m2k

# --- INICIALIZACIÓN ---
model_eff, model_m2k, processor_m2k = cargar_modelos_panopticos()

def aplicar_colores_panopticos(image, mask):
    h, w = mask.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(mask)
    
    for obj_id in unique_ids:
        if obj_id == 0: continue 
        np.random.seed(int(obj_id))
        color = np.random.randint(0, 255, size=3).tolist()
        color_map[mask == obj_id] = color
        
    img_rgb = np.array(image.convert("RGB"))
    img_rgb = cv2.resize(img_rgb, (w, h))
    return cv2.addWeighted(img_rgb, 0.5, color_map, 0.5, 0)

# --- INFERENCIA EFFICIENTPS ---
def inferir_eff_panoptico(image, model):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # Verificamos si el modelo es llamable (callable)
        if callable(model):
            output = model(input_tensor)
            if isinstance(output, dict):
                output = output.get('out', list(output.values())[0])
            mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        else:
            # Si el modelo no cargó correctamente la estructura, simulamos con bordes
            st.error("El archivo .pt no contiene la estructura del modelo, solo los pesos.")
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            mask = (gray // 64).astype(np.uint8)

    mask_resized = cv2.resize(mask, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
    return aplicar_colores_panopticos(image, mask_resized)

# --- INFERENCIA MASK2FORMER ---
def inferir_m2k_panoptico(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    panoptic_mask = result["segmentation"].cpu().numpy()
    return aplicar_colores_panopticos(image, panoptic_mask)

# --- INTERFAZ ---
st.title("🛰️ Deep Computer: Panóptica Real")
up = st.sidebar.file_uploader("Cargar Imagen Aérea", type=["jpg", "png"])

if up:
    img = Image.open(up)
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("EfficientPS (CNN)")
        # Llamada segura
        res_eff = inferir_eff_panoptico(img, model_eff)
        st.image(res_eff, use_container_width=True)

    with col2:
        st.header("Mask2Former (Transformer)")
        res_m2k = inferir_m2k_panoptico(img, model_m2k, processor_m2k)
        st.image(res_m2k, use_container_width=True)
