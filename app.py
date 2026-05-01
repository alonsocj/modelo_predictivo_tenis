import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 160
MODEL_PATH = Path(__file__).parent / 'mi_modelo.keras'
CLASES_PATH = Path(__file__).parent / 'clases.json'

st.set_page_config(
    page_title='Clasificador de Marcas de Tenis',
    page_icon='[IMG]',
    layout='centered',
)


@st.cache_resource
def cargar_modelo():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASES_PATH, 'r', encoding='utf-8') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


modelo, idx_to_class = cargar_modelo()

st.title('Clasificador de Marcas de Tenis')
st.write('Clasificador de marcas de zapatos deportivos (Nike, Adidas, Puma, Converse) entrenado con Transfer Learning sobre ResNet50.')
st.caption('Autor: Luis Alonso Cornejo Jimenez - Universidad Don Bosco - Modulo 4')

st.divider()

archivo = st.file_uploader(
    'Sube una imagen',
    type=['jpg', 'jpeg', 'png'],
    help='Formatos soportados: JPG, JPEG, PNG',
)

if archivo is not None:
    imagen = Image.open(archivo).convert('RGB')

    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(imagen, caption='Imagen subida', use_container_width=True)

    img_resized = imagen.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img_resized).astype('float32')
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = modelo.predict(arr, verbose=0)[0]
    orden = pred.argsort()[::-1]

    with col_pred:
        st.subheader('Prediccion')
        clase_top = idx_to_class[orden[0]].replace('_', ' ').title()
        st.metric('Clase predicha', clase_top, f'{pred[orden[0]]:.1%} confianza')

    st.divider()
    st.subheader('Probabilidades por clase')
    for i in orden:
        nombre = idx_to_class[i].replace('_', ' ').title()
        prob = float(pred[i])
        st.write(f'**{nombre}** - {prob:.1%}')
        st.progress(prob)
else:
    st.info('Esperando una imagen para clasificar.')
