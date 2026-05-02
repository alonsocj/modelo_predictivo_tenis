# Clasificador de Marcas de Tenis

Clasificador de imagenes que reconoce cuatro marcas de zapatos deportivos (Nike, Adidas, Puma, Converse) usando Transfer Learning sobre ResNet50 y desplegado como aplicacion web publica con Streamlit Cloud.

> Proyecto Final - Modulo 4 (Sesion 4) | Diplomado Python Avanzado | Universidad Don Bosco

---

## Demo

App publica: **https://modelopredictivotenis.streamlit.app/**

Sube una imagen `.jpg`, `.jpeg` o `.png` de un zapato y la app devuelve la marca predicha junto con la probabilidad asignada a cada una de las cuatro clases.

---

## Por que este tema

Las marcas de tenis son un dominio cotidiano, visual y con identidad grafica fuerte (logos, suelas, perfiles), pero al mismo tiempo son dificiles de distinguir entre si cuando solo se ven detalles parciales o angulos atipicos. Era un tema:

- **Personal**: zapatillas que cualquiera ha visto en la calle, no un dataset academico.
- **No prohibido**: no entra en la lista vetada (CIFAR, MNIST, Cats vs Dogs, Simpsons).
- **Imagenes faciles de conseguir**: Google Images devuelve catalogos extensos por marca.
- **Reto real**: la similitud visual entre marcas obliga al modelo a aprender features finas, no atajos.

---

## Dataset

Imagenes recolectadas manualmente desde Google Images, organizadas por carpeta de clase en Google Drive y luego copiadas al area de trabajo de Colab.

| Clase    | Imagenes |
|----------|----------|
| nike     | 133      |
| adidas   | 162      |
| puma     | 155      |
| converse | 145      |
| **Total**    | **595**      |

- **Split**: 80% train / 20% validation (`validation_split=0.2`, `seed=42`).
- **Train**: 477 imagenes | **Val**: 118 imagenes.
- **Tamano de entrada**: 160x160x3.
- **Batch size**: 32.

### Augmentation (solo en train)

`rotation_range=20`, `width_shift_range=0.1`, `height_shift_range=0.1`, `shear_range=0.1`, `zoom_range=0.2`, `horizontal_flip=True`. El set de validacion se preprocesa solo con `preprocess_input` de ResNet50, sin augmentation.

---

## Arquitectura del modelo

Transfer Learning con la base de **ResNet50 (ImageNet) congelada** y una cabeza de clasificacion entrenable encima.

```text
Input(160, 160, 3)
   |
ResNet50(weights='imagenet', include_top=False, training=False)   <- congelada
   |
GlobalAveragePooling2D
   |
Dense(256, relu)
   |
BatchNormalization
   |
Dropout(0.5)
   |
Dense(4, softmax)
```

- **Optimizer**: Adam (`lr=1e-3`).
- **Loss**: `categorical_crossentropy`.
- **Callbacks**: `EarlyStopping(patience=3, restore_best_weights=True)` + `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)`.
- **Epochs**: 12 (con corte temprano por EarlyStopping).
- **Truco clave**: `base_model(inputs, training=False)` para que las capas `BatchNormalization` de ResNet50 no actualicen sus estadisticas con batches pequenos (mismo bug que vimos en clase con el notebook de Simpsons).

---

## Resultados

- **Mejor val_accuracy**: **61.86%**
- **Accuracy final sobre validation (recargado)**: 60.17%
- **Errores**: 47 / 118 (39.8%)

### Reporte por clase

| Clase    | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| nike     | 0.465     | 0.769  | 0.580 | 26      |
| adidas   | 0.722     | 0.406  | 0.520 | 32      |
| puma     | 0.615     | 0.516  | 0.561 | 31      |
| converse | 0.710     | 0.759  | 0.733 | 29      |
| **macro avg**    | 0.628     | 0.613  | 0.599 | 118     |
| **weighted avg** | 0.634     | 0.602  | 0.596 | 118     |

**Lectura rapida**:
- **Converse** es la clase mas facil (F1 = 0.733); su silueta de tela y suela plana es muy distintiva.
- **Nike** tiene recall alto pero precision baja (0.465): el modelo "se va por Nike" cuando duda y termina absorbiendo falsos positivos de otras marcas.
- **Adidas** es la mas castigada en recall (0.406): muchos Adidas terminan etiquetados como Nike o Puma.
- **Puma** queda en un punto medio, sin un patron claro de fuga.

---

## Analisis de errores

De las 118 imagenes de validacion, el modelo falla en 47 (39.8%). Los errores no son aleatorios, siguen patrones visibles:

1. **Sesgo hacia "Nike"**: cuando la imagen tiene fondo limpio y forma generica de zapato deportivo, el modelo tiende a contestar Nike. Esto se nota en el recall altisimo de Nike (0.769) contra su precision baja (0.465). En la practica Nike actua como "clase comodin" cuando la red no esta segura.
2. **Adidas se confunde con Nike y Puma**: zapatillas Adidas con vista lateral y suela negra tienden a clasificarse como Nike; modelos casuales con perfil bajo se mueven hacia Puma. Las tres rayas no siempre son visibles desde el angulo de la foto, y cuando no aparecen, el modelo pierde la pista mas distintiva de la marca.
3. **Pistas globales > pistas locales**: la red parece apoyarse en textura general, color dominante y fondo en lugar del logo o la firma de suela, que es justo donde vive la diferencia real entre marcas.
4. **Converse se separa bien**: su forma de bota baja de lona contra suela plana de goma blanca es lo bastante diferente del resto como para sostener un F1 alto.

---

## Reflexiones

> Las siguientes preguntas vienen de las celdas reflexivas del notebook (Pasos 3, 5 y 6). Las respondo abiertamente con lo observado en el proyecto.

### Sobre el dataset (Paso 3)

**Esta balanceado tu dataset?**
Razonablemente. La clase mas pequena (nike, 133) y la mas grande (adidas, 162) tienen un desbalance leve (~22%). No hay clases dominantes y cada una supera ampliamente el minimo exigido. No fue necesario aplicar pesos por clase ni oversampling.

**Tienes minimo 30 por clase?**
Si, el minimo del dataset es 133 (nike), muy por encima del piso de 30 imagenes que pide el proyecto.

### Sobre el entrenamiento (Paso 5)

**Tu modelo mostro overfitting?**
Si, claramente. El gap entre train y validation es grande: la accuracy de entrenamiento sube hasta ~88-89% mientras la de validacion se estanca en torno a 60%. El `val_loss` deja de bajar consistentemente despues de la epoca 8, lo que dispara `ReduceLROnPlateau` y eventualmente `EarlyStopping`.

**Que accuracy obtuviste en validation?**
**61.86%** como mejor valor (`val_accuracy`). Es honesto reconocer que esta por debajo del umbral de 70% que sugiere el notebook como objetivo.

**Si esta por debajo del 70%, que harias?**
Aplicaria los siguientes pasos en orden:

1. **Mas imagenes por clase** (objetivo: 250-300 cada una), priorizando variedad de angulos, iluminacion y fondos.
2. **Augmentation mas agresiva**: rangos mayores de zoom y rotacion, `brightness_range`, posiblemente `RandomErasing` o cutout.
3. **Fine-tuning de la cola de ResNet50**: descongelar las ultimas ~20 capas convolucionales y reentrenar con un `learning_rate` muy bajo (`1e-5`) durante unas pocas epocas adicionales.
4. **Limpieza del dataset**: revisar manualmente que no haya logos ambiguos, fotos con multiples marcas en escena, o imagenes mal etiquetadas que esten introduciendo ruido.

### Sobre los errores (Paso 6)

**Hay un patron en los errores?**
Si: el modelo se inclina por Nike cuando duda y le cuesta aislar las pistas finas (logo, suela). Las marcas con silueta similar (Adidas / Puma / Nike, todas perfil bajo deportivo) son las que mas se mezclan entre si.

**Que clase confunde mas?**
Adidas. Tiene el peor recall (0.406): casi 6 de cada 10 Adidas terminan en otra clase. El destino mas frecuente de esos errores son Nike y Puma.

**Por que crees que pasa?**
El logo de las tres rayas y el "trefoil" suelen vivir en lugares pequenos de la imagen (laterales, lengueta), y cuando la foto los recorta, los oscurece o los muestra en perspectiva, el modelo se queda sin la senal mas distintiva de la marca y termina apoyandose en pistas globales que comparte con Nike y Puma.

### Aprendizajes

- **Transfer Learning ahorra tiempo, no resuelve magia**: ResNet50 trae representaciones potentes, pero si el dataset es pequeno, ruidoso y desbalanceado por dentro de cada clase (variedad de angulos, fondos, modelos), el techo de validacion se siente igual.
- **`training=False` no es un detalle cosmetico**: olvidarlo en la base con BatchNormalization rompe el modelo de manera silenciosa. Fue una de las lecciones mas concretas de la sesion.
- **El analisis de errores ensena mas que la accuracy global**: el numero 61.86% no me decia donde fallaba el modelo; el classification report y la confusion matrix si.
- **El deploy es la mitad del proyecto**: hacer que el modelo prediga es facil; hacer que cualquiera pueda usarlo desde una URL publica obliga a pensar en rutas relativas, tamano del archivo `.keras`, versiones de TensorFlow y `requirements.txt` minimos.

---

## Estructura del repositorio

```text
modelo_predictivo_tenis/
├── app.py                        # App Streamlit (carga modelo + UI)
├── mi_modelo.keras               # Modelo entrenado (~96 MB)
├── clases.json                   # Mapeo {clase: indice}
├── requirements.txt              # Dependencias para Streamlit Cloud
├── .python-version               # Python 3.12
├── Modulo4_ProyectoFinal.ipynb   # Notebook completo del proyecto
└── README.md
```

---

## Como correrlo localmente

```bash
# 1. Crear entorno (Python 3.12)
python -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar la app
streamlit run app.py
```

La app abrira `http://localhost:8501` en el navegador.

---

## Tecnologias usadas

- **Python 3.12**
- **TensorFlow / Keras 2.21** — definicion, entrenamiento y persistencia del modelo (`.keras`).
- **ResNet50 (ImageNet)** — backbone de Transfer Learning.
- **scikit-learn** — `confusion_matrix`, `classification_report`.
- **Matplotlib + Seaborn** — visualizacion de curvas, matriz y errores.
- **Pillow** — lectura y resize de imagenes.
- **Streamlit 1.57** — interfaz web.
- **Streamlit Cloud** — hosting publico conectado a GitHub.
- **Google Colab + Google Drive** — entrenamiento gratuito con GPU y almacenamiento del dataset.

---

## Autor

**Luis Alonso Cornejo Jimenez**
Universidad Don Bosco - Diplomado Python Avanzado
Modulo 4 - Proyecto Final
GitHub: [@alonsocj](https://github.com/alonsocj)
