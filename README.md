
# **Análisis de Sentimientos en Reseñas de Pueblos Mágicos - Reto MeIA 2025**

Este repositorio contiene la solución para el reto de **Análisis de Sentimientos en Pueblos Mágicos Mexicanos** del Macroentrenamiento en Inteligencia Artificial (MeIA) 2025. El objetivo es clasificar la polaridad de reseñas turísticas en una escala del 1 (muy negativo) al 5 (muy positivo) utilizando un modelo de lenguaje basado en Transformers[cite: 1].

-----

## **Descripción General** 📜

El proyecto implementa un flujo de trabajo de Procesamiento de Lenguaje Natural (PLN) para realizar análisis de sentimientos. Se utiliza un modelo **BERT pre-entrenado para español**, que es ajustado (fine-tuned) con un conjunto de datos específico de reseñas de Pueblos Mágicos. Finalmente, el modelo entrenado se usa para predecir la polaridad de nuevas reseñas no etiquetadas[cite: 1].

### **Tecnologías y Modelos** 🤖

  * **Modelo Base**: `dccuchile/bert-base-spanish-wwm-cased`, un modelo de la arquitectura BERT optimizado para el español.
  * **Librerías Principales**: `Transformers` de Hugging Face, `datasets`, `PyTorch`, `pandas` y `scikit-learn`.

-----

## **Estructura del Proyecto** 📂

```
.
├── datos/
│   ├── MeIA_2025_train.xlsx
│   └── MeIA_2025_test_wo_labels.xlsx
├── resultados/
│   └── MEIAbot-2.txt
└── modelo2.py
```

  * **`datos/`**: Contiene los archivos de datos proporcionados para el reto.
      * `MeIA_2025_train.xlsx`: **5,000 reseñas etiquetadas** para entrenamiento y validación.
      * `MeIA_2025_test_wo_labels.xlsx`: **2,500 reseñas sin etiquetar** para la predicción final.
  * **`modelo2.py`**: El script principal que contiene toda la lógica para cargar datos, entrenar el modelo y generar las predicciones.
  * **`resultados/`**: Carpeta donde se guarda el archivo final con las predicciones.
      * `MEIAbot-2.txt`: El archivo de salida generado por el script, con las predicciones en el formato requerido.

-----

## **Funcionamiento del Código (`modelo2.py`)** ⚙️

El script `modelo2.py` automatiza el proceso completo en cuatro pasos principales:

### **Paso 1: Carga y Preparación de Datos**

1.  **Carga de Datos**: Se leen los archivos `MeIA_2025_train.xlsx` y `MeIA_2025_test_wo_labels.xlsx` usando la librería `pandas`.
2.  **Limpieza y Mapeo**:
      * Se eliminan las columnas de metadatos (`Town`, `Region`, `Type`), ya que el modelo se centrará exclusivamente en el contenido textual de la reseña.
      * La columna `Review` se renombra a `texto` y `Polarity` a `labels` para compatibilidad con las librerías de Hugging Face.
      * **Paso Crítico**: Las etiquetas de polaridad (1 a 5) se transforman a un rango de **0 a 4** (`labels - 1`), ya que los modelos de clasificación requieren índices que comiencen en cero.
3.  **División del Conjunto**: El conjunto de datos etiquetado se divide en un **80% para entrenamiento** y un **20% para validación**. La división estratificada (`stratify`) asegura que la proporción de cada clase de sentimiento sea la misma en ambos conjuntos.
4.  **Conversión a `Dataset`**: Los DataFrames de `pandas` se convierten a objetos `Dataset` de la librería de Hugging Face para optimizar el manejo de memoria y el rendimiento.

### **Paso 2: Tokenización**

1.  **Carga del Tokenizador**: Se carga el tokenizador correspondiente al modelo `dccuchile/bert-base-spanish-wwm-cased`.
2.  **Función de Tokenización**: Se define una función que toma el texto de las reseñas y lo convierte en un formato numérico que el modelo BERT puede entender. Cada texto se **trunca o rellena** para tener una longitud fija de **128 tokens**.
3.  **Aplicación**: Esta función se aplica a todos los conjuntos de datos (entrenamiento, validación y el de prueba sin etiquetas).

### **Paso 3: Configuración y Fine-Tuning del Modelo**

1.  **Carga del Modelo**: Se carga el modelo `AutoModelForSequenceClassification` a partir del checkpoint `dccuchile/bert-base-spanish-wwm-cased` y se configura para una tarea de clasificación con **5 etiquetas** (de 0 a 4).
2.  **Argumentos de Entrenamiento**: Se definen los hiperparámetros del entrenamiento a través de `TrainingArguments`. Los más importantes son:
      * `num_train_epochs=5`: El modelo procesará todo el conjunto de entrenamiento 5 veces.
      * `per_device_train_batch_size=8`: Se entrenará en lotes de 8 reseñas a la vez.
      * `eval_strategy="epoch"`: El rendimiento del modelo se medirá contra el conjunto de validación al final de cada época.
      * `load_best_model_at_end=True`: Al finalizar, se cargará automáticamente el checkpoint del modelo que obtuvo los mejores resultados en la validación.
3.  **Métricas de Evaluación**: Se define una función (`compute_metrics`) para calcular la precisión (`accuracy`), F1-Score, `precision` y `recall` durante la fase de validación.
4.  **Entrenamiento**: Se inicializa el `Trainer` con el modelo, los datos y los argumentos, y se ejecuta el método `trainer.train()` para comenzar el proceso de fine-tuning.

### **Paso 4: Predicción y Generación de Resultados**

1.  **Predicción**: Una vez entrenado, el modelo se utiliza para predecir la polaridad del conjunto de datos sin etiquetar (`MeIA_2025_test_wo_labels.xlsx`).
2.  **Procesamiento de Salida**: El modelo devuelve "logits" (valores numéricos) para cada clase. Se aplica la función `argmax` para determinar la clase con la puntuación más alta, que corresponde a la etiqueta predicha (0 a 4).
3.  **Generación del Archivo Final**:
      * Las predicciones se añaden como una nueva columna al DataFrame de los datos sin etiquetar.
      * Se genera el archivo de salida, por ejemplo `MEIAbot-2.txt`. En este archivo, a cada predicción se le suma 1 para **convertirla de nuevo a la escala original de 1 a 5**, cumpliendo con el formato `MeIA {ID} {Polaridad}`.

-----

## **Cómo Ejecutar el Proyecto** ▶️

1.  **Instalar dependencias**:
    ```bash
    pip install pandas openpyxl torch transformers datasets scikit-learn
    ```
2.  **Organizar los archivos**: Asegúrate de que los archivos `.xlsx` estén en la carpeta correcta (`/datos/` o la ruta especificada en el script).
3.  **Ejecutar el script**:
    ```bash
    python modelo2.py
    ```
4.  **Verificar la salida**: El archivo con las predicciones se generará en la ruta de salida especificada dentro del script (ej. `/resultados/`).
