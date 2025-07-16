# 🧠 MeIA 2025 - Reto: Análisis de Sentimientos en Pueblos Mágicos Mexicanos

## 📄 Descripción General

Este repositorio contiene dos archivos correspondientes al conjunto de datos proporcionado para el macroentrenamiento **MeIA 2025**, en el contexto del reto de *Análisis de Sentimientos en Pueblos Mágicos Mexicanos*. El objetivo principal es desarrollar modelos capaces de predecir la polaridad del sentimiento expresado en reseñas turísticas asociadas a distintos destinos mexicanos reconocidos como **Pueblos Mágicos**.

## 📂 Archivos

| Archivo                        | Descripción                                                                |
|--------------------------------|----------------------------------------------------------------------------|
| `MeIA_2025_train.xlsx`         | Conjunto de datos etiquetado utilizado para entrenar los modelos.          |
| `MeIA_2025_test_wo_labels.xlsx`| Conjunto de datos no etiquetado destinado para evaluación.                 |

---

## 🧪 Conjunto de entrenamiento (`MeIA_2025_train.xlsx`)

Este archivo contiene un total de **5,000** instancias correspondientes a reseñas de usuarios. Las columnas incluidas son:


- `Review`: Cuerpo textual de la reseña.
- `Polarity`: Valor de sentimiento expresado, con un rango ordinal de **1 (muy negativo)** a **5 (muy positivo)**.
- `Town`: Nombre del Pueblo Mágico asociado a la reseña.
- `Region`: Región geográfica de México (Estado).
- `Type`: Tipo de servicio reseñado (por ejemplo, Hotel, Restaurante, Atractivo).

### 📊 Distribución por clase de polaridad (entrenamiento)

| Polaridad | Número de instancias |
|-----------|----------------------|
| 1         | 800                  |
| 2         | 900                  |
| 3         | 1,000                |
| 4         | 1,100                |
| 5         | 1,200                |

---

## 🔍 Conjunto de prueba (`MeIA_2025_test_wo_labels.xlsx`)

Este archivo contiene **2,500** reseñas similares a las del conjunto de entrenamiento, pero **sin etiquetas de polaridad**. Se utilizará para la evaluación oficial del desempeño de los modelos desarrollados.

- Contiene las mismas columnas que el conjunto de entrenamiento, excepto `Polarity` y se agregó la columna `ID`.

---

## 🎯 Objetivo del reto

Entrenar modelos capaces de predecir de manera precisa la **polaridad** de nuevas reseñas de visitantes a los Pueblos Mágicos, permitiendo un análisis automatizado del sentimiento en contextos turísticos mexicanos.

---

## 📌 Consideraciones adicionales

- El corpus representa una muestra semi-balanceada de opiniones reales recolectadas de plataformas turísticas.
- Se espera que los modelos respeten la naturaleza ordinal de las clases y puedan generalizar a reseñas nuevas y variadas.


---
