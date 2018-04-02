---
title: Clasificación de pistolas y smartphones
author:
  - Cristina Heredia
  - Alejandro Alcalde
  - David Charte
date: Minería de datos, Aspectos Avanzados - Universidad de Granada
compilation: pandoc -o memoria.pdf memoria.md
graphics: yes
toc: yes
numbersections: yes
linkcolor: blue
---

# Descripción del problema

La actividad consiste en participar la competición de Kaggle [pistolas vs smartphones](https://www.kaggle.com/c/pistolas-vs-smartphones-con-deep-learning/leaderboard) por grupos, donde el problema a resolver es implementar un modelo de Deep Learning para clasificar en dos clases diversas imágenes de pistolas y smartphones con la mayor tasa de acierto posible. \newline
La competición consta de 743 instancias para entrenamiento sin etiquetar y 799 instancias para test.


\begin{figure}[htbp]
\centering
\includegraphics[width=0.4\textwidth]{Train/Pistol/0.69-500x500.jpg}
\includegraphics[width=0.4\textwidth]{Train/Smartphone/smartphone_4.jpg}
\caption{\label{fig.examples}Ejemplos de instancias de pistola (izquierda) y smartphone (derecha)}
\end{figure}

# Modelo desarrollado

## Descripción

Hablar de **transfer learning**.

## Resultados

El primer paso para desarrollar el trabajo fue cómo leer los datos, se decidió redimensionarlas todas a un tamaño de 224x224 y con una interpolación bicúbica, ya que fue la interpolación con mejores resultados. Una vez leidas las imágenes, se les aplicó un procesamiento usando _caffe_. Este preprocesamiento convierte primero las imágenes de RGB a BGR, luego centra cada canal de color en el cero con respecto al conjunto de datos ImageNet, sin realizar ningún escalado. Hay dos tipos más de preprocesamiento, _tf, torch_, pero _caffe_ dio mejores resultados.

En cuanto a los modelos, el mejor ha sido VGG16 con tres capas y 10 epochs: La capa de entrada densa con 512 neuronas, una capa oculta densa de 64 neuronas y una de salida con dos neuronas. La función de activación en las dos primeras es relu, mientras que la final es softmax. Los pesos usados, como se comentó en el párrafo anterior han sido los de ImageNet.

También se probó con ResNet50, dejando la misma arquitectura para la red que en VGG16 (512+64+2) y 10 epochs, obteniendo resultados ligeramente peores (1 Instacia mal clasificada).

El último modelo competitivo probado fue VGG19, con una estructura 1024 + 128 + 2 y 4 epochs, obteniendo resultados muy buenos (99%), incluso bajando el número de epochs a 3 se obtienen resultados del 99%.

| Pre                   | Modelo            | Dout | lr   | decay | epochs | loss      |   Pri  |   Púb  |
|-----------------------|-------------------|------|------|-------|--------|-----------|:------:|:------:|
| bicubic,224x224,caffe | VGG16+512+64+2    | .2   | .001 | 0     | 10     | $10^{-7}$ |    1   |    1   |
| bicubic,224x224,caffe | VGG16+512+64+2    | .2   | .001 | 0     | 10     | $10^{-7}$ |    1   |    1   |
| bicubic,224x224,caffe | ResNet50+512+64+2 | .2   | .002 | .001  | 10     | .1252     |    1   | .99749 |
| bicubic,224x224,caffe | VGG19+1024+128+2  | .2   | def. | def.  | 4      | .1093     | .99250 | .99498 |
| bicubic,224x224,caffe | VGG19+1024+128+2  | .2   | def. | def.  | 3      | .0611     | .99500 | .99498 |

## Otros modelos probados
Además de VGG16, se han entrenado y evaluado otros modelos con pesos pre-entrenados para Imagenet disponibles en Keras, como son
VGG19, InceptionV3, ResNet50, InceptionResNetV2 y XCeption. \newline A pesar de que para el dataset de Imagenet el modelo Xception obtuvo la mayor tasa de acierto, en nuestro ejemplo obtuvo malos resultados, clasificando erróneamente más de 20 instancias. Esto puede deberse a que en este trabajo no se han configurado 22,910,480 parámetros ni se ha llevado el modelo Xception a un nivel de profundidad de 126, como se hizo para Imagenet, maximizando el rendimiento del clasificador. \newline Los modelos VGG16 y VGG19 dieron ambos buenos resultados en este problema como también sucedió en la competición de Imagenet, siendo VGG16 el mejor de los modelos probados en nuestro experimento. Ambos modelos cuentan con que no necesitan de un nivel de profundidad muy elevado para dar buenos resultados. Las experimentaciones con ResNet50 demuestran que, para estos datos, el modelo puede dar muy buenos resultados con un preprocesamiento como caffe, una estructura de 512 + 64 + 2, el optimizador rmsprop y un número alto de epochs (10), pero que sin embargo puede dar malos resultados sin preprocesamiento, una estructura (1024 + 128 + 2), el optimizador adam y un número bajo de epochs (4). \newline Los modelos InceptionV3 e InceptionResNetV2 fueron entrenados con la estructura (1024 + 128 + 2) usando el optimizador adam y 4 epochs, pero en ningún caso produjeron resultados significativos.
 
# Distribución del trabajo

- Búsqueda y recopilación de información: Cristina Heredia
- Desarrollo del código Python: Alejandro Alcalde y David Charte
- Experimentación con diversos parámetros y modelos: Cristina Heredia, Alejandro Alcalde, David Charte
