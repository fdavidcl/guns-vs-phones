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
geometry: a4paper, margin=1.2in
header-includes:
  - \usepackage{rotating}
  - \usepackage{pdflscape}
lang: es
---

# Descripción del problema

La actividad consiste en participar en la competición de Kaggle [pistolas vs smartphones](https://www.kaggle.com/c/pistolas-vs-smartphones-con-deep-learning/leaderboard) por grupos, donde el problema a resolver es implementar un modelo de Deep Learning para clasificar en dos clases diversas imágenes de pistolas y smartphones con la mayor tasa de acierto posible. 

La competición consta de 743 instancias para entrenamiento y 799 instancias para test sin etiquetar.


\begin{figure}[htbp]
\centering
\includegraphics[width=0.4\textwidth]{Train/Pistol/0.69-500x500.jpg}
\includegraphics[width=0.4\textwidth]{Train/Smartphone/smartphone_4.jpg}
\caption{\label{fig.examples}Ejemplos de instancias de pistola (izquierda) y smartphone (derecha)}
\end{figure}

# Modelo desarrollado

En los siguientes apartados se describen el modelo de clasificación desarrollado, los resultados obtenidos y otras alternativas consideradas.

## Preprocesamiento

El primer paso para desarrollar el trabajo fue diseñar una representación de los datos. Se decidió redimensionar todas las imágenes a un tamaño de 224x224 mediante una interpolación bicúbica, ya que fue la interpolación con mejores resultados pese a ser más costosa. Una vez leidas las imágenes, se les aplicó un procesamiento incluido en Keras denominado _caffe_. Este preprocesamiento convierte primero las imágenes de RGB a BGR, luego centra cada canal de color en el cero con respecto al conjunto de datos ImageNet, sin realizar ningún escalado. En Keras hay disponibles dos tipos más de preprocesamiento, _tf_ y _torch_, pero en nuestro caso _caffe_ dio mejores resultados.

## Descripción

Puesto que el problema consiste en realizar clasificación de imágenes con pocos datos de entrenamiento, planteamos una solución basada en *transfer learning*, en la que utilizamos redes neuronales convolucionales entrenadas previamente para reconocimiento de imágenes y las adaptamos al problema de distinguir pistolas de smartphones.

Para ello, se ha reutilizado la estructura y pesos de una red conocida entrenada en el conjunto de datos de la competición Imagenet, y se han sustituido las capas densas del final de la red por nuevas capas sin entrenar que tengan dos unidades a la salida. En particular, el modelo que se ha usado como base para nuestra solución es VGG16[^vgg16]. Es importante notar que los pesos de la sección convolucional de la red son fijos durante todo el entrenamiento, el espacio de búsqueda se compone así de los pesos asociados a las capas densas de la salida.

La red VGG16 se compone de cinco bloques convolucionales: los dos primeros con dos capas convolucionales y una de max-pooling cada uno, y los tres últimos con tres capas de convolución y una de max-pooling cada uno. Al final de la red normalmente hay dos capas totalmente conectadas de 4096 unidades y una capa con tantas unidades como clases (en el caso de Imagenet, 1000). En nuestro caso, sin embargo, se ha optado por reducir el tamaño de estas capas para facilitar la optimización en un espacio de búsqueda menor. Para ello, las capas densas son de 512, 64 y 2 unidades respectivamente. En la figura \ref{fig.layers} se puede observar la estructura del modelo desarrollado.

La función objetivo a minimizar de la red neuronal se ha establecido a la entropía cruzada (categórica) entre la salida de la red y la clasificación real de las imágenes. Además, se han probado varios algoritmos de optimización y se ha escogido finalmente RMSprop[^rmsprop], ya que es un optimizador robusto que suele funcionar bien con los parámetros por defecto.


[^vgg16]: Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

[^rmsprop]: Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning, 4(2), 26-31.

## Resultados

De entre los diferentes modelos probados, el mejor ha sido el basado en la sección convolucional de VGG16 con tres capas densas adicionales: dos capas ocultas de 512 y 64 neuronas respectivamente y una de salida con dos neuronas. La función de activación en las dos primeras es ReLU, mientras que la final es softmax. Esta red, optimizada durante 10 épocas mediante RMSprop con los datos de entrenamiento, es capaz de producir un acierto del 100% en los datos de test.

También se ha probado a basar el modelo en la red ResNet50, dejando la misma arquitectura para las capas finales que con VGG16 (512 + 64 + 2) y 10 epochs, obteniendo resultados ligeramente peores (1 instancia mal clasificada). El último modelo competitivo probado fue VGG19, con una estructura de capas densas de 1024 + 128 + 2 unidades y durante 4 épocas, obteniendo resultados en test muy buenos (> 99%), que se mantienen incluso con 3 épocas.

\begin{landscape}
\begin{figure}[p]
\centering
\includegraphics[width=1.6\textwidth,trim={5cm 8cm 3cm 7cm},clip]{vgg16.png}
\caption{\label{fig.layers} Ilustración gráfica de la red neuronal correspondiente a la solución desarrollada. La sección convolucional corresponde a la estructura original de VGG16. Tras esta se aplica un average-pooling global de forma que se reducen los filtros a sus medias, y se pasan por capas densas de 512, 64 y 2 unidades.}
\end{figure}
\end{landscape}

En la siguiente tabla se recogen algunas variantes competitivas de la solución desarrollada y su rendimiento en training (mediante la entropía cruzada) y en test (mediante accuracy en rankings privado y público). En todos los casos se ha realizado el mismo preprocesamiento (interpolación bicúbica a tamaño 224x224 y preparación de tipo *caffe*) y se mantienen constantes los parámetros no mencionados.

| Modelo            | learn. rate   | w. decay | epochs | loss      |   acc. pri.  |   acc. pub.  |
|:------------------|-----:|------:|-------:|----------:|-------:|-------:|
| VGG16 + 512 + 64 + 2   | 0.001 | 0     | 10     | $10^{-7}$ |    1   |    1   |
| ResNet50 + 512 + 64 + 2 | 0.002 | 0.001  | 10     | .1252     |    1   | .99749 |
| VGG19 + 1024 + 128 + 2 | 0.001 | 0  | 3      | .0611     | .99500 | .99498 |

## Otros modelos probados

Además de VGG16, se han entrenado y evaluado otros modelos con pesos pre-entrenados para Imagenet disponibles en Keras, como son
VGG19, InceptionV3, ResNet50, InceptionResNetV2 y XCeption.

A pesar de que para el dataset de Imagenet el modelo Xception obtuvo la mayor tasa de acierto, en nuestro ejemplo obtuvo malos resultados, clasificando erróneamente más de 20 instancias. Es posible que, al tratarse de una red muy profunda, la salida de su sección convolucional sea demasiado compleja como para especializarse con pocos datos en la diferenciación de pistolas y smartphones.<!--Esto puede deberse a que en este trabajo no se han configurado 22,910,480 parámetros ni se ha llevado el modelo Xception a un nivel de profundidad de 126, como se hizo para Imagenet, maximizando el rendimiento del clasificador. -->

Los modelos base VGG16 y VGG19 dieron ambos buenos resultados en este problema como también sucedió en la competición de Imagenet, siendo VGG16 el mejor de los modelos probados en nuestro experimento. <!--Ambos modelos cuentan con que no necesitan de un nivel de profundidad muy elevado para dar buenos resultados.--> Las experimentaciones con ResNet50 demuestran que, para estos datos, el modelo puede dar muy buenos resultados con un preprocesamiento como caffe, una estructura de capas densas de 512 + 64 + 2 unidades, el optimizador RMSprop y un número alto de epochs (10), pero que sin embargo puede dar malos resultados con otros parámetros, como un tamaño de imagen menor, una estructura de 1024 + 128 + 2, el optimizador adam y un número bajo de epochs (4).

Los modelos InceptionV3 e InceptionResNetV2 fueron entrenados con la estructura de capas densas 1024 + 128 + 2, usando el optimizador adam y durante 4 epochs, pero en ningún caso produjeron resultados significativos.
 
# Distribución del trabajo

Las siguientes son las tareas realizadas para la compleción de la práctica y las personas que se encargaron de las mismas:

- Búsqueda y recopilación de información: Cristina Heredia
- Desarrollo del código Python: Alejandro Alcalde y David Charte
- Experimentación con diversos parámetros y modelos: Cristina Heredia, Alejandro Alcalde, David Charte
