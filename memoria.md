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

| Pre                   | Modelo            | Dout | lr   | decay | epochs | loss      |   Pri  |   Púb  |
|-----------------------|-------------------|------|------|-------|--------|-----------|:------:|:------:|
| bicubic,224x224,caffe | VGG16+512+64+2    | .2   | .001 | 0     | 10     | $10^{-7}$ |    1   |    1   |
| bicubic,220x220,caffe | VGG16+512+64+2    | .2   | .001 | 0     | 10     | $10^{-7}$ |    1   |    1   |
| caffe                 | ResNet50+512+64+2 | .2   | .002 | .001  | 10     | .1252     |    1   | .99749 |
| bicubic,220x220,caffe | VGG16+512+64+2    | .2   | .001 | 0     | 10     | $10^{-7}$ |    1   | .99248 |
|                       | VGG19+1024+128+2  | .2   | def. | def.  | 4      | .1093     | .99250 | .99498 |

## Otros modelos probados

VGG19, InceptionV3, ResNet50, InceptionResNetV2, XCeption

# Distribución del trabajo

- Búsqueda y recopilación de información: Cristina Heredia
- Desarrollo del código Python: Alejandro Alcalde y David Charte
- Experimentación con diversos parámetros y modelos: Cristina Heredia, Alejandro Alcalde, David Charte
