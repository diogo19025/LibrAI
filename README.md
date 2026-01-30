
#  LibrAI

**Aprenda o alfabeto de LIBRAS de forma interativa usando visão computacional e machine learning.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange?logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikitlearn&logoColor=white)

## 📋 Sobre o Projeto

O **LibrAI** é uma aplicação educacional que utiliza a câmera do computador para reconhecer letras estáticas do alfabeto de **LIBRAS** (Língua Brasileira de Sinais) em tempo real.

O objetivo é ajudar pessoas que desejam **aprender LIBRAS**, especialmente o alfabeto manual, de uma forma gamificada e interativa — você faz o sinal com a mão e o sistema reconhece se está correto!

![LibrAI Demo](https://i.imgur.com/pXhXzEI.gif)

✨ Funcionalidades
Modo Fácil — Letras aleatórias com imagem de referência para aprender;

Modo Difícil — Soletrar palavras completas sem referência visual;

Feedback de dopamina — Som e flash verde na tela ao acertar;

Sistema de pontuação — +1 por letra, +2 bônus por palavra completa;

Detecção em tempo real — MediaPipe para landmarks da mão;

Classificação por Machine Learning — Random Forest treinado com dados próprios.

## 🛠️ Stack / Tecnologias

| Tecnologia | Uso |
|------------|-----|
| **Python 3.10+** | Linguagem principal |
| **OpenCV** | Captura de vídeo e processamento de imagem |
| **MediaPipe** | Detecção de landmarks da mão (21 pontos) |
| **scikit-learn** | Modelo de classificação (Random Forest) |
| **NumPy** | Manipulação de arrays e cálculos |
| **Pillow** | Conversão de imagens |
| **Pygame** | Reprodução de sons (feedback) |
| **Tkinter** | Interface gráfica |

## 📁 Estrutura do Projeto

```
LibrAI/
├── app.py                     # Aplicação principal
├── collect.py                 # Script para coletar dados de treino
├── hand_landmarker.task       # Modelo MediaPipe para detecção de mãos
│
├── models/
│   ├── librai_rf.joblib       # Modelo treinado (Random Forest)
│   └── train.py               # Script para treinar o modelo
│
├── assets/
│   ├── references/            # Imagens de referência (A.png, B.png, ...)
│   └── sounds/
│       ├── correctletter.mp3  # Som ao acertar letra
│       └── correctword.wav    # Som ao completar palavra
│
└── data/                      # Dados de treino (landmarks coletados)
```
## 🚀 Como Executar

### Pré-requisitos

- Python 3.10 ou superior
- Webcam funcional
- Mão esquerda (o modelo foi treinado com mão esquerda, palma voltada para a câmera)
  
### Instalação

```bash
# Clone o repositório
git clone https://github.com/diogo19025/LibrAI.git
cd LibrAI

# Instale as dependências
pip install opencv-python mediapipe numpy joblib pillow pygame

# Execute a aplicação
python main.py
```

## 🛹 Como Rodar

1. **Inicie o aplicativo** clicando em "Start"
2. **Escolha o modo**: Easy (com referência) ou Hard (soletrar palavras)
3. **Faça o sinal** da letra mostrada usando a **mão esquerda** com a **palma voltada para a câmera**
4. **Mantenha estável** por ~1 segundo para confirmar
5. **Acerte antes do tempo acabar!**

## 📝 Letras Suportadas

Atualmente o modelo reconhece **20 letras estáticas** de LIBRAS:

```
A  B  C  D  E  F  G  I  L  M
N  O  P  Q  R  S  T  U  V  W
```

> ⚠️ Letras que exigem movimento (H, J, K, X, Y, Z) não são suportadas nesta versão.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Diogo%20Soares-blue?logo=linkedin)](https://www.linkedin.com/in/diogos19/)
[![GitHub](https://img.shields.io/badge/GitHub-diogo19025-black?logo=github)](https://github.com/diogo19025)




