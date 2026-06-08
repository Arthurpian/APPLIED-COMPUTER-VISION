# 🌍 Monitoramento Espacial via Satélite com IA

Este projeto é uma solução autoral de **Visão Computacional** desenvolvida para a Global Solution. O sistema utiliza Redes Neurais Convolucionais (CNNs) treinadas do zero para classificar automaticamente o uso do solo através de imagens multiespectrais do satélite Sentinel-2 (dataset EuroSAT). 

O objetivo é fornecer uma ferramenta ágil para auditoria ambiental, detecção de desmatamento e monitoramento do avanço urbano, alcançando **88.22% de acurácia** em testes inéditos.

---

## 🚀 Funcionalidades

* **Classificação Multiclasse:** Identifica 10 categorias de terreno (Florestas, Rios, Rodovias, Áreas Residenciais, etc.).
* **Análise de Composição (Top-3):** Utiliza ativação Softmax para exibir as 3 classes predominantes na imagem com suas respectivas porcentagens de confiança.
* **Interface Web Interativa:** Aplicação front-end construída em Streamlit para upload de imagens e análise em tempo real.
* **Treinamento Otimizado:** Implementação de `Data Augmentation`, `Dropout` (50%) e `BatchNorm2d` para evitar overfitting e estabilizar o aprendizado.

---

## 📂 Estrutura do Projeto

* `app.py`: Interface Web interativa para o usuário final (Streamlit).
* `treinar_e_comparar.py`: Orquestrador que baixa o dataset, treina as IA's, salva os pesos e gera a Matriz de Confusão.
* `modelo_cnn.py`: Fábrica de arquiteturas contendo a estrutura matemática das redes neurais (Simples e Avançada).
* `cnn_avancada.pth`: Arquivo de pesos pré-treinado do modelo vencedor (Cérebro da IA).
* `imagens_de_teste/`: Pasta contendo exemplos de imagens de satélite para testar a aplicação.

---

## 🛠️ Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python instalado e de criar um ambiente virtual (`venv`) antes de instalar as dependências.

### 1. Instalação das Bibliotecas
Clone este repositório e execute o comando abaixo no terminal para instalar todas as ferramentas necessárias:

```bash
pip install torch torchvision streamlit scikit-learn matplotlib seaborn pillow
```

### 2. Rodando a Interface Web (Recomendado)
A IA já foi pré-treinada e os pesos estão salvos no repositório. Para testar o sistema imediatamente com a interface gráfica, execute:

```bash
streamlit run app.py
```
*Uma aba será aberta no seu navegador (geralmente em `localhost:8501`). Utilize as fotos da pasta `imagens_de_teste` para ver a IA em ação!*

### 3. Executando o Treinamento do Zero (Avançado)
Caso deseje auditar o projeto, treinar a rede neural do zero, testar o Data Augmentation e gerar os gráficos comparativos, execute o orquestrador:

```bash
python treinar_e_comparar.py
```
*Atenção: Este script fará o download automático do dataset EuroSAT (~90MB) caso a pasta não exista. O treinamento completo pode levar cerca de 15 minutos dependendo do seu hardware.*

---
