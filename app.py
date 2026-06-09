import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from modelo_cnn import CNN_Avancada
import numpy as np
import matplotlib.pyplot as plt
import io

# 1. Configuração da Página
st.set_page_config(page_title="Plataforma Ambiental", page_icon="🌍", layout="wide")

# ==========================================
# CÉREBRO DA IA (Carregamento Global)
# ==========================================
# Carregamos a IA aqui no topo uma única vez, para que ambas as telas possam usá-la.
@st.cache_resource 
def carregar_modelo():
    modelo = CNN_Avancada()
    modelo.load_state_dict(torch.load('cnn_avancada.pth', map_location=torch.device('cpu')))
    modelo.eval()
    return modelo

modelo = carregar_modelo()

classes = [
    'Plantação Anual', 'Floresta', 'Vegetação Rasteira', 'Rodovia', 
    'Área Industrial', 'Pasto', 'Plantação Permanente', 
    'Área Residencial', 'Rio', 'Mar/Lago'
]

transformacao = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ==========================================
# MENU LATERAL (SIDEBAR)
# ==========================================
st.sidebar.title("Navegação")
st.sidebar.markdown("Escolha o módulo da plataforma:")

# Cria os botões do menu
modulo_escolhido = st.sidebar.radio(
    "Módulos Disponíveis:",
    ["1️⃣ Classificador Simples (Projeto 1)", "2️⃣ Oráculo do Futuro (Projeto 2)"]
)

# ==========================================
# ROTEAMENTO DAS TELAS
# ==========================================

if modulo_escolhido == "1️⃣ Classificador Simples (Projeto 1)":
    st.title("🛰️ Classificador de Imagens de Satélite")
    st.markdown("Faça o upload de uma imagem para descobrir o tipo de terreno atual.")

    imagem_upada = st.file_uploader("Escolha uma imagem de satélite (JPG ou PNG)", type=["jpg", "jpeg", "png"])

    if imagem_upada is not None:
        imagem = Image.open(imagem_upada).convert('RGB')
        st.image(imagem, caption='Imagem Capturada pelo Satélite', use_container_width=True)
        
        st.write("Processando a imagem nos servidores orbitais...")
        
        img_tensor = transformacao(imagem).unsqueeze(0) 
        
        with torch.no_grad():
            previsao = modelo(img_tensor)
            confiancas = torch.nn.functional.softmax(previsao[0], dim=0) 
        
            top3_probabilidades, top3_indices = torch.topk(confiancas, 3)

        st.markdown("### 🎯 Top 3 Previsões da IA:")

        opcao1_nome = classes[top3_indices[0].item()]
        opcao1_valor = top3_probabilidades[0].item() * 100
        
        opcao2_nome = classes[top3_indices[1].item()]
        opcao2_valor = top3_probabilidades[1].item() * 100
        
        opcao3_nome = classes[top3_indices[2].item()]
        opcao3_valor = top3_probabilidades[2].item() * 100

        st.success(f"🥇 **1ª Opção:** {opcao1_nome} ({opcao1_valor:.2f}%)")
        st.warning(f"🥈 **2ª Opção:** {opcao2_nome} ({opcao2_valor:.2f}%)")
        st.info(f"🥉 **3ª Opção:** {opcao3_nome} ({opcao3_valor:.2f}%)")

elif modulo_escolhido == "2️⃣ Oráculo do Futuro (Projeto 2)":
    st.title("🌍 Oráculo Ambiental: Máquina do Tempo Orbital")
    st.markdown("Faça o upload de uma sequência temporal (3 fotografias) da mesma coordenada para projetarmos o cenário futuro de risco.")

    # NOVO FRONTEND: Criando a Linha do Tempo em 3 Colunas
    st.markdown("### ⏱️ Linha do Tempo (Histórico de 5 em 5 anos)")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📸 Foto 1 (Ex: 2010)")
        img1_up = st.file_uploader("Upload Ano 1", type=["jpg", "png"], key="img1")

    with col2:
        st.warning("📸 Foto 2 (Ex: 2015)")
        img2_up = st.file_uploader("Upload Ano 2", type=["jpg", "png"], key="img2")

    with col3:
        st.error("📸 Foto 3 (Ex: 2020)")
        img3_up = st.file_uploader("Upload Ano 3", type=["jpg", "png"], key="img3")

    # Gatilho: Só roda a IA quando as 3 imagens estiverem na tela
    st.markdown("---")
    if img1_up and img2_up and img3_up:
        
        colA, colB, colC = st.columns(3)
        
        img1 = Image.open(img1_up).convert('RGB')
        colA.image(img1, caption='Cenário Ano 1', use_container_width=True)
        
        img2 = Image.open(img2_up).convert('RGB')
        colB.image(img2, caption='Cenário Ano 2', use_container_width=True)
        
        img3 = Image.open(img3_up).convert('RGB')
        colC.image(img3, caption='Cenário Ano 3', use_container_width=True)

        st.success("✅ Sequência temporal carregada com sucesso! Aguardando motores de previsão...")
        
        # ==========================================
        # FASES 2 E 3: INFERÊNCIA E CÁLCULO DE FUTURO
        # ==========================================
        st.markdown("---")
        st.markdown("### 🧠 Análise Temporal e Previsão (Forecasting)")
        
        def analisar_imagem(imagem_pil):
            img_tensor = transformacao(imagem_pil).unsqueeze(0)
            with torch.no_grad():
                previsao = modelo(img_tensor)
                confiancas = torch.nn.functional.softmax(previsao[0], dim=0)
            return confiancas

        with st.spinner("Processando tensores e calculando regressão estatística..."):
            dados_ano1 = analisar_imagem(img1)
            dados_ano2 = analisar_imagem(img2)
            dados_ano3 = analisar_imagem(img3)
            
            probabilidade_atual, indice_foco = torch.max(dados_ano3, 0)
            classe_foco = classes[indice_foco.item()]
            
            hist_ano1 = dados_ano1[indice_foco].item() * 100
            hist_ano2 = dados_ano2[indice_foco].item() * 100
            hist_ano3 = probabilidade_atual.item() * 100
            
            crescimento_1_para_2 = hist_ano2 - hist_ano1
            crescimento_2_para_3 = hist_ano3 - hist_ano2
            taxa_media = (crescimento_1_para_2 + crescimento_2_para_3) / 2
            
            projecao_ano4 = hist_ano3 + taxa_media
            projecao_ano4 = max(0.0, min(100.0, projecao_ano4))

        st.markdown("---")
        st.markdown(f"### 🎯 Foco da Análise e Diagnóstico: **{classe_foco}**")
        st.write(f"Monitorando a porcentagem de território ocupado por **{classe_foco}** ao longo do tempo:")
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        
        col_res1.metric(f"Ano 1 (Base)", f"{hist_ano1:.1f}%")
        col_res2.metric(f"Ano 2", f"{hist_ano2:.1f}%", f"{crescimento_1_para_2:.1f}% (Variação)")
        col_res3.metric(f"Ano 3 (Atual)", f"{hist_ano3:.1f}%", f"{crescimento_2_para_3:.1f}% (Variação)")
        col_res4.metric(f"🔮 Previsão Ano 4", f"{projecao_ano4:.1f}%", f"{taxa_media:.1f}% (Tendência)", delta_color="off")
        
        if taxa_media > 5:
            st.error(f"**Alerta Crítico:** A expansão de '{classe_foco}' apresenta crescimento contínuo e acelerado. Se a tendência se mantiver, este elemento dominará a região em 5 anos.")
        elif taxa_media < -5:
            st.warning(f"**Alerta de Escassez:** O território de '{classe_foco}' está sofrendo redução drástica. Atenção imediata recomendada para avaliar o impacto ambiental.")
        else:
            st.success(f"**Estabilidade:** A área de '{classe_foco}' apresenta flutuações naturais e controladas. O cenário projetado indica estabilidade territorial.")

        # ==========================================
        # GERADOR DE MAPA DE RISCO TÉRMICO INTELIGENTE
        # ==========================================
        st.markdown("---")
        st.markdown(f"### 🗺️ Mapa de Risco Projetado (Projeção Ano 4)")
        
        with st.spinner("Calculando diferença de pixels e gerando máscara térmica..."):
            
            img1_base = img1.resize((64, 64))
            img3_atual = img3.resize((64, 64))
            
            img1_gray = np.array(img1_base.convert('L'), dtype=np.float32)
            img3_gray = np.array(img3_atual.convert('L'), dtype=np.float32)
            
            diferenca = np.abs(img3_gray - img1_gray)
            diferenca_normalizada = diferenca / (np.max(diferenca) + 1e-6)
            
            color_map = np.zeros((64, 64, 4), dtype=np.float32)
            color_map[:,:,0] = 1.0  
            
            intensidade_risco = projecao_ano4 / 100.0  
            color_map[:,:,3] = diferenca_normalizada * intensidade_risco * 0.9
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(np.array(img3_atual))
            ax.imshow(color_map)             
            ax.axis('off')                  
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            imagem_final = Image.open(buf)
            plt.close(fig)

        st.image(imagem_final, 
                 caption=f"Zonas de Alto Risco de Expansão de {classe_foco}. Risco Geral Projetado: {projecao_ano4:.1f}%.", 
                 use_container_width=True)
