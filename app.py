import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from modelo_cnn import CNN_Avancada

st.set_page_config(page_title="Monitoramento Espacial IA", page_icon="🌍")
st.title("🌍 Monitoramento Espacial via Satélite")
st.markdown("Faça o upload de uma imagem de satélite para classificar o uso do solo com nossa Inteligência Artificial.")

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
imagem_upada = st.file_uploader("Escolha uma imagem de satélite (JPG ou PNG)", type=["jpg", "jpeg", "png"])

if imagem_upada is not None:
    imagem = Image.open(imagem_upada).convert('RGB')
    st.image(imagem, caption='Imagem Capturada pelo Satélite', use_column_width=True)
    
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