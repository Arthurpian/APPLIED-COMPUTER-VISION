import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from modelo_cnn import CNN_Simples, CNN_Avancada

batch_size = 32
epocas = 20  
taxa_aprendizado = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔬 Hardware utilizado: {device}")

transformacoes_treino = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  
    transforms.RandomRotation(degrees=15), 
    transforms.ToTensor()
])

transformacoes_teste = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

print("\nA carregar e dividir o dataset EuroSAT...")
dataset_base = torchvision.datasets.EuroSAT(root='./dataset', download=True, transform=transforms.ToTensor())

tamanho_treino = int(0.70 * len(dataset_base))
tamanho_val = int(0.15 * len(dataset_base))
tamanho_teste = len(dataset_base) - tamanho_treino - tamanho_val
treino_data, val_data, teste_data = random_split(dataset_base, [tamanho_treino, tamanho_val, tamanho_teste])


treino_data.dataset.transform = transformacoes_treino
val_data.dataset.transform = transformacoes_teste
teste_data.dataset.transform = transformacoes_teste

treino_loader = DataLoader(treino_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
teste_loader = DataLoader(teste_data, batch_size=batch_size, shuffle=False)

def executar_treino(modelo, nome_modelo):
    print(f"\n==========================================")
    print(f"🚀 Treinando: {nome_modelo}")
    print(f"==========================================")
    
    modelo = modelo.to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=taxa_aprendizado)
    
    historico_loss = []
    historico_acc = []
    
    for epoca in range(epocas):
        modelo.train()
        erro_acumulado = 0.0
        acertos = 0
        total = 0
        
        for imagens, rotulos in treino_loader:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
            
            otimizador.zero_grad()
            previsoes = modelo(imagens)
            erro = criterio(previsoes, rotulos)
            erro.backward()
            otimizador.step()
            
            erro_acumulado += erro.item()
            _, chutes = torch.max(previsoes.data, 1)
            total += rotulos.size(0)
            acertos += (chutes == rotulos).sum().item()
            
        acuracia_treino = 100 * acertos / total
        erro_medio = erro_acumulado / len(treino_loader)
        
        historico_loss.append(erro_medio)
        historico_acc.append(acuracia_treino)
        
        print(f"Época [{epoca+1}/{epocas}] -> Loss: {erro_medio:.4f} | Acurácia: {acuracia_treino:.2f}%")
        
    torch.save(modelo.state_dict(), f"{nome_modelo.lower()}.pth")
    return historico_loss, historico_acc

historico_loss_simples, historico_acc_simples = executar_treino(CNN_Simples(), "CNN_Simples")
historico_loss_avancada, historico_acc_avancada = executar_treino(CNN_Avancada(), "CNN_Avancada")

def avaliar_e_gerar_matriz(modelo, nome_modelo):
    modelo.load_state_dict(torch.load(f"{nome_modelo.lower()}.pth"))
    modelo.to(device)
    modelo.eval()
    
    acertos = 0
    total = 0
    todas_previsoes = []
    todos_rotulos = []
    
    with torch.no_grad():
        for imagens, rotulos in teste_loader:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
            previsoes = modelo(imagens)
            _, chutes = torch.max(previsoes.data, 1)
            
            total += rotulos.size(0)
            acertos += (chutes == rotulos).sum().item()

            todas_previsoes.extend(chutes.cpu().numpy())
            todos_rotulos.extend(rotulos.cpu().numpy())
            
    acuracia_teste = 100 * acertos / total
    print(f"🎯 ACURÁCIA FINAL - {nome_modelo}: {acuracia_teste:.2f}%")
    

    if nome_modelo == "CNN_Avancada":
        classes = ['AnnualCrop', 'Forest', 'HerbVeg', 'Highway', 'Industrial', 
                   'Pasture', 'PermCrop', 'Residential', 'River', 'SeaLake']
        cm = confusion_matrix(todos_rotulos, todas_previsoes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Matriz de Confusão - CNN Avançada')
        plt.ylabel('Classe Real')
        plt.xlabel('Previsão da IA')
        plt.tight_layout()
        plt.savefig("matriz_confusao.png")
        print("✅ Matriz de Confusão gerada e salva como 'matriz_confusao.png'!")

print("\n==========================================")
print("🎯 TESTE FINAL COM IMAGENS INÉDITAS")
print("==========================================")
avaliar_e_gerar_matriz(CNN_Simples(), "CNN_Simples")
avaliar_e_gerar_matriz(CNN_Avancada(), "CNN_Avancada")

# 5. Gráficos Comparativos
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epocas + 1), historico_acc_simples, label="CNN Simples", color='#f59e0b')
plt.plot(range(1, epocas + 1), historico_acc_avancada, label="CNN Avançada", color='#0ea5e9')
plt.title("Acurácia de Treino (Com Data Augmentation)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epocas + 1), historico_loss_simples, label="CNN Simples", color='#f59e0b')
plt.plot(range(1, epocas + 1), historico_loss_avancada, label="CNN Avançada", color='#0ea5e9')
plt.title("Função de Perda (Loss)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("comparacao_modelos.png")