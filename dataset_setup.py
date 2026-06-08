import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

pasta_dados = './dataset'
os.makedirs(pasta_dados, exist_ok=True)

print("Iniciando download do EuroSAT via PyTorch...")
try:
    dataset = torchvision.datasets.EuroSAT(
        root=pasta_dados, 
        download=True, 
        transform=transforms.ToTensor() 
    )
    print(f"Sucesso! {len(dataset)} imagens baixadas e carregadas.")
except Exception as e:
    print(f"Erro ao baixar: {e}")


if 'dataset' in locals():
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

    indices = np.random.choice(len(dataset), 9, replace=False)

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_numpy = np.transpose(img.numpy(), (1, 2, 0))
        
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img_numpy)
        plt.title(classes[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()