import torch
import torch.nn as nn

class CNN_Simples(nn.Module):
    def __init__(self):
        super(CNN_Simples, self).__init__()
        
        self.camada_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.camada_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.camada_decisao = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(32 * 16 * 16, 128), 
            nn.ReLU(),
            nn.Linear(128, 10) 
        )

    def forward(self, x):
        x = self.camada_conv1(x)
        x = self.camada_conv2(x)
        x = self.camada_decisao(x)
        return x


if __name__ == "__main__":
    modelo = CNN_Simples()
    print(modelo)
    print("\n✅ Sucesso! A arquitetura da sua primeira CNN foi criada do zero.")    


class CNN_Avancada(nn.Module):
    def __init__(self):
        super(CNN_Avancada, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decisao = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.decisao(x)
        return x