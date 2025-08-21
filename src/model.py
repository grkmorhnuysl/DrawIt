import torch, torch.nn as nn
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
    # Eğer veri 784 boyutlu düz vektör geldiyse 28x28'e çevir
        if x.ndim == 3:  # (batch, 1, 784)
            x = x.view(x.size(0), 1, 28, 28)
        elif x.ndim == 4 and x.shape[2] == 1 and x.shape[3] == 784:
            x = x.view(x.size(0), 1, 28, 28)
        return self.classifier(self.features(x))
