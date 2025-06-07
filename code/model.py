import torch
import torch.nn as nn

class FingerprintEncoder(nn.Module):
    def __init__(self):
        import torchvision.models as models
        from torchvision.models import resnet18, ResNet18_Weights

        super(FingerprintEncoder, self).__init__()
        base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        base_model.fc = nn.Identity()
        self.encoder = base_model

    def forward(self, x):
        return self.encoder(x)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.encoder = FingerprintEncoder()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        diff = torch.abs(feat1 - feat2)
        out = self.fc(diff)
        return out

    def forward_once(self, x):
        """Метод для извлечения эмбеддингов из одного изображения"""
        return self.encoder(x)
