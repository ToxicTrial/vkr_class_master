import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import FingerprintPairsDataset
from model import SiameseNetwork
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((500, 500)),
    transforms.ToTensor()
])

name_path = os.path.join(os.getcwd(), "database", "fingerprints", "anton")
other_path = os.path.join(os.getcwd(), "database", "fingerprints", "others")
dataset = FingerprintPairsDataset(name_path, other_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SiameseNetwork().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    running_loss = 0.0

    for img1, img2, labels in dataloader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(img1, img2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/10 - Loss: {running_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'siamese_fingerprint_model.pth')

def compare_images(model, img_path1, img_path2):
    model.eval()
    img1 = Image.open(img_path1).convert('L')
    img2 = Image.open(img_path2).convert('L')

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img1, img2)
        score = output.item()
    return score
