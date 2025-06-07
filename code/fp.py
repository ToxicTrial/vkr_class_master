import os
import torch.nn.functional as F
import torch
from PIL import Image

class Fingerprints:
    def __init__(self):
        from model import SiameseNetwork
        import torchvision.transforms as transforms

        self.transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((500, 500)), 
        transforms.ToTensor()
    ])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SiameseNetwork().to(self.device)
        self.model_path = os.path.join(os.getcwd(), "siamese_fingerprint_model.pth")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))


    def compare_images_cosine(self, img_path1, img_path2):
        self.model.eval()
        img1 = Image.open(img_path1).convert('L')
        img2 = Image.open(img_path2).convert('L')

        img1 = self.transform(img1).unsqueeze(0).to(self.device)
        img2 = self.transform(img2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb1 = self.model.forward_once(img1)
            emb2 = self.model.forward_once(img2)
            sim = F.cosine_similarity(emb1, emb2).item()

        return sim