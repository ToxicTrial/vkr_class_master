import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FingerprintPairsDataset(Dataset):
    def __init__(self, mine_dir, others_dir, transform=None):
        self.mine_images = [os.path.join(mine_dir, f) for f in os.listdir(mine_dir)]
        self.others_images = [os.path.join(others_dir, f) for f in os.listdir(others_dir)]
        self.transform = transform

        self.positive_pairs = []
        for i in range(len(self.mine_images)):
            for j in range(i + 1, len(self.mine_images)):
                self.positive_pairs.append((self.mine_images[i], self.mine_images[j]))

        self.negative_pairs = []
        for mine_img in self.mine_images:
            other_img = random.choice(self.others_images)
            self.negative_pairs.append((mine_img, other_img))

        self.pairs = self.positive_pairs + self.negative_pairs
        self.labels = [1] * len(self.positive_pairs) + [0] * len(self.negative_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
