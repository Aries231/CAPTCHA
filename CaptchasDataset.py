import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import string

# Custom Dataset class
class CAPTCHADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [file for file in os.listdir(root_dir+"/input") if file.endswith('.jpg')]
        # Mapping characters to integers
        self.char_to_int = {char: idx for idx, char in enumerate(string.ascii_uppercase + string.digits)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir+"/input", self.images[idx])
        image = Image.open(img_name)
        sample_idx = self.images[idx][5:7]

        label_name = os.path.join(self.root_dir+"/output", "output"+sample_idx+".txt")
        with open(label_name, 'r') as file:
            label_str = file.readline().strip()
        
        # Convert label string to a list of integers
        label = [self.char_to_int[char] for char in label_str]
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample