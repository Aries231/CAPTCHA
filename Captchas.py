import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from CaptchasDataset import CAPTCHADataset
from PIL import Image
import string

class CaptchaNet(nn.Module):
    def __init__(self, num_chars=5, num_classes=36):
        super(CaptchaNet, self).__init__()
        self.num_chars=num_chars
        self.num_classes=num_classes

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc_out = nn.Linear(1680, self.num_chars * self.num_classes)

    def forward(self, x): # x.shape [N, 30, 60]
        x = self.pool1(F.relu(self.conv1(x))) # x.shape [N, 32, 15, 30]
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x))) # x.shape [N, 64, 7, 15]
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc_out(x)
        x = x.view(x.size(0), self.num_chars, self.num_classes)  # Reshape to (batch, num_chars, num_classes)
        return x


# Data loading and preprocessing
def load_data(data_path, is_train=True):
    # Define transformations
    transform_list = [
        transforms.ToTensor() # Convert to tensor
    ]
    
    if is_train:
        # Add data augmentation transformations only for the training set
        transform_list.extend([
            #transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)), # Random affine transformation
            #transforms.RandomCrop((30, 60), padding=4),  # Random cropping with padding
            #transforms.RandomRotation(10),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
            #transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),  # Adding Gaussian noise
        ])
    transform = transforms.Compose(transform_list)

    # Load the dataset
    dataset = CAPTCHADataset(root_dir=data_path, transform=transform)
    batch_size = 2 if is_train else 1 # Usually, no need to batch during validation/test
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return dataloader

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample["image"].to(device), sample["label"].to(device)
        optimizer.zero_grad()
        output = model(data)

        #loss = criterion(output, target)
        # Each output tensor corresponds to predictions for each character position
        loss = 0
        for i in range(output.shape[1]):  # Loop over character positions
            loss += criterion(output[:, i, :], target[:, i])

            _, predicted = torch.max(output[:, i, :], 1)
            correct += (predicted == target[:, i]).sum().item()
            total += target.size(0)
        
        loss.backward()
        optimizer.step()


        if batch_idx % 5 == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))
    
    accuracy = 100 * correct // total
    print(f'Training Accuracy: {accuracy}%')


def valid(model, device, valid_loader, optimizer, epoch):
    model.eval()
    correct = 0
    total = 0
    for batch_idx, sample in enumerate(valid_loader):
        data, target = sample["image"].to(device), sample["label"].to(device)
        output = model(data)

        for i in range(output.shape[1]):  # Loop over character positions
            _, predicted = torch.max(output[:, i, :], 1)
            correct += (predicted == target[:, i]).sum().item()
            total += target.size(0)
    
    accuracy = 100 * correct // total
    print(f'Validation Accuracy: {accuracy}%')
    return accuracy

class Captcha(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CaptchaNet().to(self.device)
        self.model.load_state_dict(torch.load("captcha_model.pth", map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.int_to_char = {idx: char for idx, char in enumerate(string.ascii_uppercase + string.digits)}

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        image = Image.open(im_path)
        image = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension and send to device
        output = self.model(image)
        predictions = torch.argmax(output, dim=2).squeeze(0)  # Remove batch dimension and take argmax
        captcha_text = ''.join([self.int_to_char[int(pred)] for pred in predictions])

        with open(save_path, 'w') as file:
            file.write(captcha_text)


def main():
    ########### Configuration ############ 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model, dataset, and dataloader setup
    model = CaptchaNet().to(device)
    train_loader = load_data('./sampleCaptchas_train')
    valid_loader = load_data('./sampleCaptchas_valid', is_train=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    ########### Training loop ############ 
    best_acc = 0
    for epoch in range(1, 50):
       train(model, device, train_loader, criterion, optimizer, epoch)
       
       valid_acc = valid(model, device, valid_loader, optimizer, epoch)
       
       if valid_acc > best_acc:
            print("Saving the model.")
            torch.save(model.state_dict(), "captcha_model.pth")
            best_acc = valid_acc

    ########### Inference ###########
    Inference = Captcha()
    Inference(im_path='./sampleCaptchas_test/input/input100.jpg', save_path='./sampleCaptchas_test/output/output100.txt')


if __name__ == "__main__":
    main()

   
