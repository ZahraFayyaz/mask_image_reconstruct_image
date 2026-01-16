import torch, json, urllib.request, os
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import distributed as dist

class CustomImageNetDataset(Dataset):
    def __init__(self, root, transform, class_to_imagenet_idx):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        
        # Overwrite class_to_idx with ImageNet indices
        self.class_to_idx = class_to_imagenet_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # Reverse mapping

        # Remap targets to ImageNet indices
        self.targets = [self.class_to_idx[self.dataset.classes[label]] for label in self.dataset.targets]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = self.targets[idx]  # Get correctly mapped label
        return image, label

class_index_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(class_index_url) as url:
    class_idx = json.loads(url.read().decode())
class_to_idx = {value[1] : int(key) for key, value in class_idx.items()}

transform = transforms.Compose([
    transforms.ToTensor(),
])
img_dir = '/home/abghamtm/work/masking_comparison/image/100class-vqvae-reconstruction'
img_labels = os.listdir(img_dir) 
custom_imagenet_idx = {cls_name : class_to_idx[cls_name] for cls_name in img_labels if cls_name in class_to_idx}
dataset = CustomImageNetDataset(root=img_dir, transform=transform, class_to_imagenet_idx=custom_imagenet_idx)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=12)

torch.cuda.set_device(1)  # Use GPU 1 (if desired)
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
classifier = resnet50(pretrained=False)
classifier.load_state_dict(torch.load('/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/weights_epoch30.pth'))
classifier.to(device)
classifier.eval()

correct = 0
total = 0
total_loss = 0
with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
        inputs = preprocess(inputs)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # loss = criterion(outputs, labels)
        # total_loss += loss.item()

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')
# print(f'loss: {total_loss:.2f}%')
# return accuracy, error_rate