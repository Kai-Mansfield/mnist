import os
import torch
import torch.nn as nn
import torch.optim as optim
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# 1. Define paths for ImageNet validation images and annotations
IMAGE_DIR = "/home/kajm20/mnist/ILSVRC/Data/CLS-LOC/val"  # Path to validation images
ANNOTATION_DIR = "/home/kajm20/mnist/ILSVRC/Annotations/CLS-LOC/val"  # Path to XML annotations

# 2. Define transformations for EfficientNet input (resize, crop, normalize)
imagenet_transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(224),  # Crop the image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean/std
])

# 3. Load the synset mapping
synset_mapping_path = "/home/kajm20/mnist/ILSVRC/LOC_synset_mapping.txt"
wordnet_to_imagenet = {}

# Load synset mapping from file
with open(synset_mapping_path) as f:
    for idx, line in enumerate(f.readlines()):
        wordnet_id, _ = line.split(' ', 1)  # Get WordNet ID from the line (skip class name)
        wordnet_to_imagenet[wordnet_id] = idx  # Map WordNet ID to class index

# 4. Define the custom dataset class
class ImageNetValDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        # Get all annotation file names
        self.annotation_files = sorted(os.listdir(annotation_dir))

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        # Get annotation file path
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
        
        # Parse XML to extract class label
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        wordnet_id = root.find("object").find("name").text  # WordNet ID, e.g., 'n01751748'

        # Use the synset mapping to convert WordNet ID to ImageNet class index
        class_idx = wordnet_to_imagenet.get(wordnet_id, -1)  # Default to -1 if not found (shouldn't happen)

        # Get image filename from XML and construct image path
        image_filename = root.find("filename").text + ".JPEG"
        image_path = os.path.join(self.image_dir, image_filename)

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, class_idx

# 5. Initialize the dataset and dataloader
imagenet_val_dataset = ImageNetValDataset(IMAGE_DIR, ANNOTATION_DIR, transform=imagenet_transform)
imagenet_val_loader = DataLoader(imagenet_val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 6. Define the model (EfficientNet-B0 with pre-trained weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(pretrained=True)  # Load pre-trained EfficientNet-B0 model
model.to(device)
model.eval()  # Set the model to evaluation mode

# 7. Define the evaluation function
def evaluate_model(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for inference
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get highest probability class
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)

    accuracy = (correct / total) * 100
    return accuracy

# 8. Evaluate the model on ImageNet validation set
accuracy = evaluate_model(model, imagenet_val_loader)
print(f"EfficientNet-B0 Top-1 Accuracy on ImageNet: {accuracy:.2f}%")
