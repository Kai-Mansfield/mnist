import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ---------- CONFIG ----------

VAL_IMAGE_DIR = "/mnt/lustre/users/inf/kajm20/ILSVRC/Data/CLS-LOC/val"
VAL_ANNOTATION_DIR = "/mnt/lustre/users/inf/kajm20/ILSVRC/Annotations/CLS-LOC/val"
SYNSET_PATH = "/mnt/lustre/users/inf/kajm20/ILSVRC/LOC_synset_mapping.txt"
CHECKPOINT_PATH = "checkpoints/efficientnet_bs64_epoch20.pt"
BATCH_SIZE = 64
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- SYNSET MAPPING ----------

wordnet_to_imagenet = {}
with open(SYNSET_PATH) as f:
    for idx, line in enumerate(f.readlines()):
        wordnet_id, _ = line.strip().split(' ', 1)
        wordnet_to_imagenet[wordnet_id] = idx

# ---------- TRANSFORMS ----------

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- DATASET ----------

class ImageNetValDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.samples = []
        self.transform = transform
        for fname in os.listdir(annotation_dir):
            if fname.endswith(".xml"):
                ann_path = os.path.join(annotation_dir, fname)
                tree = ET.parse(ann_path)
                root = tree.getroot()
                wordnet_id = root.find("object").find("name").text
                class_idx = wordnet_to_imagenet[wordnet_id]
                img_fname = root.find("filename").text + ".JPEG"
                img_path = os.path.join(image_dir, img_fname)
                self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------- LOAD MODEL ----------

model = models.efficientnet_b0(weights=None, num_classes=1000)
model = nn.DataParallel(model).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

# ---------- DATALOADER ----------

val_dataset = ImageNetValDataset(VAL_IMAGE_DIR, VAL_ANNOTATION_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# ---------- EVALUATION ----------

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validating"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"✅ Validation Accuracy: {accuracy:.2f}%")
