import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ---------- CONFIG ----------

TRAIN_IMAGE_DIR = "/its/home/kajm20/ILSVRC/Data/CLS-LOC/train"
TRAIN_ANNOTATION_DIR = "/its/home/kajm20/ILSVRC/Annotations/CLS-LOC/train"
SYNSET_PATH = "/its/home/kajm20/ILSVRC/LOC_synset_mapping.txt"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = min(4, os.cpu_count())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- SYNSET MAPPING ----------

wordnet_to_imagenet = {}
with open(SYNSET_PATH) as f:
    for idx, line in enumerate(f.readlines()):
        wordnet_id, _ = line.strip().split(' ', 1)
        wordnet_to_imagenet[wordnet_id] = idx

# ---------- TRANSFORMS ----------

imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------- DATASET ----------

class ImageNetTrainDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.annotation_paths = []

        for wnid in os.listdir(annotation_dir):
            wnid_dir = os.path.join(annotation_dir, wnid)
            if os.path.isdir(wnid_dir):
                for fname in os.listdir(wnid_dir):
                    self.annotation_paths.append(os.path.join(wnid_dir, fname))

    def __len__(self):
        return len(self.annotation_paths)

    def __getitem__(self, idx):
        annotation_path = self.annotation_paths[idx]
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        wordnet_id = root.find("object").find("name").text
        class_idx = wordnet_to_imagenet[wordnet_id]
        image_filename = root.find("filename").text + ".JPEG"
        image_path = os.path.join(TRAIN_IMAGE_DIR, wordnet_id, image_filename)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, class_idx

# ---------- DATALOADER ----------

train_dataset = ImageNetTrainDataset(TRAIN_IMAGE_DIR, TRAIN_ANNOTATION_DIR, transform=imagenet_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# ---------- MODEL ----------

model = models.efficientnet_b0(weights=None, num_classes=1000)
model = nn.DataParallel(model).to(DEVICE)

# ---------- OPTIMIZER & LOSS ----------

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ---------- RESUME FROM CHECKPOINT ----------

start_epoch = 0
if os.path.exists(CHECKPOINT_DIR):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR)
                   if f.startswith(f"efficientnet_bs{BATCH_SIZE}_epoch")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split("_epoch")[1].split(".")[0]))
        latest = checkpoints[-1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest)
        model.load_state_dict(torch.load(checkpoint_path))
        start_epoch = int(latest.split("_epoch")[1].split(".")[0])
        print(f"Resuming from checkpoint: {checkpoint_path} at epoch {start_epoch}")

# ---------- TRAINING ----------

def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} Training Loss: {running_loss / len(dataloader):.4f}")

def save_checkpoint(model, epoch, batch_size, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    # Save new checkpoint first
    new_ckpt_name = f"efficientnet_bs{batch_size}_epoch{epoch+1}.pt"
    new_ckpt_path = os.path.join(save_dir, new_ckpt_name)
    torch.save(model.state_dict(), new_ckpt_path)

    # Now remove all older checkpoints for this batch size
    for f in os.listdir(save_dir):
        if (
            f.startswith(f"efficientnet_bs{batch_size}_epoch")
            and f != new_ckpt_name
        ):
            os.remove(os.path.join(save_dir, f))

# ---------- MAIN LOOP ----------

for epoch in range(start_epoch, NUM_EPOCHS):
    train(model, train_loader, optimizer, criterion, epoch)
    save_checkpoint(model, epoch, BATCH_SIZE)

print("✅ Training completed successfully.")
