import os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Using device: {device}")

def load_batches(split, data_dir="dataset"):
    all_images = []
    all_labels = []
    for file in os.listdir(data_dir):
        if file.startswith(f"{split}_batch_") and file.endswith(".pt"):
            print(f"[DEBUG] Loading {file}...")
            data = torch.load(os.path.join(data_dir, file))
            images, labels = data
            print(f"[DEBUG] {file} contains {images.size(0)} images.")
            all_images.append(images)
            all_labels.append(labels)
    if not all_images:
        print(f"[ERROR] No {split} batch files found in {data_dir}!")
        return None, None
    images_tensor = torch.cat(all_images, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    print(f"[DEBUG] Total {split} images loaded: {images_tensor.size(0)}")
    return images_tensor, labels_tensor

# Load test data
test_images, test_labels = load_batches("test")
if test_images is None:
    raise Exception("Test data not found. Check your dataset folder!")

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# Define the model (must match the architecture used in training)
model = models.resnet18(pretrained=False)  # We'll load the saved weights next
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)
print("[DEBUG] Model architecture for testing is set.")

# Load the saved model weights
model_path = "model_final.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[DEBUG] Loaded model weights from {model_path}")
else:
    raise Exception(f"Model file {model_path} not found!")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing")):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        batch_acc = (predicted == labels).float().mean().item() * 100
        print(f"[DEBUG] Processed batch {batch_idx+1}/{len(test_loader)} - Batch accuracy: {batch_acc:.2f}%")

test_accuracy = 100 * correct / total
print(f"[DEBUG] Final Test Accuracy: {test_accuracy:.2f}%")
