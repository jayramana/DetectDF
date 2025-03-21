import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Using device: {device}")

def load_batches(split, data_dir="dataset", max_batches=5):
    all_images = []
    all_labels = []
    batch_counter = 0
    for file in os.listdir(data_dir):
        if file.startswith(f"{split}_batch_") and file.endswith(".pt"):
            print(f"[DEBUG] Loading {file}...")
            data = torch.load(os.path.join(data_dir, file))
            images, labels = data
            print(f"[DEBUG] {file} contains {images.size(0)} images.")
            all_images.append(images)
            all_labels.append(labels)
            batch_counter += 1
            if batch_counter >= max_batches:
                print(f"[DEBUG] Loaded {max_batches} batches, stopping for debugging.")
                break
    if not all_images:
        print(f"[ERROR] No {split} batch files found in {data_dir}!")
        return None, None
    images_tensor = torch.cat(all_images, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    print(f"[DEBUG] Total {split} images loaded: {images_tensor.size(0)}")
    return images_tensor, labels_tensor


# Load training and validation data
train_images, train_labels = load_batches("train")
val_images, val_labels = load_batches("validate")

if train_images is None or val_images is None:
    raise Exception("Training or validation data not found. Check your dataset folder!")

# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(train_images, train_labels)
val_dataset = TensorDataset(val_images, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a pre-trained model and modify the final layer for 2 classes (real, fake)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)
print("[DEBUG] Model loaded and modified for binary classification.")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5  # Set to a small number for debugging

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"[DEBUG] Epoch [{epoch+1}/{num_epochs}] starting...")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"[DEBUG] Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Validation step
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    print(f"[DEBUG] Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    
# Save the trained model
model_path = "model_final.pth"
torch.save(model.state_dict(), model_path)
print(f"[DEBUG] Model saved to {model_path}")
