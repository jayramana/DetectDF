import os
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

# Define paths and parameters
DATA_DIR = "dataset/"
IMAGE_SIZE = 224        # Resize images to 224x224
BATCH_SIZE = 500       # Save a batch file after processing 1000 images per label

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_images(split):
    # Process images for a given split: "train", "test", or "validate"
    label_map = {"real": 0, "fake": 1}
    images_batch, labels_batch = [], []
    batch_count = 0
    
    split_path = os.path.join(DATA_DIR, split)
    
    # Iterate over each label folder ("real" and "fake")
    for label in ["real", "fake"]:
        folder = os.path.join(split_path, label)
        if not os.path.exists(folder):
            print(f"Folder {folder} not found. Skipping...")
            continue
        
        for file in tqdm(os.listdir(folder), desc=f"Processing {split}/{label} images"):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {img_path}, skipping...")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img)
            images_batch.append(img)
            labels_batch.append(label_map[label])
            
            # Save the batch if we've reached the threshold
            if len(images_batch) >= BATCH_SIZE:
                save_batch(split, batch_count, images_batch, labels_batch)
                batch_count += 1
                images_batch, labels_batch = [], []  # Clear the lists

    # Save any remaining images in the batch
    if images_batch:
        save_batch(split, batch_count, images_batch, labels_batch)

def save_batch(split, batch_index, images_list, labels_list):
    images_tensor = torch.stack(images_list)
    labels_tensor = torch.tensor(labels_list)
    batch_filename = f"{split}_batch_{batch_index}.pt"
    out_path = os.path.join(DATA_DIR, batch_filename)
    torch.save((images_tensor, labels_tensor), out_path)
    print(f"Saved {len(images_list)} images to {batch_filename}")

if __name__ == "__main__":
    for split in ["train", "test", "validate"]:
        process_images(split)
