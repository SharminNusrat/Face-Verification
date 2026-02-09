import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
import os
import copy
import time
import numpy as np

def train_model():
    # =========================
    # Config
    # =========================
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15
    # Updated path based on user instruction
    DATA_DIR = "/kaggle/input/data-cleaning-glasses-no-glasses/Images/Images" 
    MODEL_SAVE_PATH = "glasses_detection.pth"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VALIDATION_SPLIT = 0.2
    SEED = 42
    
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")

    # =========================
    # Data Transforms
    # =========================
    # Augmentation for training
    train_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # No augmentation for validation
    val_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # =========================
    # Dataset Loading & Splitting
    # =========================
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory {DATA_DIR} not found.")
        return

    # Load full dataset twice to apply different transforms later (via Subsets)
    # Note: We will use the same indices for split.
    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}") 
    # Expectation: ['glasses', 'no_glasses'] or similar. 
    # Make sure 'glasses' is Class 0 or 1. ImageFolder sorts alphabetically.
    # If ['glasses', 'no_glasses']: glasses=0, no_glasses=1.
    
    dataset_size = len(full_dataset)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    
    # Generate split indices with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Apply transforms
    # random_split returns Subset objects. Subset doesn't have 'transform' attribute directly usually,
    # it relies on parent. But we want different transforms.
    # Workaround: A custom wrapper class.
    
    class TransformedSubset(Subset):
        def __init__(self, subset, transform=None):
            super().__init__(subset.dataset, subset.indices)
            self.transform = transform

        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            if self.transform:
                x = self.transform(x) # ImageFolder returns tensor if transformed, or PIL if not.
                # Wait, ImageFolder applies transform upon loading. 
                # If we passed no transform to ImageFolder, it returns PIL.
                pass
            return x, y
            
    # Better approach for PyTorch standard:
    # 1. Load ImageFolder WITHOUT transforms (returns PIL images)
    # 2. Wrap in a custom Dataset that applies transform
    
    class CustomImageFolder(datasets.ImageFolder):
        def __init__(self, root, transform=None):
            super().__init__(root, transform=None) # Don't pass transform to parent
            self.real_transform = transform
            
        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.real_transform is not None:
                sample = self.real_transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target

    # Re-load with our logic to ensure correct splits + transforms
    full_dataset_no_transform = datasets.ImageFolder(DATA_DIR) # Returns PIL
    
    # Split indices
    indices = list(range(len(full_dataset_no_transform)))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    split = int(np.floor(VALIDATION_SPLIT * len(full_dataset_no_transform)))
    train_idx, val_idx = indices[split:], indices[:split]
    
    # Create Subsets with transforms
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            # Get raw item (PIL image)
            # ImageFolder.__getitem__ applies transform if set. We initialized full_dataset_no_transform without one.
            # So it returns (sample, target) where sample is PIL.
            image, label = self.dataset[real_idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    train_data = TransformedDataset(full_dataset_no_transform, train_idx, train_transforms)
    val_data = TransformedDataset(full_dataset_no_transform, val_idx, val_transforms)

    dataloaders = {
        'training': DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'validation': DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    
    dataset_sizes = {'training': len(train_data), 'validation': len(val_data)}
    print(f"Dataset sizes: {dataset_sizes}")

    # =========================
    # Model (EfficientNet-B0)
    # =========================
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Modify classifier for binary classification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) 
    
    model = model.to(DEVICE)

    # =========================
    # Loss & Optimizer
    # =========================
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =========================
    # Training Loop
    # =========================
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
        print('-' * 10)

        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1) # Reshape for BCE loss

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs) > 0.5 

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # =========================
    # Save Model
    # =========================
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Print class mapping for reference
    print(f"Class Mapping: {full_dataset.class_to_idx}")
    # Example: {'glasses': 0, 'no_glasses': 1}
    # If so, 0=glasses, 1=no_glasses.
    # In GlassDetector inference, we need to match this.
    # Sigmoid > 0.5 means class 1. 
    # If class 1 is no_glasses, then >0.5 is no_glasses.

if __name__ == "__main__":
    train_model()
