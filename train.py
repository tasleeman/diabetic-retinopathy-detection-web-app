import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import DRDataset, get_transforms
from model import DRModel
import pandas as pd
import os
from torchinfo import summary  # pip install torchinfo

def train_model():
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize datasets
    full_dataset = DRDataset(
        image_dir="train_images",
        csv_path="train_labels.csv",
        transform=get_transforms(train=True)
    )
    
    # Dataset verification
    print("\nDataset verification:")
    print(f"Total samples: {len(full_dataset)}")
    print("Class distribution:")
    print(full_dataset.get_class_distribution())
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Verify first sample
    sample_img, sample_label = train_dataset[0]
    print(f"\nSample shape: {sample_img.shape}")
    print(f"Sample label: {sample_label} ({full_dataset.classes[sample_label]})")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    model = DRModel(num_classes=5).to(device)
    
    # Model summary
    print("\nModel architecture:")
    summary(model, input_size=(BATCH_SIZE, 3, 224, 224))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    # Training loop
    best_val_accuracy = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
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
        scheduler.step(val_accuracy)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {running_loss/len(train_loader):.4f}")
        print(f"Val Accuracy: {val_accuracy:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "dr_model.pth")
            print(f"Saved new best model with val accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()