import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DRDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_dir: Directory containing the images
            csv_path: Path to CSV file with id_code and diagnosis columns
            transform: Optional transforms to be applied to images
        """
        self.image_dir = image_dir
        self.transform = transform
        self.classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        
        # Validate paths
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load and validate CSV
        try:
            self.labels_df = pd.read_csv(csv_path)
            required_columns = {'id_code', 'diagnosis'}
            if not required_columns.issubset(self.labels_df.columns):
                raise ValueError(f"CSV missing required columns: {required_columns}")
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

        # Clean and validate data
        self.labels_df['id_code'] = self.labels_df['id_code'].astype(str).str.strip().str.rstrip('.png')
        self.labels_df['diagnosis'] = pd.to_numeric(self.labels_df['diagnosis'], errors='coerce')
        
        # Remove rows with invalid diagnoses
        valid_diagnoses = self.labels_df['diagnosis'].between(0, 4)
        self.labels_df = self.labels_df[valid_diagnoses].copy()
        
        # Build valid samples list
        self.valid_samples = []
        self.missing_files = []
        
        for idx, row in self.labels_df.iterrows():
            img_name = f"{row['id_code']}.png"
            img_path = os.path.join(self.image_dir, img_name)
            
            if os.path.exists(img_path):
                self.valid_samples.append(idx)
            else:
                self.missing_files.append(img_path)
        
        if not self.valid_samples:
            raise RuntimeError("No valid images found!")
        
        logger.info(f"Loaded {len(self.valid_samples)}/{len(self.labels_df)} valid images")
        if self.missing_files:
            logger.warning(f"{len(self.missing_files)} images missing. First 5: {self.missing_files[:5]}")

    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            tuple: (image_tensor, label) where label is an integer 0-4
        """
        try:
            row = self.labels_df.iloc[self.valid_samples[idx]]
            img_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            label = int(row['diagnosis'])
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return zero tensor with same shape as transformed images
            dummy_image = torch.zeros(3, 224, 224, dtype=torch.float32)
            return dummy_image, 0  # Default to class 0

    def get_class_distribution(self) -> pd.DataFrame:
        """Returns DataFrame with class distribution statistics"""
        return self.labels_df.iloc[self.valid_samples]['diagnosis'].value_counts().sort_index()

def get_transforms(train: bool = True) -> transforms.Compose:
    """Create transforms for training or validation
    
    Args:
        train: Whether to include augmentation transforms
    """
    base_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ]
        base_transforms = augmentations + base_transforms
    
    return transforms.Compose(base_transforms)


# Example usage
if __name__ == "__main__":
    try:
        # Initialize dataset
        transform = get_transforms(train=True)
        dataset = DRDataset(
            image_dir="train_images",
            csv_path="train_labels.csv",
            transform=transform
        )
        
        # Print stats
        print("\nDataset Statistics:")
        print(f"Total samples: {len(dataset)}")
        print("Class distribution:")
        print(dataset.get_class_distribution())
        
        # Test loading
        sample, label = dataset[0]
        print(f"\nSample shape: {sample.shape}")
        print(f"Sample label: {label} ({dataset.classes[label]})")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")

def get_transforms(train: bool = True):
    """Image transformations for both training and inference"""
    base = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ]
    
    if train:
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]
        base = augmentations + base
    
    return transforms.Compose(base)

def preprocess_image(image):
    """Basic preprocessing pipeline for single images"""
    return get_transforms(train=False)(image)