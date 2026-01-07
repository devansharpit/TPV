
"""
Efficient PyTorch DataLoader for ImageNet dataset.
This implementation loads images on-demand and doesn't require the entire dataset to be in memory.
"""

import os
import csv
import json
from typing import Tuple, Dict, List, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


def load_imagenet_class_mapping(json_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load official ImageNet class mapping from JSON file.
    
    Args:
        json_path: Path to imagenet_class_index.json file
        
    Returns:
        Tuple of (class_to_idx, idx_to_class) dictionaries
    """
    with open(json_path, 'r') as f:
        class_index = json.load(f)
    
    # Convert to proper mappings
    class_to_idx = {}
    idx_to_class = {}
    
    for idx_str, (class_id, class_name) in class_index.items():
        idx = int(idx_str)
        class_to_idx[class_id] = idx
        idx_to_class[idx] = class_id
    
    return class_to_idx, idx_to_class


class ImageNetTrainDataset(Dataset):
    """
    ImageNet Training Dataset
    Assumes training images are organized in subfolders by class ID.
    Uses official ImageNet class indexing for compatibility with pretrained models.
    """
    
    def __init__(self, 
                 root_dir: str,
                 class_mapping_json: str,
                 transform: Optional[Callable] = None):
        """
        Args:
            root_dir: Path to the training directory containing class subfolders
            class_mapping_json: Path to imagenet_class_index.json file
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load official ImageNet class mapping
        self.class_to_idx, self.idx_to_class = load_imagenet_class_mapping(class_mapping_json)
        
        # Get all class folders that exist in the dataset
        available_class_folders = [d for d in os.listdir(root_dir) 
                                 if os.path.isdir(os.path.join(root_dir, d))]
        
        # Filter to only classes that exist in our official mapping
        self.class_folders = [cls for cls in available_class_folders if cls in self.class_to_idx]
        
        if len(self.class_folders) != len(available_class_folders):
            missing_classes = set(available_class_folders) - set(self.class_folders)
            print(f"Warning: {len(missing_classes)} class folders not found in official mapping: {list(missing_classes)[:5]}...")
        
        print(f"Using {len(self.class_folders)} classes from official ImageNet mapping")
        
        # Build list of all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        print("Building ImageNet training dataset index...")
        for class_name in self.class_folders:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files in this class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
        
        print(f"Found {len(self.image_paths)} training images across {len(self.class_folders)} classes")
        print(f"Class indices range from {min(self.labels)} to {max(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """Get class name from class index"""
        return self.idx_to_class[idx]
    
    def get_num_classes(self) -> int:
        """Get total number of classes"""
        return len(self.class_folders)


class ImageNetValDataset(Dataset):
    """
    ImageNet Validation Dataset
    Uses CSV file to map image filenames to class IDs.
    Uses official ImageNet class indexing for compatibility with pretrained models.
    """
    
    def __init__(self, 
                 root_dir: str,
                 csv_file: str,
                 class_mapping_json: str,
                 transform: Optional[Callable] = None):
        """
        Args:
            root_dir: Path to validation images directory
            csv_file: Path to CSV file mapping image names to class IDs
            class_mapping_json: Path to imagenet_class_index.json file
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load official ImageNet class mapping
        self.class_to_idx, self.idx_to_class = load_imagenet_class_mapping(class_mapping_json)
        
        # Parse CSV file to build image to class mapping
        self.image_paths = []
        self.labels = []
        
        print("Building ImageNet validation dataset index...")
        
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                if len(row) >= 2:
                    image_id = row[0]
                    prediction_string = row[1].strip()
                    
                    # Extract class ID (first part before space)
                    class_id = prediction_string.split()[0]
                    
                    # Check if class exists in our mapping
                    if class_id in self.class_to_idx:
                        img_path = os.path.join(root_dir, f"{image_id}.JPEG")
                        
                        # Verify image file exists
                        if os.path.exists(img_path):
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_id])
        
        print(f"Found {len(self.image_paths)} validation images")
        if self.labels:
            print(f"Validation class indices range from {min(self.labels)} to {max(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_imagenet_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Get standard ImageNet transforms for training or validation.
    
    Args:
        is_training: If True, applies training transforms (augmentation), 
                    else applies validation transforms
    
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_imagenet_dataloaders(
                              train_batch_size: int = 32,
                              val_batch_size: Optional[int] = 32,
                              use_augmentation: bool = True,
                              num_workers: int = 4,
                              pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Create ImageNet training and validation data loaders with official class mapping.
    
    Args:
        train_batch_size: Batch size for training data loader
        val_batch_size: Batch size for validation data loader
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, class_to_idx_mapping)
    """
    
    train_dir = "/home/darpit/Desktop/projects/datasets/imagenet/train"
    val_dir = "/home/darpit/Desktop/projects/datasets/imagenet/val/imgs"
    val_csv = "/home/darpit/Desktop/projects/datasets/imagenet/LOC_val_solution.csv"
    class_mapping_json = "/home/darpit/Desktop/projects/datasets/imagenet/imagenet_class_index.json"

    # Create datasets
    train_transform = get_imagenet_transforms(is_training=use_augmentation)
    val_transform = get_imagenet_transforms(is_training=False)
    
    print("Creating training dataset...")
    train_dataset = ImageNetTrainDataset(train_dir, class_mapping_json, transform=train_transform)
    
    print("Creating validation dataset...")
    val_dataset = ImageNetValDataset(val_dir, val_csv, class_mapping_json, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print(f"Training loader: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"Validation loader: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    
    return train_loader, val_loader, train_dataset.class_to_idx


# Example usage
if __name__ == "__main__":
    # Define paths
    TRAIN_DIR = "/home/darpit/Desktop/projects/datasets/imagenet/train"
    VAL_DIR = "/home/darpit/Desktop/projects/datasets/imagenet/val/imgs"
    VAL_CSV = "/home/darpit/Desktop/projects/datasets/imagenet/LOC_val_solution.csv"
    CLASS_MAPPING_JSON = "/home/darpit/Desktop/projects/datasets/imagenet/imagenet_class_index.json"
    
    # Create data loaders
    train_loader, val_loader, class_to_idx = create_imagenet_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        val_csv=VAL_CSV,
        class_mapping_json=CLASS_MAPPING_JSON,
        batch_size=32,
        val_batch_size=32,
        num_workers=4,
        pin_memory=True
    )
    
    # Test loading a batch
    print("\nTesting data loading...")
    
    # Test training loader
    train_batch = next(iter(train_loader))
    print(f"Training batch shape: {train_batch[0].shape}, Labels shape: {train_batch[1].shape}")
    print(f"Training batch label range: {train_batch[1].min().item()} - {train_batch[1].max().item()}")
    
    # Test validation loader
    val_batch = next(iter(val_loader))
    print(f"Validation batch shape: {val_batch[0].shape}, Labels shape: {val_batch[1].shape}")
    print(f"Validation batch label range: {val_batch[1].min().item()} - {val_batch[1].max().item()}")
    
    print(f"\nExample class mappings:")
    for i, (class_name, class_idx) in enumerate(list(class_to_idx.items())[:5]):
        print(f"  {class_name}: {class_idx}")

# def create_imagenet_dataloaders(train_batch_size=512, val_batch_size=512):
#     # should return train_loader, val_loader with given batch sizes
#     # make sure no data augmentation is used in both loaders
#     raise NotImplementedError("Please implement data loading for ImageNet dataset.")