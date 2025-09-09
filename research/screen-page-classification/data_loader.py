"""
Data loading utilities for screen page classification experiments.
Handles downloading labeled data from annotation API and preparing PyTorch datasets.
"""

import os
import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preparation."""
    annotation_api_url: str = "http://localhost:5000"
    data_root: str = "/workspace/data"
    output_dir: str = "/workspace/research/screen-page-classification/data"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


class AnnotationAPIClient:
    """Client for interacting with the annotation API."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects from the annotation API."""
        response = self.session.get(f"{self.base_url}/api/projects")
        response.raise_for_status()
        return response.json()
    
    def get_datasets(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all datasets for a project."""
        response = self.session.get(f"{self.base_url}/api/projects/{project_id}/datasets")
        response.raise_for_status()
        return response.json()
    
    def get_dataset_progress(self, dataset_id: int) -> Dict[str, Any]:
        """Get progress information for a dataset."""
        response = self.session.get(f"{self.base_url}/api/datasets/{dataset_id}/progress")
        response.raise_for_status()
        return response.json()
    
    def get_labeled_data(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Get all labeled data for a dataset."""
        response = self.session.get(f"{self.base_url}/api/datasets/{dataset_id}/labeled")
        response.raise_for_status()
        return response.json()
    
    def get_dataset_classes(self, dataset_id: int) -> List[Dict[str, Any]]:
        """Get class information for a dataset."""
        response = self.session.get(f"{self.base_url}/api/datasets/{dataset_id}/classes")
        response.raise_for_status()
        return response.json()


class ScreenPageDataset(Dataset):
    """PyTorch dataset for screen page classification."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        data_root: str,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data = data
        self.data_root = Path(data_root)
        self.transform = transform or self._get_default_transform()
        self.class_to_idx = class_to_idx or {}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """Validate that all required fields are present in the data."""
        required_fields = ['session_id', 'frame_id', 'frame_path_rel']
        for i, item in enumerate(self.data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field '{field}' in item {i}")
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.data[idx]
        
        # Load image
        image_path = self.data_root / item['frame_path_rel']
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        if 'single_label_class_id' in item and item['single_label_class_id'] is not None:
            label = item['single_label_class_id']
        elif 'class_name' in item:
            label = self.class_to_idx.get(item['class_name'], 0)
        else:
            raise ValueError(f"No valid label found for item {idx}")
        
        return image, label


class DatasetInspector:
    """Utility class for inspecting and analyzing datasets."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.api_client = AnnotationAPIClient(config.annotation_api_url)
    
    def inspect_available_data(self) -> Dict[str, Any]:
        """Inspect all available projects, datasets, and their progress."""
        logger.info("Inspecting available annotation data...")
        
        try:
            projects = self.api_client.get_projects()
            logger.info(f"Found {len(projects)} projects")
            
            all_datasets = []
            for project in projects:
                project_id = project['id']
                datasets = self.api_client.get_datasets(project_id)
                logger.info(f"Project '{project['name']}' has {len(datasets)} datasets")
                
                for dataset in datasets:
                    dataset_id = dataset['id']
                    progress = self.api_client.get_dataset_progress(dataset_id)
                    classes = self.api_client.get_dataset_classes(dataset_id)
                    
                    dataset_info = {
                        'project_id': project_id,
                        'project_name': project['name'],
                        'dataset_id': dataset_id,
                        'dataset_name': dataset['name'],
                        'target_type': dataset.get('target_type_name', 'Unknown'),
                        'progress': progress,
                        'classes': classes,
                        'num_classes': len(classes)
                    }
                    all_datasets.append(dataset_info)
            
            return {
                'projects': projects,
                'datasets': all_datasets,
                'total_projects': len(projects),
                'total_datasets': len(all_datasets)
            }
            
        except Exception as e:
            logger.error(f"Error inspecting data: {e}")
            raise
    
    def analyze_class_balance(self, dataset_id: int) -> Dict[str, Any]:
        """Analyze class balance for a specific dataset."""
        logger.info(f"Analyzing class balance for dataset {dataset_id}")
        
        try:
            # Get labeled data
            labeled_data = self.api_client.get_labeled_data(dataset_id)
            classes = self.api_client.get_dataset_classes(dataset_id)
            
            if not labeled_data:
                logger.warning(f"No labeled data found for dataset {dataset_id}")
                return {'error': 'No labeled data found'}
            
            # Count samples per class
            class_counts = {}
            class_names = {cls['id']: cls['name'] for cls in classes}
            
            for item in labeled_data:
                if 'single_label_class_id' in item and item['single_label_class_id'] is not None:
                    class_id = item['single_label_class_id']
                    class_name = class_names.get(class_id, f'Unknown_{class_id}')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Calculate statistics
            total_samples = sum(class_counts.values())
            class_balances = {name: count / total_samples for name, count in class_counts.items()}
            
            # Find most/least balanced classes
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            most_common = sorted_classes[0] if sorted_classes else None
            least_common = sorted_classes[-1] if sorted_classes else None
            
            analysis = {
                'dataset_id': dataset_id,
                'total_samples': total_samples,
                'num_classes': len(class_counts),
                'class_counts': class_counts,
                'class_balances': class_balances,
                'most_common_class': most_common,
                'least_common_class': least_common,
                'imbalance_ratio': most_common[1] / least_common[1] if most_common and least_common and least_common[1] > 0 else float('inf')
            }
            
            logger.info(f"Class balance analysis complete:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Number of classes: {len(class_counts)}")
            logger.info(f"  Most common: {most_common[0]} ({most_common[1]} samples)" if most_common else "N/A")
            logger.info(f"  Least common: {least_common[0]} ({least_common[1]} samples)" if least_common else "N/A")
            logger.info(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing class balance: {e}")
            raise
    
    def download_dataset(self, dataset_id: int, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Download and prepare a dataset for training."""
        output_dir = output_dir or self.config.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading dataset {dataset_id} to {output_path}")
        
        try:
            # Get labeled data
            labeled_data = self.api_client.get_labeled_data(dataset_id)
            classes = self.api_client.get_dataset_classes(dataset_id)
            
            if not labeled_data:
                raise ValueError(f"No labeled data found for dataset {dataset_id}")
            
            # Create class mapping
            class_to_idx = {cls['name']: cls['id'] for cls in classes}
            idx_to_class = {cls['id']: cls['name'] for cls in classes}
            
            # Save metadata
            metadata = {
                'dataset_id': dataset_id,
                'num_samples': len(labeled_data),
                'num_classes': len(classes),
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class,
                'classes': classes
            }
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save labeled data
            with open(output_path / 'labeled_data.json', 'w') as f:
                json.dump(labeled_data, f, indent=2)
            
            # Create DataFrame for easy analysis
            df_data = []
            for item in labeled_data:
                if 'single_label_class_id' in item and item['single_label_class_id'] is not None:
                    df_data.append({
                        'session_id': item['session_id'],
                        'frame_id': item['frame_id'],
                        'frame_path_rel': item['frame_path_rel'],
                        'class_id': item['single_label_class_id'],
                        'class_name': idx_to_class.get(item['single_label_class_id'], 'Unknown')
                    })
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path / 'labeled_data.csv', index=False)
            
            logger.info(f"Dataset downloaded successfully:")
            logger.info(f"  Samples: {len(labeled_data)}")
            logger.info(f"  Classes: {len(classes)}")
            logger.info(f"  Output directory: {output_path}")
            
            return {
                'output_dir': str(output_path),
                'metadata': metadata,
                'dataframe': df
            }
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise


def create_data_loaders(
    data: List[Dict[str, Any]],
    data_root: str,
    class_to_idx: Dict[str, int],
    config: DatasetConfig,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Split data
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=config.random_state, stratify=[item.get('single_label_class_id', 0) for item in data]
    )
    train_data, val_data = train_test_split(
        train_data, test_size=val_size/(1-test_size), random_state=config.random_state, stratify=[item.get('single_label_class_id', 0) for item in train_data]
    )
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ScreenPageDataset(train_data, data_root, train_transform, class_to_idx)
    val_dataset = ScreenPageDataset(val_data, data_root, val_test_transform, class_to_idx)
    test_dataset = ScreenPageDataset(test_data, data_root, val_test_transform, class_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    config = DatasetConfig()
    inspector = DatasetInspector(config)
    
    # Inspect available data
    data_info = inspector.inspect_available_data()
    print(f"Found {data_info['total_datasets']} datasets across {data_info['total_projects']} projects")
    
    # If datasets are available, analyze the first one
    if data_info['datasets']:
        dataset = data_info['datasets'][0]
        print(f"\nAnalyzing dataset: {dataset['dataset_name']}")
        
        # Analyze class balance
        balance_analysis = inspector.analyze_class_balance(dataset['dataset_id'])
        
        # Download dataset
        download_result = inspector.download_dataset(dataset['dataset_id'])
        print(f"Dataset downloaded to: {download_result['output_dir']}")