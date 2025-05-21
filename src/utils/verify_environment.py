import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import yaml
import json
from typing import Dict, List, Tuple

def verify_cuda() -> bool:
    """Verify CUDA availability and version."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU only.")
        return False
    
    print(f"CUDA is available. Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    return True

def verify_dependencies() -> Dict[str, bool]:
    """Verify all required dependencies are installed."""
    dependencies = {
        'torch': True,
        'torchvision': True,
        'numpy': True,
        'pyyaml': True,
        'tqdm': True,
        'pycocotools': True,
        'onnx': True,
        'onnxruntime': True,
        'psutil': True,
        'GPUtil': True,
        'matplotlib': True,
        'pandas': True,
        'scikit-learn': True,
        'tensorboard': True
    }
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            dependencies[dep] = False
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
    else:
        print("All dependencies are installed.")
    
    return dependencies

def verify_dataset_structure(data_yaml_path: str) -> bool:
    """Verify dataset structure and annotations."""
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in data_config:
                print(f"Missing required key in data config: {key}")
                return False
        
        # Verify dataset paths
        for split in ['train', 'val']:
            path = Path(data_config[split])
            if not path.exists():
                print(f"Dataset path does not exist: {path}")
                return False
            
            # Check for annotations
            ann_file = path / 'annotations.json'
            if not ann_file.exists():
                print(f"Annotations file not found: {ann_file}")
                return False
        
        print("Dataset structure is valid.")
        return True
        
    except Exception as e:
        print(f"Error verifying dataset structure: {e}")
        return False

def verify_model_components() -> bool:
    """Verify all model components are properly implemented."""
    components = [
        'YOLOv5Trainer',
        'KnowledgeDistillationTrainer',
        'StructuredPruner',
        'ModelQuantizer',
        'ModelDeployer'
    ]
    
    missing = []
    for component in components:
        try:
            __import__(f'src.training.{component.lower()}')
        except ImportError:
            missing.append(component)
    
    if missing:
        print(f"Missing model components: {', '.join(missing)}")
        return False
    
    print("All model components are implemented.")
    return True

def verify_hardware_profile(profile: str) -> bool:
    """Verify hardware profile configuration."""
    valid_profiles = ['jetson_nano', 'raspberry_pi']
    if profile not in valid_profiles:
        print(f"Invalid hardware profile: {profile}")
        return False
    
    print(f"Hardware profile '{profile}' is valid.")
    return True

def main():
    """Run all verification checks."""
    print("Starting environment verification...")
    
    # Verify CUDA
    cuda_available = verify_cuda()
    
    # Verify dependencies
    dependencies = verify_dependencies()
    
    # Verify dataset structure
    data_yaml_path = 'data/coco.yaml'
    dataset_valid = verify_dataset_structure(data_yaml_path)
    
    # Verify model components
    components_valid = verify_model_components()
    
    # Verify hardware profile
    hardware_valid = verify_hardware_profile('jetson_nano')
    
    # Print summary
    print("\nVerification Summary:")
    print(f"CUDA Available: {cuda_available}")
    print(f"Dependencies Valid: {all(dependencies.values())}")
    print(f"Dataset Valid: {dataset_valid}")
    print(f"Components Valid: {components_valid}")
    print(f"Hardware Profile Valid: {hardware_valid}")
    
    # Check if all verifications passed
    all_valid = all([
        cuda_available,
        all(dependencies.values()),
        dataset_valid,
        components_valid,
        hardware_valid
    ])
    
    if all_valid:
        print("\nAll verifications passed! Environment is ready.")
    else:
        print("\nSome verifications failed. Please fix the issues above.")

if __name__ == "__main__":
    main() 