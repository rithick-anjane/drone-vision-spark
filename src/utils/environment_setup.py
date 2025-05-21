import os
import sys
import subprocess
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the Python environment and verify installations."""
    try:
        # Verify CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. Using CPU only.")

        # Verify PyTorch installation
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Torchvision version: {torchvision.__version__}")

        # Verify TensorFlow installation
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")

        return True
    except Exception as e:
        logger.error(f"Error during environment setup: {str(e)}")
        return False

class DatasetLoader:
    """Utility class for loading and preprocessing common object detection datasets."""
    
    def __init__(self, dataset_path: str, dataset_type: str = "coco"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path (str): Path to the dataset
            dataset_type (str): Type of dataset ("coco" or "voc")
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_type = dataset_type.lower()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        if self.dataset_type == "coco":
            self._setup_coco()
        elif self.dataset_type == "voc":
            self._setup_voc()
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def _setup_coco(self):
        """Set up COCO dataset."""
        self.coco = COCO(self.dataset_path / "annotations" / "instances_train2017.json")
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_names = [cat['name'] for cat in self.categories]

    def _setup_voc(self):
        """Set up Pascal VOC dataset."""
        # Implement VOC dataset setup
        pass

    def load_image(self, image_id: int):
        """
        Load and preprocess an image from the dataset.
        
        Args:
            image_id (int): ID of the image to load
            
        Returns:
            tuple: (image tensor, annotations)
        """
        if self.dataset_type == "coco":
            img_info = self.coco.loadImgs(image_id)[0]
            image = Image.open(self.dataset_path / "train2017" / img_info['file_name'])
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ann_ids)
            
            # Convert image to tensor
            image_tensor = self.transform(image)
            
            return image_tensor, annotations
        else:
            raise NotImplementedError(f"Image loading not implemented for {self.dataset_type}")

    def get_dataset_stats(self):
        """Get basic statistics about the dataset."""
        if self.dataset_type == "coco":
            return {
                "num_images": len(self.coco.getImgIds()),
                "num_categories": len(self.categories),
                "categories": self.category_names
            }
        else:
            raise NotImplementedError(f"Dataset stats not implemented for {self.dataset_type}")

def main():
    """Main function to set up the environment and verify installations."""
    if setup_environment():
        logger.info("Environment setup completed successfully!")
    else:
        logger.error("Environment setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 