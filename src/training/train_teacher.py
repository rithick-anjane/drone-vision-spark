import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from tqdm import tqdm

# Add the YOLOv5 repository to the path
YOLO_PATH = Path("yolov5")
if not YOLO_PATH.exists():
    os.system("git clone https://github.com/ultralytics/yolov5.git")
    os.system("cd yolov5 && pip install -r requirements.txt")

# Add YOLOv5 to Python path
sys.path.insert(0, str(YOLO_PATH.absolute()))

# Import YOLOv5 modules
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImagesAndLabels
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import ap_per_class
from yolov5.utils.torch_utils import select_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TeacherModelTrainer:
    def __init__(
        self,
        data_yaml_path: str,
        weights_path: str = "yolov5s.pt",
        img_size: int = 640,
        batch_size: int = 16,
        epochs: int = 100,
        device: str = None
    ):
        """
        Initialize the teacher model trainer.
        
        Args:
            data_yaml_path (str): Path to the data configuration YAML file
            weights_path (str): Path to pretrained weights
            img_size (int): Input image size
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            device (str): Device to use for training (cuda/cpu)
        """
        self.data_yaml_path = data_yaml_path
        self.weights_path = weights_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = select_device(device)
        
        # Load data configuration
        with open(data_yaml_path, 'r') as f:
            self.data_dict = yaml.safe_load(f)
        
        # Create output directory
        self.output_dir = Path("runs/train") / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and training components
        self._initialize_model()
        self._initialize_optimizer()
        
    def _initialize_model(self):
        """Initialize the YOLOv5 model with pretrained weights."""
        self.model = DetectMultiBackend(self.weights_path, device=self.device)
        self.model.train()
        
        # Configure model
        self.model.model[-1].nc = len(self.data_dict['names'])  # Update number of classes
        self.model.model[-1].anchors = self.model.model[-1].anchors * self.img_size / 640  # Scale anchors
        
        # Initialize loss function
        self.compute_loss = ComputeLoss(self.model.model)
        
    def _initialize_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0005)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
    def _create_dataloader(self, mode='train'):
        """Create DataLoader for training or validation."""
        dataset = LoadImagesAndLabels(
            self.data_dict[mode],
            img_size=self.img_size,
            batch_size=self.batch_size,
            augment=mode == 'train',
            hyp=self.data_dict.get('hyp', {}),
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=int(self.model.stride.max()),
            pad=0.5,
            prefix=f'{mode}: '
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=mode == 'train',
            num_workers=min(8, os.cpu_count()),
            pin_memory=True,
            collate_fn=LoadImagesAndLabels.collate_fn
        )
        
    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        
        # Create dataloaders
        train_loader = self._create_dataloader('train')
        val_loader = self._create_dataloader('val')
        
        # Training loop
        best_map = 0.0
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}')
            for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                preds = self.model(imgs)
                loss, loss_items = self.compute_loss(preds, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss_items.mean():.3f}'})
            
            # Validation phase
            metrics = self.validate(val_loader)
            map50 = metrics[0]  # mAP@0.5
            
            # Save best model
            if map50 > best_map:
                best_map = map50
                self.save_model('best.pt')
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{self.epochs} - mAP@0.5: {map50:.3f}")
            
        # Save final model
        self.save_model('last.pt')
        logger.info("Training completed!")
        
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        stats = []
        
        with torch.no_grad():
            for imgs, targets, paths, shapes in tqdm(dataloader, desc='Validating'):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Inference
                preds = self.model(imgs)
                preds = non_max_suppression(preds, 0.001, 0.65)
                
                # Calculate metrics
                for i, pred in enumerate(preds):
                    labels = targets[targets[:, 0] == i, 1:]
                    stats.append((pred, labels))
        
        # Calculate mAP
        metrics = ap_per_class(*zip(*stats))
        return metrics
        
    def save_model(self, filename):
        """Save the model."""
        save_path = self.output_dir / filename
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epochs,
            'data_dict': self.data_dict
        }, save_path)
        logger.info(f"Model saved to {save_path}")

def main():
    # Create data configuration YAML
    data_yaml = {
        'train': 'path/to/coco/train2017.txt',  # Update with your path
        'val': 'path/to/coco/val2017.txt',      # Update with your path
        'nc': 80,  # Number of classes in COCO
        'names': ['person', 'bicycle', 'car', ...]  # COCO class names
    }
    
    # Save data configuration
    data_yaml_path = 'data/coco.yaml'
    os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)
    
    # Initialize trainer
    trainer = TeacherModelTrainer(
        data_yaml_path=data_yaml_path,
        weights_path='yolov5s.pt',  # Start with small model
        img_size=640,
        batch_size=16,
        epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main() 