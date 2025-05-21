import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import logging
from tqdm import tqdm

# Add YOLOv5 to path
YOLO_PATH = Path("yolov5")
sys.path.insert(0, str(YOLO_PATH.absolute()))

from models.common import DetectMultiBackend
from utils.datasets import LoadImagesAndLabels
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileNetSSD(nn.Module):
    """Lightweight SSD model with MobileNet backbone."""
    def __init__(self, num_classes=80):
        super().__init__()
        # MobileNet backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        
        # Remove the last fully connected layer
        self.backbone.classifier = nn.Identity()
        
        # SSD detection heads
        self.loc_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * 6, kernel_size=1)  # 4 coords * 6 anchors
        )
        
        self.conf_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes * 6, kernel_size=1)  # num_classes * 6 anchors
        )
        
    def forward(self, x):
        features = self.backbone.features(x)
        loc = self.loc_head(features)
        conf = self.conf_head(features)
        return loc, conf

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_outputs, teacher_outputs, targets):
        """
        Compute combined loss.
        
        Args:
            student_outputs: Tuple of (loc, conf) from student model
            teacher_outputs: Tuple of (loc, conf) from teacher model
            targets: Ground truth targets
        """
        student_loc, student_conf = student_outputs
        teacher_loc, teacher_conf = teacher_outputs
        
        # Hard loss (ground truth)
        hard_loss = self.compute_hard_loss(student_loc, student_conf, targets)
        
        # Soft loss (teacher knowledge)
        soft_loss = self.compute_soft_loss(student_conf, teacher_conf)
        
        # Combine losses
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return total_loss
    
    def compute_hard_loss(self, loc, conf, targets):
        """Compute loss using ground truth labels."""
        # Convert targets to the format expected by the loss
        # This is a simplified version - you'll need to adapt this to your specific format
        loc_loss = F.smooth_l1_loss(loc, targets[0])
        conf_loss = F.cross_entropy(conf, targets[1])
        return loc_loss + conf_loss
    
    def compute_soft_loss(self, student_conf, teacher_conf):
        """Compute KL divergence loss between student and teacher predictions."""
        # Apply temperature scaling
        student_logits = student_conf / self.temperature
        teacher_logits = teacher_conf / self.temperature
        
        # Compute softmax probabilities
        student_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence
        kl_loss = self.kl_div(student_probs, teacher_probs) * (self.temperature ** 2)
        return kl_loss

class KnowledgeDistillationTrainer:
    def __init__(
        self,
        teacher_model_path: str,
        data_yaml_path: str,
        img_size: int = 640,
        batch_size: int = 16,
        epochs: int = 100,
        device: str = None
    ):
        """
        Initialize the knowledge distillation trainer.
        
        Args:
            teacher_model_path: Path to the trained teacher model
            data_yaml_path: Path to the data configuration YAML file
            img_size: Input image size
            batch_size: Batch size for training
            epochs: Number of training epochs
            device: Device to use for training
        """
        self.device = select_device(device)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Load teacher model
        self.teacher = self._load_teacher_model(teacher_model_path)
        self.teacher.eval()  # Set teacher to evaluation mode
        
        # Initialize student model
        self.student = MobileNetSSD(num_classes=80).to(self.device)
        
        # Initialize loss function
        self.criterion = DistillationLoss(alpha=0.5, temperature=2.0)
        
        # Initialize optimizer and scheduler
        self.optimizer = Adam(self.student.parameters(), lr=0.001)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Create output directory
        self.output_dir = Path("runs/distill") / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_teacher_model(self, model_path):
        """Load the trained teacher model."""
        model = DetectMultiBackend(model_path, device=self.device)
        model.eval()
        return model
    
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
            stride=32,
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
        """Train the student model using knowledge distillation."""
        logger.info("Starting knowledge distillation...")
        
        # Create dataloaders
        train_loader = self._create_dataloader('train')
        val_loader = self._create_dataloader('val')
        
        # Training loop
        best_map = 0.0
        for epoch in range(self.epochs):
            # Training phase
            self.student.train()
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}')
            
            for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher(imgs)
                
                # Get student predictions
                student_outputs = self.student(imgs)
                
                # Compute loss
                loss = self.criterion(student_outputs, teacher_outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            
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
        logger.info("Knowledge distillation completed!")
    
    def validate(self, dataloader):
        """Validate the student model."""
        self.student.eval()
        stats = []
        
        with torch.no_grad():
            for imgs, targets, paths, shapes in tqdm(dataloader, desc='Validating'):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions
                loc, conf = self.student(imgs)
                preds = self.decode_predictions(loc, conf)
                preds = non_max_suppression(preds, 0.001, 0.65)
                
                # Calculate metrics
                for i, pred in enumerate(preds):
                    labels = targets[targets[:, 0] == i, 1:]
                    stats.append((pred, labels))
        
        # Calculate mAP
        metrics = ap_per_class(*zip(*stats))
        return metrics
    
    def decode_predictions(self, loc, conf):
        """Decode SSD predictions to YOLO format."""
        # This is a simplified version - you'll need to implement proper decoding
        # based on your anchor boxes and format requirements
        return torch.cat([loc, conf], dim=1)
    
    def save_model(self, filename):
        """Save the student model."""
        save_path = self.output_dir / filename
        torch.save({
            'model': self.student.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epochs
        }, save_path)
        logger.info(f"Model saved to {save_path}")

def main():
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        teacher_model_path='runs/train/exp_*/best.pt',  # Update with your teacher model path
        data_yaml_path='data/coco.yaml',
        img_size=640,
        batch_size=16,
        epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Start distillation
    trainer.train()

if __name__ == "__main__":
    main() 