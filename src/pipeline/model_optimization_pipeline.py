import os
import sys
from pathlib import Path
import logging
import json
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import time
from datetime import datetime
import shutil

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import required packages with error handling
try:
    import torch
    import torch.nn as nn
    import torch.utils.data as data
    from torchvision import transforms
    import numpy as np
    import yaml
    from tqdm import tqdm
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    logger.error("Please install required packages using: pip install -r requirements.txt")
    sys.exit(1)

# Import optimization modules with error handling
try:
    from src.training.train_teacher import YOLOv5Trainer
    from src.training.knowledge_distillation import KnowledgeDistillationTrainer
    from src.optimization.model_pruning import StructuredPruner
    from src.optimization.model_quantization import ModelQuantizer
    from src.deployment.model_deployment import ModelDeployer
except ImportError as e:
    logger.error(f"Failed to import optimization modules: {e}")
    logger.error("Please ensure all required modules are in the correct directories")
    sys.exit(1)

@dataclass
class PipelineConfig:
    """Configuration for the optimization pipeline."""
    # Data configuration
    data_yaml_path: str
    num_classes: int
    img_size: int
    batch_size: int
    
    # Training configuration
    teacher_epochs: int
    distillation_epochs: int
    learning_rate: float
    
    # Pruning configuration
    pruning_iterations: int
    target_sparsity: float
    fine_tune_epochs: int
    
    # Quantization configuration
    calibration_samples: int
    
    # Deployment configuration
    hardware_profile: str
    input_shape: tuple
    
    # General configuration
    output_dir: str
    checkpoint_dir: str
    device: str
    num_workers: int
    seed: int

class OptimizationPipeline:
    def __init__(self, config: PipelineConfig):
        """
        Initialize the optimization pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize state
        self.state = self._load_state()
        
        # Set device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._init_components()
        
        # Initialize data transforms
        self.transforms = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_dataloader(self, mode: str) -> data.DataLoader:
        """Create DataLoader for the specified mode."""
        try:
            # Load dataset configuration
            with open(self.config.data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Get dataset paths
            if mode == 'train':
                dataset_path = data_config['train']
            elif mode == 'val':
                dataset_path = data_config['val']
            elif mode == 'calibration':
                # Use a subset of training data for calibration
                dataset_path = data_config['train']
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Create dataset
            dataset = self._create_dataset(dataset_path, mode)
            
            # Create sampler for calibration mode
            if mode == 'calibration':
                indices = np.random.choice(
                    len(dataset),
                    min(self.config.calibration_samples, len(dataset)),
                    replace=False
                )
                sampler = data.SubsetRandomSampler(indices)
            else:
                sampler = None
            
            # Create DataLoader
            return data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(mode == 'train' and sampler is None),
                num_workers=self.config.num_workers,
                sampler=sampler,
                pin_memory=True
            )
            
        except Exception as e:
            logger.error(f"Error creating DataLoader for {mode}: {e}")
            raise

    def _create_dataset(self, dataset_path: str, mode: str) -> data.Dataset:
        """Create dataset for the specified path and mode."""
        try:
            # This should be implemented based on your specific dataset
            # For example, using COCO dataset:
            from pycocotools.coco import COCO
            from torchvision.datasets import CocoDetection
            
            return CocoDetection(
                root=dataset_path,
                annFile=os.path.join(dataset_path, 'annotations.json'),
                transform=self.transforms
            )
            
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise

    def _init_components(self):
        """Initialize pipeline components."""
        try:
            # Teacher model trainer
            self.teacher_trainer = YOLOv5Trainer(
                data_yaml_path=self.config.data_yaml_path,
                img_size=self.config.img_size,
                batch_size=self.config.batch_size,
                epochs=self.config.teacher_epochs,
                device=self.device
            )
            
            # Knowledge distillation trainer
            if 'teacher_model_path' in self.state:
                self.distillation_trainer = KnowledgeDistillationTrainer(
                    teacher_model_path=self.state['teacher_model_path'],
                    data_yaml_path=self.config.data_yaml_path,
                    img_size=self.config.img_size,
                    batch_size=self.config.batch_size,
                    epochs=self.config.distillation_epochs,
                    device=self.device
                )
            
            # Model pruner
            if 'student_model_path' in self.state:
                self.pruner = StructuredPruner(
                    model=self.state.get('student_model'),
                    dataloader=self._create_dataloader('train'),
                    device=self.device,
                    pruning_iterations=self.config.pruning_iterations,
                    target_sparsity=self.config.target_sparsity,
                    fine_tune_epochs=self.config.fine_tune_epochs
                )
            
            # Model quantizer
            if 'pruned_model_path' in self.state:
                self.quantizer = ModelQuantizer(
                    model=self.state.get('pruned_model'),
                    calibration_dataloader=self._create_dataloader('calibration'),
                    eval_dataloader=self._create_dataloader('val'),
                    device=self.device
                )
            
            # Model deployer
            if 'quantized_model_path' in self.state:
                self.deployer = ModelDeployer(
                    model=self.state.get('quantized_model'),
                    input_shape=self.config.input_shape,
                    hardware_profile=self.config.hardware_profile,
                    device=self.device
                )
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def _load_state(self) -> Dict:
        """Load pipeline state from checkpoint."""
        state_path = self.checkpoint_dir / 'pipeline_state.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_state(self, state: Dict):
        """Save pipeline state to checkpoint."""
        state_path = self.checkpoint_dir / 'pipeline_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=4)
    
    def _save_checkpoint(self, step: str, model: nn.Module, metrics: Dict):
        """Save checkpoint for a pipeline step."""
        try:
            checkpoint_dir = self.checkpoint_dir / step
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = checkpoint_dir / 'model.pt'
            torch.save({
                'model': model.state_dict(),
                'metrics': metrics,
                'config': self.config.__dict__
            }, model_path)
            
            # Save metrics
            metrics_path = checkpoint_dir / 'metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Update state
            self.state[f'{step}_model_path'] = str(model_path)
            self.state[f'{step}_metrics'] = metrics
            self._save_state(self.state)
            
            logger.info(f"Saved checkpoint for {step} step")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint for {step}: {e}")
            raise
    
    def train_teacher(self) -> Dict:
        """Train the teacher model."""
        if 'teacher_model_path' in self.state:
            logger.info("Teacher model already trained, loading from checkpoint...")
            return self.state['teacher_metrics']
        
        logger.info("Training teacher model...")
        try:
            metrics = self.teacher_trainer.train()
            self._save_checkpoint('teacher', self.teacher_trainer.model, metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error training teacher model: {e}")
            raise
    
    def distill_knowledge(self) -> Dict:
        """Perform knowledge distillation."""
        if 'student_model_path' in self.state:
            logger.info("Student model already trained, loading from checkpoint...")
            return self.state['student_metrics']
        
        logger.info("Performing knowledge distillation...")
        try:
            metrics = self.distillation_trainer.train()
            self._save_checkpoint('student', self.distillation_trainer.student, metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error in knowledge distillation: {e}")
            raise
    
    def prune_model(self) -> Dict:
        """Prune the student model."""
        if 'pruned_model_path' in self.state:
            logger.info("Model already pruned, loading from checkpoint...")
            return self.state['pruning_metrics']
        
        logger.info("Pruning model...")
        try:
            metrics = self.pruner.prune_model()
            self._save_checkpoint('pruned', self.pruner.model, metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error pruning model: {e}")
            raise
    
    def quantize_model(self) -> Dict:
        """Quantize the pruned model."""
        if 'quantized_model_path' in self.state:
            logger.info("Model already quantized, loading from checkpoint...")
            return self.state['quantization_metrics']
        
        logger.info("Quantizing model...")
        try:
            metrics = self.quantizer.quantize_and_evaluate()
            self._save_checkpoint('quantized', self.quantizer.model_quantized, metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error quantizing model: {e}")
            raise
    
    def deploy_model(self) -> Dict:
        """Deploy the quantized model."""
        if 'deployment_metrics' in self.state:
            logger.info("Model already deployed, loading from checkpoint...")
            return self.state['deployment_metrics']
        
        logger.info("Deploying model...")
        try:
            metrics = self.deployer.deploy_and_test(self._create_dataloader('test'))
            self._save_checkpoint('deployment', self.deployer.model, metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def run_pipeline(self, steps: Optional[list] = None):
        """
        Run the optimization pipeline.
        
        Args:
            steps: List of steps to run (None for all steps)
        """
        if steps is None:
            steps = ['teacher', 'distill', 'prune', 'quantize', 'deploy']
        
        logger.info(f"Starting optimization pipeline with steps: {steps}")
        
        try:
            if 'teacher' in steps:
                self.train_teacher()
            
            if 'distill' in steps:
                self.distill_knowledge()
            
            if 'prune' in steps:
                self.prune_model()
            
            if 'quantize' in steps:
                self.quantize_model()
            
            if 'deploy' in steps:
                self.deploy_model()
            
            logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Save final state
            self._save_state(self.state)
    
    def generate_report(self):
        """Generate a comprehensive report of the optimization process."""
        report_path = self.output_dir / 'optimization_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Model Optimization Pipeline Report\n")
            f.write("===============================\n\n")
            
            # Teacher model metrics
            if 'teacher_metrics' in self.state:
                f.write("Teacher Model Training:\n")
                f.write(f"Accuracy: {self.state['teacher_metrics']['accuracy']:.2f}%\n")
                f.write(f"Training Time: {self.state['teacher_metrics']['training_time']:.2f}s\n\n")
            
            # Student model metrics
            if 'student_metrics' in self.state:
                f.write("Knowledge Distillation:\n")
                f.write(f"Accuracy: {self.state['student_metrics']['accuracy']:.2f}%\n")
                f.write(f"Training Time: {self.state['student_metrics']['training_time']:.2f}s\n\n")
            
            # Pruning metrics
            if 'pruning_metrics' in self.state:
                f.write("Model Pruning:\n")
                f.write(f"Final Sparsity: {self.state['pruning_metrics']['sparsity']:.2f}%\n")
                f.write(f"Accuracy Change: {self.state['pruning_metrics']['accuracy_change']:.2f}%\n\n")
            
            # Quantization metrics
            if 'quantization_metrics' in self.state:
                f.write("Model Quantization:\n")
                f.write(f"Model Size Reduction: {self.state['quantization_metrics']['size_reduction']:.2f}%\n")
                f.write(f"Accuracy Change: {self.state['quantization_metrics']['accuracy_change']:.2f}%\n\n")
            
            # Deployment metrics
            if 'deployment_metrics' in self.state:
                f.write("Model Deployment:\n")
                f.write(f"Hardware: {self.state['deployment_metrics']['hardware_profile']}\n")
                f.write(f"Mean Latency: {self.state['deployment_metrics']['inference_metrics']['mean_latency_ms']:.2f}ms\n")
                f.write(f"FPS: {self.state['deployment_metrics']['inference_metrics']['fps']:.2f}\n")
                f.write(f"Power Consumption: {self.state['deployment_metrics']['hardware_metrics']['power_consumption']:.2f}W\n")
        
        logger.info(f"Report generated at {report_path}")

def main():
    # Load configuration
    config_path = 'config/pipeline_config.yaml'
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create pipeline configuration
    config = PipelineConfig(**config_dict)
    
    # Initialize pipeline
    pipeline = OptimizationPipeline(config)
    
    # Run pipeline
    pipeline.run_pipeline()
    
    # Generate report
    pipeline.generate_report()

if __name__ == "__main__":
    main() 