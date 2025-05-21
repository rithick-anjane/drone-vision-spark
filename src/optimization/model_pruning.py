import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import logging
from tqdm import tqdm
import copy
import numpy as np
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredPruner:
    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = None,
        pruning_iterations: int = 5,
        target_sparsity: float = 0.7,
        fine_tune_epochs: int = 5,
        learning_rate: float = 0.001
    ):
        """
        Initialize the structured pruner.
        
        Args:
            model: The model to prune
            dataloader: DataLoader for fine-tuning
            device: Device to use for pruning
            pruning_iterations: Number of pruning iterations
            target_sparsity: Target sparsity ratio (0-1)
            fine_tune_epochs: Number of epochs to fine-tune after each pruning
            learning_rate: Learning rate for fine-tuning
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pruning_iterations = pruning_iterations
        self.target_sparsity = target_sparsity
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        
        # Create output directory
        self.output_dir = Path("runs/pruned") / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=fine_tune_epochs)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get list of prunable layers (Conv2d layers)."""
        prunable_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prunable_layers.append((name, module))
        return prunable_layers
    
    def compute_layer_importance(self, layer: nn.Conv2d) -> torch.Tensor:
        """Compute importance scores for filters using L1 norm."""
        return torch.norm(layer.weight.data, p=1, dim=(1, 2, 3))
    
    def get_pruning_mask(
        self,
        importance_scores: torch.Tensor,
        current_sparsity: float,
        target_sparsity: float
    ) -> torch.Tensor:
        """Compute binary mask for pruning based on importance scores."""
        n_filters = len(importance_scores)
        n_filters_to_prune = int(n_filters * (target_sparsity - current_sparsity))
        
        if n_filters_to_prune <= 0:
            return torch.ones(n_filters, dtype=torch.bool)
        
        # Get indices of least important filters
        _, indices = torch.topk(importance_scores, n_filters_to_prune, largest=False)
        
        # Create binary mask
        mask = torch.ones(n_filters, dtype=torch.bool)
        mask[indices] = False
        return mask
    
    def apply_structured_pruning(self, layer: nn.Conv2d, mask: torch.Tensor):
        """Apply structured pruning to a layer."""
        # Create a copy of the layer's weight
        weight = layer.weight.data.clone()
        
        # Zero out pruned filters
        weight[~mask] = 0
        
        # Update layer weights
        layer.weight.data.copy_(weight)
        
        # If the layer has a bias, zero it out for pruned filters
        if layer.bias is not None:
            bias = layer.bias.data.clone()
            bias[~mask] = 0
            layer.bias.data.copy_(bias)
    
    def fine_tune(self):
        """Fine-tune the model after pruning."""
        self.model.train()
        for epoch in range(self.fine_tune_epochs):
            pbar = tqdm(self.dataloader, desc=f'Fine-tuning Epoch {epoch + 1}/{self.fine_tune_epochs}')
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.compute_loss(output, target)
                loss.backward()
                self.optimizer.step()
                
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            
            self.scheduler.step()
    
    def compute_loss(self, output, target):
        """Compute loss for fine-tuning."""
        # This should be adapted based on your specific loss function
        return nn.CrossEntropyLoss()(output, target)
    
    def compute_model_sparsity(self) -> float:
        """Compute current model sparsity."""
        total_params = 0
        zero_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                total_params += module.weight.numel()
                zero_params += (module.weight.data == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0
    
    def prune_model(self):
        """Perform iterative structured pruning with fine-tuning."""
        logger.info("Starting structured pruning...")
        
        # Get prunable layers
        prunable_layers = self.get_prunable_layers()
        current_sparsity = 0.0
        sparsity_increment = (self.target_sparsity - current_sparsity) / self.pruning_iterations
        
        for iteration in range(self.pruning_iterations):
            logger.info(f"Pruning iteration {iteration + 1}/{self.pruning_iterations}")
            
            # Compute and apply pruning for each layer
            for name, layer in prunable_layers:
                # Compute importance scores
                importance_scores = self.compute_layer_importance(layer)
                
                # Get pruning mask
                target_sparsity = current_sparsity + sparsity_increment
                mask = self.get_pruning_mask(importance_scores, current_sparsity, target_sparsity)
                
                # Apply pruning
                self.apply_structured_pruning(layer, mask)
                
                logger.info(f"Layer {name}: Pruned {(~mask).sum().item()}/{len(mask)} filters")
            
            # Update current sparsity
            current_sparsity = self.compute_model_sparsity()
            logger.info(f"Current model sparsity: {current_sparsity:.3f}")
            
            # Fine-tune the model
            logger.info("Fine-tuning after pruning...")
            self.fine_tune()
            
            # Save intermediate model
            self.save_model(f'pruned_iter_{iteration + 1}.pt')
        
        # Save final pruned model
        self.save_model('pruned_final.pt')
        logger.info("Pruning completed!")
    
    def save_model(self, filename: str):
        """Save the pruned model."""
        save_path = self.output_dir / filename
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'sparsity': self.compute_model_sparsity()
        }, save_path)
        logger.info(f"Model saved to {save_path}")

def main():
    # Load your distilled student model
    model_path = 'runs/distill/exp_*/best.pt'  # Update with your model path
    checkpoint = torch.load(model_path)
    model = MobileNetSSD(num_classes=80)  # Your student model class
    model.load_state_dict(checkpoint['model'])
    
    # Create your dataloader for fine-tuning
    # This should be adapted based on your dataset
    dataloader = ...  # Your DataLoader
    
    # Initialize pruner
    pruner = StructuredPruner(
        model=model,
        dataloader=dataloader,
        pruning_iterations=5,
        target_sparsity=0.7,
        fine_tune_epochs=5,
        learning_rate=0.001
    )
    
    # Start pruning
    pruner.prune_model()

if __name__ == "__main__":
    main() 