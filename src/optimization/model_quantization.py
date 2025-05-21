import os
import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import quantize_dynamic
from torch.quantization import get_default_qconfig
from torch.quantization import prepare_qat, convert
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import time
from tqdm import tqdm
from typing import Dict, Tuple, List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelQuantizer:
    def __init__(
        self,
        model: nn.Module,
        calibration_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: str = None,
        output_dir: str = "runs/quantized"
    ):
        """
        Initialize the model quantizer.
        
        Args:
            model: The model to quantize
            calibration_dataloader: DataLoader for calibration
            eval_dataloader: DataLoader for evaluation
            device: Device to use for quantization
            output_dir: Directory to save quantized models
        """
        self.model = model
        self.calibration_dataloader = calibration_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def prepare_for_quantization(self):
        """Prepare the model for quantization."""
        # Set model to eval mode
        self.model.eval()
        
        # Specify quantization configuration
        self.model.qconfig = get_default_qconfig('fbgemm')  # For x86 architecture
        
        # Prepare the model for quantization
        self.model_prepared = prepare_qat(self.model)
        logger.info("Model prepared for quantization")
    
    def calibrate(self):
        """Calibrate the model with calibration data."""
        logger.info("Starting calibration...")
        self.model_prepared.eval()
        
        with torch.no_grad():
            for data, _ in tqdm(self.calibration_dataloader, desc="Calibrating"):
                data = data.to(self.device)
                self.model_prepared(data)
        
        logger.info("Calibration completed")
    
    def quantize(self):
        """Convert the model to quantized version."""
        logger.info("Converting to quantized model...")
        self.model_quantized = convert(self.model_prepared)
        logger.info("Quantization completed")
    
    def evaluate_accuracy(self, model: nn.Module) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.eval_dataloader, desc="Evaluating accuracy"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def measure_latency(self, model: nn.Module, num_runs: int = 100) -> Dict[str, float]:
        """Measure model inference latency."""
        model.eval()
        latencies = []
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(10):
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)  # Adjust input size as needed
                _ = model(dummy_input)
        
        # Measure latency
        with torch.no_grad():
            for _ in tqdm(range(num_runs), desc="Measuring latency"):
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                
                # Measure inference time
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize()  # For GPU measurements
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        return {
            'mean_latency_ms': mean_latency,
            'std_latency_ms': std_latency,
            'p95_latency_ms': p95_latency
        }
    
    def evaluate_model_size(self, model: nn.Module) -> float:
        """Evaluate model size in MB."""
        # Save model to temporary file
        temp_path = self.output_dir / "temp_model.pt"
        torch.save(model.state_dict(), temp_path)
        
        # Get file size
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        # Remove temporary file
        os.remove(temp_path)
        
        return size_mb
    
    def quantize_and_evaluate(self):
        """Perform quantization and evaluate the model."""
        # Prepare model for quantization
        self.prepare_for_quantization()
        
        # Evaluate original model
        logger.info("Evaluating original model...")
        original_accuracy = self.evaluate_accuracy(self.model)
        original_metrics = self.measure_latency(self.model)
        original_size = self.evaluate_model_size(self.model)
        
        logger.info(f"Original Model Metrics:")
        logger.info(f"Accuracy: {original_accuracy:.2f}%")
        logger.info(f"Mean Latency: {original_metrics['mean_latency_ms']:.2f}ms")
        logger.info(f"Model Size: {original_size:.2f}MB")
        
        # Calibrate and quantize
        self.calibrate()
        self.quantize()
        
        # Evaluate quantized model
        logger.info("Evaluating quantized model...")
        quantized_accuracy = self.evaluate_accuracy(self.model_quantized)
        quantized_metrics = self.measure_latency(self.model_quantized)
        quantized_size = self.evaluate_model_size(self.model_quantized)
        
        logger.info(f"Quantized Model Metrics:")
        logger.info(f"Accuracy: {quantized_accuracy:.2f}%")
        logger.info(f"Mean Latency: {quantized_metrics['mean_latency_ms']:.2f}ms")
        logger.info(f"Model Size: {quantized_size:.2f}MB")
        
        # Save results
        results = {
            'original': {
                'accuracy': original_accuracy,
                'latency': original_metrics,
                'size_mb': original_size
            },
            'quantized': {
                'accuracy': quantized_accuracy,
                'latency': quantized_metrics,
                'size_mb': quantized_size
            }
        }
        
        # Save quantized model and results
        self.save_results(results)
    
    def save_results(self, results: Dict):
        """Save quantized model and evaluation results."""
        # Save quantized model
        model_path = self.output_dir / "quantized_model.pt"
        torch.save({
            'model': self.model_quantized.state_dict(),
            'results': results
        }, model_path)
        
        # Save results to text file
        results_path = self.output_dir / "quantization_results.txt"
        with open(results_path, 'w') as f:
            f.write("Quantization Results\n")
            f.write("===================\n\n")
            
            f.write("Original Model:\n")
            f.write(f"Accuracy: {results['original']['accuracy']:.2f}%\n")
            f.write(f"Mean Latency: {results['original']['latency']['mean_latency_ms']:.2f}ms\n")
            f.write(f"Model Size: {results['original']['size_mb']:.2f}MB\n\n")
            
            f.write("Quantized Model:\n")
            f.write(f"Accuracy: {results['quantized']['accuracy']:.2f}%\n")
            f.write(f"Mean Latency: {results['quantized']['latency']['mean_latency_ms']:.2f}ms\n")
            f.write(f"Model Size: {results['quantized']['size_mb']:.2f}MB\n\n")
            
            f.write("Improvements:\n")
            f.write(f"Size Reduction: {(1 - results['quantized']['size_mb']/results['original']['size_mb'])*100:.2f}%\n")
            f.write(f"Latency Improvement: {(1 - results['quantized']['latency']['mean_latency_ms']/results['original']['latency']['mean_latency_ms'])*100:.2f}%\n")
            f.write(f"Accuracy Change: {results['quantized']['accuracy'] - results['original']['accuracy']:.2f}%\n")
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Quantized model saved to {model_path}")

def main():
    # Load your pruned model
    model_path = 'runs/pruned/exp_*/pruned_final.pt'  # Update with your model path
    checkpoint = torch.load(model_path)
    model = MobileNetSSD(num_classes=80)  # Your model class
    model.load_state_dict(checkpoint['model'])
    
    # Create your dataloaders
    # This should be adapted based on your dataset
    calibration_dataloader = ...  # Your calibration DataLoader
    eval_dataloader = ...  # Your evaluation DataLoader
    
    # Initialize quantizer
    quantizer = ModelQuantizer(
        model=model,
        calibration_dataloader=calibration_dataloader,
        eval_dataloader=eval_dataloader
    )
    
    # Perform quantization and evaluation
    quantizer.quantize_and_evaluate()

if __name__ == "__main__":
    main() 