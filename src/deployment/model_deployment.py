import os
import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import logging
import time
import psutil
import GPUtil
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    """Hardware profile for simulation."""
    name: str
    cpu_cores: int
    cpu_freq_mhz: float
    gpu_memory_mb: int
    system_memory_mb: int
    power_consumption_w: float
    is_gpu_available: bool

# Predefined hardware profiles
HARDWARE_PROFILES = {
    'jetson_nano': HardwareProfile(
        name='NVIDIA Jetson Nano',
        cpu_cores=4,
        cpu_freq_mhz=1400,
        gpu_memory_mb=4096,
        system_memory_mb=4096,
        power_consumption_w=10.0,
        is_gpu_available=True
    ),
    'raspberry_pi_4': HardwareProfile(
        name='Raspberry Pi 4',
        cpu_cores=4,
        cpu_freq_mhz=1500,
        gpu_memory_mb=0,
        system_memory_mb=4096,
        power_consumption_w=7.5,
        is_gpu_available=False
    )
}

class ModelDeployer:
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int, int],
        output_dir: str = "runs/deployment",
        hardware_profile: str = "jetson_nano"
    ):
        """
        Initialize the model deployer.
        
        Args:
            model: The quantized model to deploy
            input_shape: Input tensor shape (batch_size, channels, height, width)
            output_dir: Directory to save exported models
            hardware_profile: Target hardware profile for simulation
        """
        self.model = model
        self.input_shape = input_shape
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set hardware profile
        if hardware_profile not in HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware profile: {hardware_profile}")
        self.hardware = HARDWARE_PROFILES[hardware_profile]
        
        # Create dummy input for export
        self.dummy_input = torch.randn(input_shape)
        
    def export_to_onnx(self) -> str:
        """Export model to ONNX format."""
        logger.info("Exporting model to ONNX format...")
        
        # Set model to eval mode
        self.model.eval()
        
        # Export to ONNX
        onnx_path = self.output_dir / "model.onnx"
        torch.onnx.export(
            self.model,
            self.dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Model exported to {onnx_path}")
        return str(onnx_path)
    
    def simulate_hardware_constraints(self) -> Dict:
        """Simulate hardware constraints and limitations."""
        logger.info(f"Simulating hardware constraints for {self.hardware.name}...")
        
        # Get current system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpu_info = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        
        # Calculate resource utilization
        cpu_utilization = min(100, (cpu_percent / self.hardware.cpu_cores) * 100)
        memory_utilization = (memory.used / (self.hardware.system_memory_mb * 1024 * 1024)) * 100
        
        if self.hardware.is_gpu_available and gpu_info:
            gpu_utilization = gpu_info.memoryUtil * 100
        else:
            gpu_utilization = 0
        
        # Calculate power consumption
        power_consumption = self.hardware.power_consumption_w * (cpu_utilization / 100)
        if self.hardware.is_gpu_available and gpu_info:
            power_consumption += gpu_info.power / 1000  # Convert mW to W
        
        return {
            'cpu_utilization': cpu_utilization,
            'memory_utilization': memory_utilization,
            'gpu_utilization': gpu_utilization,
            'power_consumption': power_consumption
        }
    
    def test_inference_speed(self, num_runs: int = 100) -> Dict[str, float]:
        """Test model inference speed on target hardware."""
        logger.info("Testing inference speed...")
        
        # Create ONNX runtime session
        onnx_path = self.export_to_onnx()
        session = onnxruntime.InferenceSession(onnx_path)
        
        # Prepare input data
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Warm-up runs
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Measure inference time
        latencies = []
        for _ in tqdm(range(num_runs), desc="Measuring inference speed"):
            start_time = time.time()
            session.run(None, {input_name: dummy_input})
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        return {
            'mean_latency_ms': mean_latency,
            'std_latency_ms': std_latency,
            'p95_latency_ms': p95_latency,
            'fps': 1000 / mean_latency
        }
    
    def test_memory_usage(self) -> Dict[str, float]:
        """Test model memory usage."""
        logger.info("Testing memory usage...")
        
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Load and run model
        onnx_path = self.export_to_onnx()
        session = onnxruntime.InferenceSession(onnx_path)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        session.run(None, {input_name: dummy_input})
        
        # Get peak memory usage
        peak_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        return {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - initial_memory
        }
    
    def test_accuracy(self, test_dataloader) -> float:
        """Test model accuracy on test dataset."""
        logger.info("Testing model accuracy...")
        
        # Create ONNX runtime session
        onnx_path = self.export_to_onnx()
        session = onnxruntime.InferenceSession(onnx_path)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_dataloader, desc="Evaluating accuracy"):
                # Convert input to numpy
                input_data = data.numpy()
                
                # Run inference
                output = session.run(None, {'input': input_data})[0]
                pred = np.argmax(output, axis=1)
                
                # Calculate accuracy
                correct += np.sum(pred == target.numpy())
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy
    
    def deploy_and_test(self, test_dataloader) -> Dict:
        """Deploy model and run comprehensive tests."""
        logger.info(f"Starting deployment and testing for {self.hardware.name}...")
        
        # Export model
        onnx_path = self.export_to_onnx()
        
        # Run tests
        inference_metrics = self.test_inference_speed()
        memory_metrics = self.test_memory_usage()
        accuracy = self.test_accuracy(test_dataloader)
        hardware_metrics = self.simulate_hardware_constraints()
        
        # Compile results
        results = {
            'hardware_profile': self.hardware.name,
            'inference_metrics': inference_metrics,
            'memory_metrics': memory_metrics,
            'accuracy': accuracy,
            'hardware_metrics': hardware_metrics
        }
        
        # Save results
        self.save_results(results)
        
        return results
    
    def save_results(self, results: Dict):
        """Save deployment results."""
        # Save results to JSON
        results_path = self.output_dir / "deployment_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save results to text file
        text_path = self.output_dir / "deployment_results.txt"
        with open(text_path, 'w') as f:
            f.write(f"Deployment Results for {results['hardware_profile']}\n")
            f.write("==========================================\n\n")
            
            f.write("Inference Metrics:\n")
            f.write(f"Mean Latency: {results['inference_metrics']['mean_latency_ms']:.2f}ms\n")
            f.write(f"FPS: {results['inference_metrics']['fps']:.2f}\n\n")
            
            f.write("Memory Usage:\n")
            f.write(f"Peak Memory: {results['memory_metrics']['peak_memory_mb']:.2f}MB\n")
            f.write(f"Memory Increase: {results['memory_metrics']['memory_increase_mb']:.2f}MB\n\n")
            
            f.write("Accuracy:\n")
            f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n\n")
            
            f.write("Hardware Utilization:\n")
            f.write(f"CPU Utilization: {results['hardware_metrics']['cpu_utilization']:.2f}%\n")
            f.write(f"Memory Utilization: {results['hardware_metrics']['memory_utilization']:.2f}%\n")
            f.write(f"GPU Utilization: {results['hardware_metrics']['gpu_utilization']:.2f}%\n")
            f.write(f"Power Consumption: {results['hardware_metrics']['power_consumption']:.2f}W\n")
        
        logger.info(f"Results saved to {results_path} and {text_path}")

def main():
    # Load your quantized model
    model_path = 'runs/quantized/quantized_model.pt'  # Update with your model path
    checkpoint = torch.load(model_path)
    model = MobileNetSSD(num_classes=80)  # Your model class
    model.load_state_dict(checkpoint['model'])
    
    # Create your test dataloader
    # This should be adapted based on your dataset
    test_dataloader = ...  # Your test DataLoader
    
    # Initialize deployer for Jetson Nano
    deployer = ModelDeployer(
        model=model,
        input_shape=(1, 3, 640, 640),  # Adjust based on your model
        hardware_profile='jetson_nano'
    )
    
    # Deploy and test
    results = deployer.deploy_and_test(test_dataloader)
    
    # Print summary
    logger.info("\nDeployment Summary:")
    logger.info(f"Hardware: {results['hardware_profile']}")
    logger.info(f"Mean Latency: {results['inference_metrics']['mean_latency_ms']:.2f}ms")
    logger.info(f"FPS: {results['inference_metrics']['fps']:.2f}")
    logger.info(f"Accuracy: {results['accuracy']:.2f}%")
    logger.info(f"Peak Memory: {results['memory_metrics']['peak_memory_mb']:.2f}MB")
    logger.info(f"Power Consumption: {results['hardware_metrics']['power_consumption']:.2f}W")

if __name__ == "__main__":
    main() 