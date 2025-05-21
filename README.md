# Drone Vision Environment Setup

This repository contains the necessary setup files and utilities for deep learning model optimization and drone deployment simulation.

## Environment Setup

1. Create a new Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Verify the installation:
```bash
python src/utils/environment_setup.py
```

## Dataset Setup

### COCO Dataset
1. Download the COCO dataset from [COCO website](https://cocodataset.org/#download)
2. Extract the dataset to a directory of your choice
3. Use the DatasetLoader class to load and process the dataset:

```python
from src.utils.environment_setup import DatasetLoader

# Initialize the dataset loader
dataset = DatasetLoader(
    dataset_path="/path/to/coco/dataset",
    dataset_type="coco"
)

# Get dataset statistics
stats = dataset.get_dataset_stats()
print(f"Number of images: {stats['num_images']}")
print(f"Number of categories: {stats['num_categories']}")

# Load a specific image
image_tensor, annotations = dataset.load_image(image_id=1)
```

### Pascal VOC Dataset
Support for Pascal VOC dataset is planned for future updates.

## Features

- Deep Learning Framework Support:
  - PyTorch
  - TensorFlow
  - ONNX
  - TensorRT

- Model Optimization Tools:
  - Model Pruning
  - Quantization
  - Knowledge Distillation

- Drone Simulation:
  - AirSim integration
  - Performance simulation
  - Battery life estimation

## Development

The project uses several development tools:
- Black for code formatting
- Pylint for code linting
- Pytest for testing

To run the development tools:
```bash
# Format code
black .

# Run linter
pylint src/

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
