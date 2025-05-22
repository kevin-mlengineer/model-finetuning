# Knowledge Distillation for Text Classification

This project implements knowledge distillation for text classification using the AG News dataset. It demonstrates how to transfer knowledge from a larger BERT model (teacher) to a smaller DistilBERT model (student) while maintaining good performance.

## Overview

The project implements a knowledge distillation pipeline that:
1. Uses a pre-trained BERT model as the teacher
2. Trains a baseline DistilBERT student model
3. Trains a distilled DistilBERT student model using knowledge distillation
4. Compares the performance of all models

### What is Knowledge Distillation?

Knowledge distillation is a technique where a smaller model (student) learns to mimic the behavior of a larger model (teacher). The process involves:
- Training the student model on both hard labels and soft predictions from the teacher
- Using temperature scaling to soften the probability distributions
- Combining standard cross-entropy loss with distillation loss

## Features

- Automatic GPU detection and utilization (MPS for MacBook, CUDA for NVIDIA, CPU fallback)
- Knowledge distillation with temperature scaling
- Comprehensive logging and error handling
- Model evaluation with accuracy and F1-score metrics
- Model comparison with parameter counts
- Automatic model saving and loading
- Efficient data preprocessing and caching
- Memory-optimized training pipeline

## Requirements

The project requires the following Python packages:
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
transformers==4.36.2
datasets==2.18.0
torch==2.2.1
accelerate==0.25.0
```

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with MPS support (for MacBook) or CUDA support (for NVIDIA)
- 10GB free disk space for models and datasets

## Setup

1. Clone the repository:
```bash
git clone https://github.com/kevin-mlengineer/model-finetuning.git
cd model-finetuning
```

2. Create a virtual environment:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

4. Install the required packages:
```bash
pip install -r requirements.txt
```

## Code Structure


1. **Data Loading and Preparation** (`load_and_prepare_data`):
   - Loads the AG News dataset
   - Tokenizes the text data
   - Prepares the dataset for training
   - Implements efficient caching and cleanup
   - Handles dataset formatting for PyTorch

2. **Baseline Student Training** (`train_baseline_student`):
   - Initializes and trains a DistilBERT model
   - Uses standard cross-entropy loss
   - Evaluates model performance
   - Implements early stopping
   - Saves checkpoints and best model

3. **Knowledge Distillation** (`DistillationTrainer`):
   - Custom trainer class for knowledge distillation
   - Implements temperature scaling
   - Combines student and distillation losses
   - Handles gradient computation and backpropagation
   - Manages teacher model inference

4. **Distilled Student Training** (`train_distilled_student`):
   - Trains DistilBERT using knowledge distillation
   - Uses both hard labels and teacher's soft predictions
   - Evaluates distilled model performance
   - Implements custom loss computation
   - Manages model checkpoints

5. **Teacher Evaluation** (`evaluate_teacher`):
   - Evaluates the pre-trained BERT teacher model
   - Computes accuracy and F1-score
   - Handles batch processing
   - Manages GPU memory efficiently

### Key Parameters

- **Temperature (T)**: 5.0 (controls softness of probability distributions)
- **Alpha**: 0.5 (balances student loss and distillation loss)
- **Learning Rate**: 2e-5
- **Batch Size**: 32 (MPS/GPU) or 16 (CPU)
- **Epochs**: 3
- **Warmup Steps**: 500
- **Weight Decay**: 0.01
- **Max Sequence Length**: 128
- **Model Checkpointing**: Every epoch
- **Evaluation Strategy**: End of each epoch

## Usage

### Basic Usage

Run the main script:
```bash
python finetune.py
```

### Advanced Usage

1. Customize training parameters:
```python
training_args = TrainingArguments(
    output_dir='./custom_model',
    num_train_epochs=5,  # Increase epochs
    per_device_train_batch_size=64,  # Larger batch size
    learning_rate=1e-5,  # Custom learning rate
    # ... other parameters
)
```

2. Modify distillation parameters:
```python
distillation_trainer = DistillationTrainer(
    model=student_model,
    teacher_model=teacher_model,
    temp=2.0,  # Lower temperature
    alpha=0.7,  # More weight on student loss
    # ... other parameters
)
```

## Output

The script provides:
- Training progress logs
- Model evaluation metrics
- Comparison table showing:
  - Model accuracy
  - F1-scores
  - Parameter counts
- Saved models in:
  - `./baseline_student_model`
  - `./distilled_student_model`
  - `./tokenizer`

### Logging

The project uses Python's logging module with the following features:
- Timestamp for each log entry
- Log level indication
- Detailed error messages
- Training progress updates
- Performance metrics logging

## GPU Support

The code automatically detects and uses the appropriate device:
- MPS (Metal Performance Shaders) for MacBook GPUs
- CUDA for NVIDIA GPUs
- CPU as fallback

### GPU Optimization
- Automatic batch size adjustment
- Memory-efficient data loading
- Gradient accumulation for large models
- Mixed precision training support
- Efficient model checkpointing

## Error Handling

The code includes comprehensive error handling for:
- Data loading issues
- Training errors
- Model evaluation problems
- Resource management
- GPU memory management
- File system operations
- Network connectivity issues

## Performance Optimization

- Automatic batch size adjustment based on device
- Efficient data loading and preprocessing
- Proper GPU memory management
- Caching and cleanup of temporary files
- Optimized tokenization pipeline
- Efficient model checkpointing
- Memory-efficient training loop

## Model Architecture

### Teacher Model (BERT)
- Base BERT architecture
- 12 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 110M parameters

### Student Model (DistilBERT)
- Distilled BERT architecture
- 6 transformer layers
- 768 hidden dimensions
- 12 attention heads
- 66M parameters
