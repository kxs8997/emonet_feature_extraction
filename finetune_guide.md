# EMONET Finetuning Guide for AFEW-VA Dataset

This guide walks you through finetuning EMONET on the AFEW-VA dataset for valence and arousal regression.

## Prerequisites

1. **AFEW-VA Dataset**: Download and organize the dataset
2. **Pretrained EMONET weights**: Ensure you have `pretrained/emonet_8.pth` or `pretrained/emonet_5.pth`
3. **Environment setup**: Install required dependencies

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate your environment
source emonet_env/bin/activate

# Install training requirements
pip install -r requirements_training.txt
```

### 2. Prepare Dataset

Your AFEW-VA dataset should be organized as:
```
afew_va_root/
├── 001/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
├── 002/
│   └── ...
└── features_manifest.json  # Your existing manifest file
```

The script will automatically create an `annotations.json` file from your `features_manifest.json`.

### 3. Basic Finetuning Command

```bash
python train_emonet.py \
    --data_root /path/to/afew_va \
    --pretrained_path pretrained/emonet_8.pth \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --output_dir output/afew_va_finetune
```

### 4. Advanced Configuration Options

#### Training Parameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: L2 regularization (default: 1e-5)
- `--grad_clip`: Gradient clipping threshold (default: 1.0)

#### Model Configuration
- `--n_expression`: Number of expression classes (5 or 8, default: 8)
- `--freeze_backbone`: Freeze backbone, only train emotion head
- `--pretrained_path`: Path to pretrained weights

#### Loss Functions
- `--loss_function`: Choose from 'mse', 'mae', or 'ccc' (default: 'mse')
  - **MSE**: Mean Squared Error - standard regression loss
  - **MAE**: Mean Absolute Error - more robust to outliers
  - **CCC**: Concordance Correlation Coefficient - commonly used for VA regression

#### Data Splitting
- `--train_split`: Training data ratio (default: 0.8)
- `--val_split`: Validation data ratio (default: 0.2)

### 5. Recommended Training Strategies

#### Strategy 1: Conservative Finetuning (Recommended for small datasets)
```bash
python train_emonet.py \
    --data_root /path/to/afew_va \
    --pretrained_path pretrained/emonet_8.pth \
    --freeze_backbone \
    --batch_size 16 \
    --epochs 30 \
    --learning_rate 1e-5 \
    --loss_function ccc \
    --output_dir output/conservative_finetune
```

#### Strategy 2: Full Finetuning (For larger datasets)
```bash
python train_emonet.py \
    --data_root /path/to/afew_va \
    --pretrained_path pretrained/emonet_8.pth \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-4 \
    --loss_function mse \
    --output_dir output/full_finetune
```

#### Strategy 3: Progressive Unfreezing
1. First, train with frozen backbone:
```bash
python train_emonet.py \
    --data_root /path/to/afew_va \
    --pretrained_path pretrained/emonet_8.pth \
    --freeze_backbone \
    --epochs 20 \
    --learning_rate 1e-4 \
    --output_dir output/stage1_frozen
```

2. Then, continue training with unfrozen backbone:
```bash
python train_emonet.py \
    --data_root /path/to/afew_va \
    --pretrained_path output/stage1_frozen/best_model.pth \
    --epochs 30 \
    --learning_rate 1e-5 \
    --output_dir output/stage2_unfrozen
```

### 6. Monitoring Training

The training script provides:
- **Real-time progress**: Progress bars with loss values
- **Logging**: Detailed logs saved to `output_dir/logs/`
- **Checkpoints**: Model checkpoints saved every 10 epochs
- **Best model**: Automatically saves the best model based on validation loss
- **Training curves**: Plots saved as `training_curves.png`

### 7. Output Structure

After training, your output directory will contain:
```
output/
├── logs/
│   └── training_YYYYMMDD_HHMMSS.log
├── args.json                    # Training arguments
├── best_model.pth              # Best model checkpoint
├── checkpoint_epoch_N.pth      # Regular checkpoints
└── training_curves.png         # Loss and learning rate plots
```

### 8. Using the Finetuned Model

To use your finetuned model for inference:

```python
import torch
from emonet.models.emonet import EmoNet

# Load the finetuned model
model = EmoNet(n_expression=8)
checkpoint = torch.load('output/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
with torch.no_grad():
    outputs = model(input_image)
    valence = outputs['emo_feat_2'][:, -2]  # Second to last output
    arousal = outputs['emo_feat_2'][:, -1]  # Last output
```

### 9. Troubleshooting

#### Common Issues:

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Face detection fails**: The dataset loader includes fallback to resize original images
3. **Slow training**: Use more workers in DataLoader or reduce image resolution
4. **Poor convergence**: Try different loss functions (CCC often works better for VA regression)

#### Performance Tips:

- Start with a small learning rate (1e-5) if finetuning from pretrained weights
- Use CCC loss for better correlation with ground truth
- Monitor both training and validation losses to detect overfitting
- Consider data augmentation if you have limited training data

### 10. Expected Results

With proper finetuning, you should expect:
- **Training loss**: Steadily decreasing over epochs
- **Validation loss**: Should follow training loss without large gaps
- **Convergence**: Usually within 20-50 epochs depending on dataset size
- **Performance**: CCC values > 0.3 are considered reasonable for VA regression

## Next Steps

1. Run the basic finetuning command with your dataset
2. Monitor the training progress and adjust hyperparameters as needed
3. Evaluate the finetuned model on your test set
4. Consider ensemble methods or additional data augmentation for better performance

For questions or issues, check the training logs and adjust parameters accordingly.
