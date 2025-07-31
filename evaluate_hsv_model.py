#!/usr/bin/env python3
"""
HSV-based EMONET Evaluation Script

Evaluates a trained HSV-based EMONET model on test set using comprehensive metrics:
- PCC (Pearson Correlation Coefficient)
- CCC (Concordance Correlation Coefficient) 
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- SAGR (Sign Agreement Rate)

Supports both feature-based and full model evaluation on HSV images.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import EMONET modules
from emonet.models import EmoNet
from emonet.metrics import PCC, CCC, RMSE, SAGR
from emonet.evaluation import evaluate_metrics

class HSVFeatureDataset(Dataset):
    """Dataset for loading pre-extracted HSV features"""
    
    def __init__(self, manifest_file, data_root='.'):
        self.data_root = Path(data_root)
        
        # Load manifest
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        
        self.feature_paths = list(self.manifest.keys())
        print(f"Loaded {len(self.feature_paths)} HSV feature samples")
    
    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        full_path = self.data_root / feature_path
        
        # Load feature
        data = np.load(full_path)
        feature = torch.FloatTensor(data['feature'])
        
        # Load labels
        labels = self.manifest[feature_path]
        valence = torch.FloatTensor([labels['valence']])
        arousal = torch.FloatTensor([labels['arousal']])
        
        return {
            'feature': feature,
            'valence': valence,
            'arousal': arousal,
            'path': feature_path
        }

class HSVImageDataset(Dataset):
    """Dataset for loading images and converting to HSV on-the-fly"""
    
    def __init__(self, manifest_file, data_root='.', image_size=256):
        self.data_root = Path(data_root)
        self.image_size = image_size
        
        # Load manifest (assuming it points to original images)
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        
        # Convert feature paths to image paths
        self.image_data = []
        for feature_path, labels in self.manifest.items():
            # Convert features_hsv/video_id/frame_id.npz to original image path
            parts = Path(feature_path).parts
            if len(parts) >= 3 and parts[0] == 'features_hsv':
                video_id = parts[1]
                frame_id = Path(parts[2]).stem  # Remove .npz extension
                
                # Try both .png and .jpg
                img_path_png = self.data_root / os.environ.get('AFEW_VA_ROOT', 
                    "/home/ksubramanian/.cache/kagglehub/datasets/hoanguyensgu/afew-va/versions/1/AFEW-VA") / video_id / f"{frame_id}.png"
                img_path_jpg = self.data_root / os.environ.get('AFEW_VA_ROOT', 
                    "/home/ksubramanian/.cache/kagglehub/datasets/hoanguyensgu/afew-va/versions/1/AFEW-VA") / video_id / f"{frame_id}.jpg"
                
                if img_path_png.exists():
                    img_path = img_path_png
                elif img_path_jpg.exists():
                    img_path = img_path_jpg
                else:
                    continue
                
                self.image_data.append({
                    'image_path': img_path,
                    'valence': labels['valence'],
                    'arousal': labels['arousal']
                })
        
        print(f"Loaded {len(self.image_data)} HSV image samples")
    
    def rgb_to_hsv_tensor(self, rgb_tensor):
        """Convert RGB tensor to HSV tensor"""
        # Convert tensor to numpy for OpenCV processing
        rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        rgb_np = (rgb_np * 255).astype(np.uint8)  # Convert to 0-255 range
        
        # Convert RGB to HSV using OpenCV
        hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
        
        # Normalize HSV values to [0, 1] range
        hsv_np = hsv_np.astype(np.float32)
        hsv_np[:, :, 0] /= 179.0  # Hue
        hsv_np[:, :, 1] /= 255.0  # Saturation
        hsv_np[:, :, 2] /= 255.0  # Value
        
        # Convert back to tensor
        hsv_tensor = torch.from_numpy(hsv_np).permute(2, 0, 1)  # (C, H, W)
        return hsv_tensor
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        data = self.image_data[idx]
        
        # Load image
        image = io.imread(str(data['image_path']))
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass  # Already RGB
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, :3]
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        rgb_tensor = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        # Convert to HSV
        hsv_tensor = self.rgb_to_hsv_tensor(rgb_tensor)
        
        return {
            'image': hsv_tensor,
            'valence': torch.FloatTensor([data['valence']]),
            'arousal': torch.FloatTensor([data['arousal']]),
            'path': str(data['image_path'])
        }

class SimpleRegressionHead(torch.nn.Module):
    """Simple regression head for feature-based evaluation"""
    
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=2):
        super().__init__()
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.regressor(x)

def evaluate_feature_model(model_path, manifest_file, data_root, device, batch_size=64):
    """Evaluate feature-based model"""
    print(f"Evaluating feature-based model: {model_path}")
    
    # Load dataset
    dataset = HSVFeatureDataset(manifest_file, data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = SimpleRegressionHead()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Collect predictions and ground truth
    all_valence_pred = []
    all_arousal_pred = []
    all_valence_gt = []
    all_arousal_gt = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch['feature'].to(device)
            valence_gt = batch['valence'].squeeze().cpu().numpy()
            arousal_gt = batch['arousal'].squeeze().cpu().numpy()
            
            # Forward pass
            predictions = model(features)
            valence_pred = predictions[:, 0].cpu().numpy()
            arousal_pred = predictions[:, 1].cpu().numpy()
            
            all_valence_pred.extend(valence_pred)
            all_arousal_pred.extend(arousal_pred)
            all_valence_gt.extend(valence_gt)
            all_arousal_gt.extend(arousal_gt)
    
    return np.array(all_valence_pred), np.array(all_arousal_pred), \
           np.array(all_valence_gt), np.array(all_arousal_gt)

def evaluate_full_model(model_path, manifest_file, data_root, device, batch_size=32):
    """Evaluate full EMONET model on HSV images"""
    print(f"Evaluating full EMONET model: {model_path}")
    
    # Load dataset
    dataset = HSVImageDataset(manifest_file, data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Load model
    model = EmoNet(n_expression=8)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Collect predictions and ground truth
    all_valence_pred = []
    all_arousal_pred = []
    all_valence_gt = []
    all_arousal_gt = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            valence_gt = batch['valence'].squeeze().cpu().numpy()
            arousal_gt = batch['arousal'].squeeze().cpu().numpy()
            
            # Forward pass
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'valence' in outputs and 'arousal' in outputs:
                    valence_pred = outputs['valence'].squeeze().cpu().numpy()
                    arousal_pred = outputs['arousal'].squeeze().cpu().numpy()
                else:
                    # Extract from final predictions
                    predictions = list(outputs.values())[-1]
                    if len(predictions.shape) == 2 and predictions.shape[1] >= 2:
                        valence_pred = predictions[:, -2].cpu().numpy()
                        arousal_pred = predictions[:, -1].cpu().numpy()
                    else:
                        print("Warning: Unable to extract valence/arousal from model outputs")
                        continue
            else:
                # Direct tensor output
                if len(outputs.shape) == 2 and outputs.shape[1] >= 2:
                    valence_pred = outputs[:, -2].cpu().numpy()
                    arousal_pred = outputs[:, -1].cpu().numpy()
                else:
                    print("Warning: Unable to extract valence/arousal from model outputs")
                    continue
            
            all_valence_pred.extend(valence_pred)
            all_arousal_pred.extend(arousal_pred)
            all_valence_gt.extend(valence_gt)
            all_arousal_gt.extend(arousal_gt)
    
    return np.array(all_valence_pred), np.array(all_arousal_pred), \
           np.array(all_valence_gt), np.array(all_arousal_gt)

def compute_metrics(valence_pred, arousal_pred, valence_gt, arousal_gt):
    """Compute comprehensive evaluation metrics"""
    
    # Define metrics
    metrics = {
        'PCC': PCC,
        'CCC': CCC,
        'RMSE': RMSE,
        'SAGR': SAGR
    }
    
    # MSE function
    def MSE(gt, pred):
        return np.mean((gt - pred) ** 2)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Valence metrics
    print("\nVALENCE METRICS:")
    print("-" * 30)
    valence_results = {}
    for name, metric_fn in metrics.items():
        try:
            score = metric_fn(valence_gt, valence_pred)
            valence_results[name] = score
            print(f"{name:>8}: {score:.4f}")
        except Exception as e:
            print(f"{name:>8}: Error - {e}")
    
    # Add MSE
    valence_mse = MSE(valence_gt, valence_pred)
    valence_results['MSE'] = valence_mse
    print(f"{'MSE':>8}: {valence_mse:.4f}")
    
    # Arousal metrics
    print("\nAROUSAL METRICS:")
    print("-" * 30)
    arousal_results = {}
    for name, metric_fn in metrics.items():
        try:
            score = metric_fn(arousal_gt, arousal_pred)
            arousal_results[name] = score
            print(f"{name:>8}: {score:.4f}")
        except Exception as e:
            print(f"{name:>8}: Error - {e}")
    
    # Add MSE
    arousal_mse = MSE(arousal_gt, arousal_pred)
    arousal_results['MSE'] = arousal_mse
    print(f"{'MSE':>8}: {arousal_mse:.4f}")
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print("-" * 30)
    overall_results = {}
    for name in ['PCC', 'CCC', 'RMSE', 'MSE']:
        if name in valence_results and name in arousal_results:
            avg_score = (valence_results[name] + arousal_results[name]) / 2
            overall_results[name] = avg_score
            print(f"{name:>8}: {avg_score:.4f}")
    
    return valence_results, arousal_results, overall_results

def plot_results(valence_pred, arousal_pred, valence_gt, arousal_gt, output_dir):
    """Create evaluation plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Valence scatter plot
    axes[0].scatter(valence_gt, valence_pred, alpha=0.6, s=20)
    axes[0].plot([-1, 1], [-1, 1], 'r--', lw=2)
    axes[0].set_xlabel('Ground Truth Valence')
    axes[0].set_ylabel('Predicted Valence')
    axes[0].set_title('Valence Predictions vs Ground Truth')
    axes[0].grid(True, alpha=0.3)
    
    # Arousal scatter plot
    axes[1].scatter(arousal_gt, arousal_pred, alpha=0.6, s=20)
    axes[1].plot([-1, 1], [-1, 1], 'r--', lw=2)
    axes[1].set_xlabel('Ground Truth Arousal')
    axes[1].set_ylabel('Predicted Arousal')
    axes[1].set_title('Arousal Predictions vs Ground Truth')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Valence distributions
    axes[0, 0].hist(valence_gt, bins=50, alpha=0.7, label='Ground Truth', density=True)
    axes[0, 0].hist(valence_pred, bins=50, alpha=0.7, label='Predictions', density=True)
    axes[0, 0].set_xlabel('Valence')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Valence Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Arousal distributions
    axes[0, 1].hist(arousal_gt, bins=50, alpha=0.7, label='Ground Truth', density=True)
    axes[0, 1].hist(arousal_pred, bins=50, alpha=0.7, label='Predictions', density=True)
    axes[0, 1].set_xlabel('Arousal')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Arousal Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distributions
    valence_error = valence_pred - valence_gt
    arousal_error = arousal_pred - arousal_gt
    
    axes[1, 0].hist(valence_error, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_xlabel('Valence Error (Pred - GT)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Valence Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(arousal_error, bins=50, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Arousal Error (Pred - GT)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Arousal Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEvaluation plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Evaluate HSV-based EMONET model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--manifest_file', type=str, default='features_hsv_manifest.json',
                        help='Path to HSV features manifest file')
    parser.add_argument('--data_root', type=str, default='.',
                        help='Root directory for data')
    parser.add_argument('--model_type', type=str, choices=['feature', 'full'], default='feature',
                        help='Type of model to evaluate: feature-based or full EMONET')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Run evaluation
    if args.model_type == 'feature':
        valence_pred, arousal_pred, valence_gt, arousal_gt = evaluate_feature_model(
            args.model_path, args.manifest_file, args.data_root, device, args.batch_size
        )
    else:
        valence_pred, arousal_pred, valence_gt, arousal_gt = evaluate_full_model(
            args.model_path, args.manifest_file, args.data_root, device, args.batch_size
        )
    
    # Compute metrics
    valence_results, arousal_results, overall_results = compute_metrics(
        valence_pred, arousal_pred, valence_gt, arousal_gt
    )
    
    # Create plots
    plot_results(valence_pred, arousal_pred, valence_gt, arousal_gt, args.output_dir)
    
    # Convert numpy values to regular Python floats for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save results
    results = {
        'valence_metrics': valence_results,
        'arousal_metrics': arousal_results,
        'overall_metrics': overall_results,
        'model_path': args.model_path,
        'model_type': args.model_type,
        'num_samples': len(valence_pred)
    }
    
    # Convert to JSON-serializable format
    results = convert_to_json_serializable(results)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to {output_dir}/")
    print(f"Number of samples evaluated: {len(valence_pred)}")

if __name__ == '__main__':
    main()
