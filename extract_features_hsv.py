#!/usr/bin/env python3
"""
Extract EMONET features from AFEW-VA dataset using HSV color space
"""

import os
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from skimage import io

# Import EMONET model
from emonet.models.emonet import EmoNet

# Configuration
N_CLASSES = 8  # Use 8-class model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_ROOT = Path(os.environ.get('AFEW_VA_ROOT', "/home/ksubramanian/.cache/kagglehub/datasets/hoanguyensgu/afew-va/versions/1/AFEW-VA"))

def rgb_to_hsv_tensor(rgb_tensor):
    """
    Convert RGB tensor to HSV tensor
    
    Args:
        rgb_tensor: Tensor of shape (C, H, W) with values in [0, 1]
    
    Returns:
        hsv_tensor: Tensor of shape (C, H, W) with HSV values
    """
    # Convert tensor to numpy for OpenCV processing
    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    rgb_np = (rgb_np * 255).astype(np.uint8)  # Convert to 0-255 range
    
    # Convert RGB to HSV using OpenCV
    hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    
    # Normalize HSV values to [0, 1] range
    # H: 0-179 -> 0-1, S: 0-255 -> 0-1, V: 0-255 -> 0-1
    hsv_np = hsv_np.astype(np.float32)
    hsv_np[:, :, 0] /= 179.0  # Hue
    hsv_np[:, :, 1] /= 255.0  # Saturation
    hsv_np[:, :, 2] /= 255.0  # Value
    
    # Convert back to tensor
    hsv_tensor = torch.from_numpy(hsv_np).permute(2, 0, 1)  # (C, H, W)
    
    return hsv_tensor

def load_and_preprocess_image_hsv(image_path, target_size=256):
    """
    Load image and convert to HSV color space with preprocessing
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
    
    Returns:
        hsv_tensor: Preprocessed HSV image tensor
    """
    try:
        # Load image
        image = io.imread(str(image_path))
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        image = cv2.resize(image, (target_size, target_size))
        
        # Convert to tensor and normalize to [0, 1]
        rgb_tensor = torch.from_numpy(image).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1)  # (C, H, W)
        
        # Convert RGB to HSV
        hsv_tensor = rgb_to_hsv_tensor(rgb_tensor)
        
        # Add batch dimension
        hsv_tensor = hsv_tensor.unsqueeze(0)  # (1, C, H, W)
        
        return hsv_tensor
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def extract_features_hsv():
    """
    Extract EMONET features from AFEW-VA dataset using HSV color space
    """
    print(f"Using device: {DEVICE}")
    print(f"Dataset root: {DATASET_ROOT}")
    
    # Load pretrained EMONET model
    state_dict_path = Path(__file__).parent / 'pretrained' / f'emonet_{N_CLASSES}.pth'
    
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {state_dict_path}")
    
    print(f"Loading model from: {state_dict_path}")
    
    # Create model
    model = EmoNet(n_expression=N_CLASSES)
    
    # Load pretrained weights
    checkpoint = torch.load(state_dict_path, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
    
    print(f"Loaded pretrained weights from {state_dict_path}")
    
    model = model.to(DEVICE)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Create output directories
    features_dir = Path('features_hsv')
    features_dir.mkdir(exist_ok=True)
    
    # Initialize manifest
    manifest = {}
    
    # Process each video directory
    video_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir() and d.name.isdigit()])
    
    total_processed = 0
    total_errors = 0
    
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_id = video_dir.name
        
        # Create output directory for this video
        output_video_dir = features_dir / video_id
        output_video_dir.mkdir(exist_ok=True)
        
        # Get all image files in this video directory
        image_files = sorted(list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png')))
        
        if not image_files:
            print(f"No images found in {video_dir}")
            continue
        
        # Process each image in the video
        for image_file in tqdm(image_files, desc=f"Video {video_id}", leave=False):
            try:
                # Load and preprocess image in HSV
                hsv_tensor = load_and_preprocess_image_hsv(image_file)
                
                if hsv_tensor is None:
                    total_errors += 1
                    continue
                
                hsv_tensor = hsv_tensor.to(DEVICE)
                
                # Extract features using EMONET
                with torch.no_grad():
                    # Forward pass through the model
                    outputs = model(hsv_tensor)
                    
                    # Extract features from the penultimate layer
                    # This should be the 256-dimensional feature vector
                    if isinstance(outputs, dict):
                        # If model returns a dictionary, extract the feature vector
                        if 'emo_feat_2' in outputs:
                            features = outputs['emo_feat_2']
                        else:
                            # Use the last layer before final predictions
                            features = list(outputs.values())[-1]
                    else:
                        features = outputs
                    
                    # If features have multiple dimensions, take the feature representation
                    if len(features.shape) > 2:
                        features = torch.mean(features, dim=(2, 3))  # Global average pooling
                    
                    # Take the first 256 dimensions as features (before final classification)
                    if features.shape[1] > 256:
                        features = features[:, :256]
                    
                    features = features.cpu().numpy().squeeze()  # Remove batch dimension
                
                # Save features
                output_file = output_video_dir / f"{image_file.stem}.npz"
                np.savez_compressed(output_file, feature=features)
                
                # Add to manifest (you'll need to add arousal/valence labels)
                # For now, using placeholder values - replace with actual labels
                manifest[f"features_hsv/{video_id}/{image_file.stem}.npz"] = {
                    "arousal": 0.0,  # Replace with actual arousal value
                    "valence": 0.0   # Replace with actual valence value
                }
                
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                total_errors += 1
                continue
    
    # Save manifest
    manifest_file = Path('features_hsv_manifest.json')
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nFeature extraction completed!")
    print(f"Total processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Features saved to: {features_dir}")
    print(f"Manifest saved to: {manifest_file}")
    print(f"Feature dimension: {features.shape if 'features' in locals() else 'Unknown'}")

def copy_labels_from_rgb_manifest():
    """
    Copy arousal/valence labels from the original RGB manifest to HSV manifest
    """
    rgb_manifest_file = Path('features_manifest.json')
    hsv_manifest_file = Path('features_hsv_manifest.json')
    
    if not rgb_manifest_file.exists():
        print(f"RGB manifest not found: {rgb_manifest_file}")
        return
    
    if not hsv_manifest_file.exists():
        print(f"HSV manifest not found: {hsv_manifest_file}")
        return
    
    # Load both manifests
    with open(rgb_manifest_file, 'r') as f:
        rgb_manifest = json.load(f)
    
    with open(hsv_manifest_file, 'r') as f:
        hsv_manifest = json.load(f)
    
    # Copy labels from RGB to HSV manifest
    updated_count = 0
    for hsv_path, hsv_data in hsv_manifest.items():
        # Convert HSV path to RGB path
        rgb_path = hsv_path.replace('features_hsv/', 'features/')
        
        if rgb_path in rgb_manifest:
            hsv_data['arousal'] = rgb_manifest[rgb_path]['arousal']
            hsv_data['valence'] = rgb_manifest[rgb_path]['valence']
            updated_count += 1
    
    # Save updated HSV manifest
    with open(hsv_manifest_file, 'w') as f:
        json.dump(hsv_manifest, f, indent=2)
    
    print(f"Updated {updated_count} entries in HSV manifest with RGB labels")

if __name__ == "__main__":
    print("Extracting EMONET features from AFEW-VA dataset using HSV color space...")
    extract_features_hsv()
    
    print("\nCopying labels from RGB manifest...")
    copy_labels_from_rgb_manifest()
    
    print("\nHSV feature extraction completed!")
    print("You can now train using: python train_emonet_features.py --manifest_file features_hsv_manifest.json --data_root .")
