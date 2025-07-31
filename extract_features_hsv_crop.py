import os
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from skimage import io
import cv2

from emonet.models import EmoNet

# --- CONFIG ---
# Update these as needed
DATASET_ROOT = Path(os.environ.get('AFEW_VA_ROOT', "/home/ksubramanian/.cache/kagglehub/datasets/hoanguyensgu/afew-va/versions/1/AFEW-VA"))
FEATURES_ROOT = Path('features_hsv')
N_CLASSES = 8
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 256

def rgb_to_hsv_tensor(rgb_tensor):
    """
    Convert RGB tensor to HSV tensor
    
    Args:
        rgb_tensor: RGB tensor of shape (C, H, W) with values in [0, 1]
    
    Returns:
        hsv_tensor: HSV tensor of shape (C, H, W) with values in [0, 1]
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

# --- LOAD EMONET ---
print(f"Using device: {DEVICE}")
print(f"Dataset root: {DATASET_ROOT}")

state_dict_path = Path(__file__).parent / 'pretrained' / f'emonet_{N_CLASSES}.pth'

if not state_dict_path.exists():
    raise FileNotFoundError(f"Pretrained model not found: {state_dict_path}")

print(f"Loading model from: {state_dict_path}")

# Load pretrained weights with different checkpoint format handling
checkpoint = torch.load(str(state_dict_path), map_location='cpu')

# Handle different checkpoint formats
if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Remove 'module.' prefix if present
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# Create and load model
net = EmoNet(n_expression=N_CLASSES).to(DEVICE)
net.load_state_dict(state_dict, strict=False)
net.eval()

print(f"Loaded pretrained weights from {state_dict_path}")

# --- Identify feature layer ---
def extract_features_from_emonet(net, image_tensor):
    """Extracts features from the layer before the decision (FC) layer of EmoNet (output of avg_pool_2)."""
    features = {}
    def hook(module, input, output):
        # output shape: (batch, 256, 1, 1)
        features['pre_fc'] = output.detach().cpu().numpy()
    handle = net.avg_pool_2.register_forward_hook(hook)
    with torch.no_grad():
        _ = net(image_tensor.unsqueeze(0))
    handle.remove()
    # Flatten to (256,)
    return features['pre_fc'].reshape(-1)

# --- Main extraction loop ---
print("Starting HSV feature extraction from RGB face crops...")

manifest = {}
FEATURES_ROOT.mkdir(exist_ok=True)

video_folders = sorted([f for f in DATASET_ROOT.iterdir() if f.is_dir()])
total_processed = 0
total_errors = 0

for video_dir in video_folders:
    json_path = video_dir / f"{video_dir.name}.json"
    if not json_path.exists():
        print(f"No JSON metadata found for {video_dir.name}")
        continue
        
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    frames = metadata['frames']
    
    out_dir = FEATURES_ROOT / video_dir.name
    out_dir.mkdir(exist_ok=True)
    
    frame_ids = sorted(list(frames.keys()))
    print(f"Processing video {video_dir.name} with {len(frame_ids)} frames...")
    
    for frame_id in frame_ids:
        img_path = video_dir / f"{frame_id}.png"
        if not img_path.exists():
            # Try .jpg extension
            img_path = video_dir / f"{frame_id}.jpg"
            if not img_path.exists():
                print(f"Image not found: {frame_id}")
                total_errors += 1
                continue
        
        try:
            # Load image in RGB
            image_rgb = io.imread(str(img_path))
            
            # Ensure RGB format
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
                pass  # Already RGB
            elif len(image_rgb.shape) == 2:
                # Convert grayscale to RGB
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image_rgb[:, :, :3]  # Take first 3 channels
            
            # Resize to standard size (this is the "face crop" step - simplified)
            image_rgb = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            
            # Convert to tensor and normalize to [0, 1]
            rgb_tensor = torch.Tensor(image_rgb).permute(2, 0, 1) / 255.0
            
            # Convert RGB face crop to HSV
            hsv_tensor = rgb_to_hsv_tensor(rgb_tensor)
            
            # Move to device
            hsv_tensor = hsv_tensor.to(DEVICE)
            
            # Extract features from HSV face crop
            feat = extract_features_from_emonet(net, hsv_tensor)
            
            # Save feature
            feat_path = out_dir / f"{frame_id}.npz"
            np.savez_compressed(feat_path, feature=feat)
            
            # Add to manifest (relative path to script's directory)
            rel_feat_path = os.path.relpath(str(feat_path), start=str(Path(__file__).parent))
            manifest[rel_feat_path] = {
                'arousal': frames[frame_id]['arousal'],
                'valence': frames[frame_id]['valence']
            }
            
            total_processed += 1
            
        except Exception as e:
            print(f"Error processing {video_dir.name}/{frame_id}: {e}")
            total_errors += 1
            continue

# Save manifest
manifest_path = 'features_hsv_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\nHSV Feature extraction completed!")
print(f"Total processed: {total_processed}")
print(f"Total errors: {total_errors}")
print(f"Saved features for {len(manifest)} frames")
print(f"Features directory: {FEATURES_ROOT}")
print(f"Manifest file: {manifest_path}")
