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
FEATURES_ROOT = Path('features')
N_CLASSES = 8
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 256
NUM_VIDEOS = None  # None means process all videos
FRAMES_PER_VIDEO = None  # None means process all frames per video

# --- LOAD EMONET ---
state_dict_path = Path(__file__).parent / 'pretrained' / f'emonet_{N_CLASSES}.pth'
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
net = EmoNet(n_expression=N_CLASSES).to(DEVICE)
net.load_state_dict(state_dict, strict=False)
net.eval()

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
manifest = {}
FEATURES_ROOT.mkdir(exist_ok=True)

video_folders = sorted([f for f in DATASET_ROOT.iterdir() if f.is_dir()])[:NUM_VIDEOS]
for video_dir in video_folders:
    json_path = video_dir / f"{video_dir.name}.json"
    if not json_path.exists():
        continue
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    frames = metadata['frames']
    out_dir = FEATURES_ROOT / video_dir.name
    out_dir.mkdir(exist_ok=True)
    frame_ids = sorted(list(frames.keys()))[:FRAMES_PER_VIDEO]
    for frame_id in frame_ids:
        img_path = video_dir / f"{frame_id}.png"
        if not img_path.exists():
            continue
        # Load image
        image_rgb = io.imread(str(img_path))[:, :, :3]
        image_rgb = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        image_tensor = torch.Tensor(image_rgb).permute(2, 0, 1).to(DEVICE) / 255.0
        # Extract features
        feat = extract_features_from_emonet(net, image_tensor)
        # Save feature
        feat_path = out_dir / f"{frame_id}.npz"
        np.savez_compressed(feat_path, feature=feat)
        # Add to manifest (relative path to script's directory)
        rel_feat_path = os.path.relpath(str(feat_path), start=str(Path(__file__).parent))
        manifest[rel_feat_path] = {
            'arousal': frames[frame_id]['arousal'],
            'valence': frames[frame_id]['valence']
        }
# Save manifest
with open('features_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"Done. Saved features for {len(manifest)} frames. Manifest: features_manifest.json")
