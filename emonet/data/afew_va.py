import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage import io
import cv2
from PIL import Image
import face_alignment


class AFEWVA(Dataset):
    """
    AFEW-VA Dataset loader for training/finetuning EMONET
    
    Expected directory structure:
    afew_va_root/
    ├── 001/
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    ├── 002/
    │   └── ...
    └── annotations.json  # Contains arousal/valence labels
    """
    
    def __init__(self, root_path, annotations_file=None, subset='train', 
                 transform=None, face_detector=None, image_size=256):
        """
        Args:
            root_path: Path to AFEW-VA dataset root
            annotations_file: Path to annotations JSON file (if None, looks for annotations.json in root)
            subset: 'train', 'val', or 'test'
            transform: Optional transform to be applied on a sample
            face_detector: Face alignment detector (if None, creates new one)
            image_size: Target image size for face crops
        """
        self.root_path = Path(root_path)
        self.subset = subset
        self.transform = transform
        self.image_size = image_size
        
        # Initialize face detector
        if face_detector is None:
            self.face_detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType._2D, 
                flip_input=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.face_detector = face_detector
        
        # Load annotations
        if annotations_file is None:
            annotations_file = self.root_path / 'annotations.json'
        
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter annotations for the specified subset if needed
        # Assuming annotations format: {"image_path": {"arousal": float, "valence": float, "subset": str}}
        self.samples = []
        for img_path, data in self.annotations.items():
            if 'subset' in data and data['subset'] == subset:
                self.samples.append((img_path, data))
            elif 'subset' not in data:  # If no subset info, include all
                self.samples.append((img_path, data))
        
        print(f"Loaded {len(self.samples)} samples for {subset} subset")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, annotation = self.samples[idx]
        
        # Load image
        full_img_path = self.root_path / img_path
        image = io.imread(str(full_img_path))
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Detect face and crop
        face_crop = self._detect_and_crop_face(image)
        
        # Convert to tensor and normalize
        if face_crop is not None:
            face_crop = face_crop.astype(np.float32) / 255.0
            face_crop = torch.from_numpy(face_crop).permute(2, 0, 1)  # HWC -> CHW
        else:
            # Fallback: resize original image if face detection fails
            face_crop = cv2.resize(image, (self.image_size, self.image_size))
            face_crop = face_crop.astype(np.float32) / 255.0
            face_crop = torch.from_numpy(face_crop).permute(2, 0, 1)
        
        # Apply transforms if provided
        if self.transform:
            face_crop = self.transform(face_crop)
        
        # Get labels
        arousal = torch.tensor(annotation['arousal'], dtype=torch.float32)
        valence = torch.tensor(annotation['valence'], dtype=torch.float32)
        
        return {
            'image': face_crop,
            'arousal': arousal,
            'valence': valence,
            'image_path': img_path
        }
    
    def _detect_and_crop_face(self, image):
        """
        Detect face and return cropped face region
        """
        try:
            # Detect landmarks
            landmarks = self.face_detector.get_landmarks(image)
            
            if landmarks is None or len(landmarks) == 0:
                return None
            
            # Use first detected face
            landmarks = landmarks[0]
            
            # Calculate bounding box from landmarks
            x_min, y_min = np.min(landmarks, axis=0).astype(int)
            x_max, y_max = np.max(landmarks, axis=0).astype(int)
            
            # Add padding
            padding = 0.3
            width = x_max - x_min
            height = y_max - y_min
            
            x_min = max(0, int(x_min - padding * width))
            y_min = max(0, int(y_min - padding * height))
            x_max = min(image.shape[1], int(x_max + padding * width))
            y_max = min(image.shape[0], int(y_max + padding * height))
            
            # Crop face
            face_crop = image[y_min:y_max, x_min:x_max]
            
            # Resize to target size
            face_crop = cv2.resize(face_crop, (self.image_size, self.image_size))
            
            return face_crop
            
        except Exception as e:
            print(f"Face detection failed: {e}")
            return None


def create_afew_va_annotations(dataset_root, manifest_file=None):
    """
    Create annotations file from features_manifest.json
    
    Args:
        dataset_root: Path to AFEW-VA dataset
        manifest_file: Path to features_manifest.json (if None, looks in dataset_root)
    """
    if manifest_file is None:
        manifest_file = Path(dataset_root) / 'features_manifest.json'
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    annotations = {}
    for feature_path, labels in manifest.items():
        # Convert feature path to image path
        # e.g., "features/001/00000.npz" -> "001/00000.jpg"
        img_path = feature_path.replace('features/', '').replace('.npz', '.jpg')
        
        annotations[img_path] = {
            'arousal': labels['arousal'],
            'valence': labels['valence']
        }
    
    # Save annotations
    output_file = Path(dataset_root) / 'annotations.json'
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created annotations file: {output_file}")
    print(f"Total samples: {len(annotations)}")
    
    return output_file


if __name__ == "__main__":
    # Example usage
    dataset_root = "/path/to/afew_va"
    
    # Create annotations file from manifest
    create_afew_va_annotations(dataset_root)
    
    # Create dataset
    dataset = AFEWVA(dataset_root)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Arousal: {sample['arousal']}, Valence: {sample['valence']}")
