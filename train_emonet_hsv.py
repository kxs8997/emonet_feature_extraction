#!/usr/bin/env python3
"""
Training script for finetuning EMONET on AFEW-VA dataset with runtime RGB→HSV conversion
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime
import cv2
from skimage import io
import face_alignment

# Import EMONET model
from emonet.models.emonet import EmoNet


def rgb_to_hsv_tensor(rgb_tensor):
    """
    Convert RGB tensor to HSV tensor on GPU
    
    Args:
        rgb_tensor: Tensor of shape (B, C, H, W) with values in [0, 1]
    
    Returns:
        hsv_tensor: Tensor of shape (B, C, H, W) with HSV values
    """
    batch_size = rgb_tensor.shape[0]
    device = rgb_tensor.device
    
    # Convert each image in the batch
    hsv_batch = []
    
    for i in range(batch_size):
        # Get single image
        rgb_img = rgb_tensor[i]  # (C, H, W)
        
        # Convert tensor to numpy for OpenCV processing
        rgb_np = rgb_img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
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
        hsv_batch.append(hsv_tensor)
    
    # Stack batch
    hsv_batch = torch.stack(hsv_batch).to(device)
    
    return hsv_batch


class AFEWVADatasetHSV(Dataset):
    """
    AFEW-VA Dataset loader with runtime RGB→HSV conversion
    """
    
    def __init__(self, root_path, annotations_file=None, subset='train', 
                 transform=None, face_detector=None, image_size=256):
        """
        Args:
            root_path: Path to AFEW-VA dataset root
            annotations_file: Path to annotations JSON file
            subset: 'train', 'val', or 'test'
            transform: Optional transform to be applied on a sample
            face_detector: Face alignment detector
            image_size: Target image size for face crops
        """
        self.root_path = Path(root_path)
        self.subset = subset
        self.transform = transform
        self.image_size = image_size
        
        # Initialize face detector
        if face_detector is None:
            try:
                # Try newer face_alignment API
                self.face_detector = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D, 
                    flip_input=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            except AttributeError:
                try:
                    # Try older face_alignment API
                    self.face_detector = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType._2D, 
                        flip_input=False,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                except AttributeError:
                    # Fallback - disable face detection
                    print("Warning: Could not initialize face detector, using simple resize")
                    self.face_detector = None
        else:
            self.face_detector = face_detector
        
        # Load annotations
        if annotations_file is None:
            annotations_file = self.root_path / 'annotations.json'
        
        # If annotations.json doesn't exist, create it from features_manifest.json
        if not annotations_file.exists():
            self._create_annotations_from_manifest()
            
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Convert manifest entries to image paths and filter for existing files
        self.samples = []
        dataset_root = Path(os.environ.get('AFEW_VA_ROOT', 
            "/home/ksubramanian/.cache/kagglehub/datasets/hoanguyensgu/afew-va/versions/1/AFEW-VA"))
        
        for img_path, data in self.annotations.items():
            # Convert feature path to image path if needed
            if 'features/' in img_path:
                img_path = img_path.replace('features/', '').replace('.npz', '.png')  # AFEW-VA uses .png files
            else:
                # Convert .jpg to .png for AFEW-VA dataset
                img_path = img_path.replace('.jpg', '.png')
            
            full_img_path = dataset_root / img_path
            if full_img_path.exists():
                self.samples.append((str(full_img_path), data))
            else:
                # Try .jpg as fallback
                jpg_path = img_path.replace('.png', '.jpg')
                full_jpg_path = dataset_root / jpg_path
                if full_jpg_path.exists():
                    self.samples.append((str(full_jpg_path), data))
        
        print(f"Loaded {len(self.samples)} image samples for runtime HSV conversion")
    
    def _create_annotations_from_manifest(self):
        """Create annotations.json from features_manifest.json"""
        manifest_file = self.root_path / 'features_manifest.json'
        if not manifest_file.exists():
            raise FileNotFoundError(f"Neither annotations.json nor features_manifest.json found in {self.root_path}")
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        annotations = {}
        for feature_path, labels in manifest.items():
            # Convert feature path to image path
            img_path = feature_path.replace('features/', '').replace('.npz', '.jpg')
            annotations[img_path] = labels
        
        # Save annotations
        annotations_file = self.root_path / 'annotations.json'
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Created annotations.json with {len(annotations)} entries")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, annotation = self.samples[idx]
        
        # Load image
        image = io.imread(img_path)
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Detect face and crop
        face_crop = self._detect_and_crop_face(image)
        
        # Convert to tensor and normalize to [0, 1]
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
            'image': face_crop,  # RGB image - will be converted to HSV in training loop
            'arousal': arousal,
            'valence': valence,
            'image_path': img_path
        }
    
    def _detect_and_crop_face(self, image):
        """Detect face and return cropped face region"""
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


class ConcordanceCorrelationCoefficient(nn.Module):
    """Concordance Correlation Coefficient (CCC) loss function"""
    def __init__(self):
        super(ConcordanceCorrelationCoefficient, self).__init__()
    
    def forward(self, predictions, targets):
        pred_mean = torch.mean(predictions)
        target_mean = torch.mean(targets)
        
        pred_var = torch.var(predictions)
        target_var = torch.var(targets)
        covariance = torch.mean((predictions - pred_mean) * (targets - target_mean))
        
        numerator = 2 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
        
        ccc = numerator / (denominator + 1e-8)
        return 1 - ccc


class EmoNetHSVTrainer:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        
        # Loss functions
        if args.loss_function == 'mse':
            self.criterion = nn.MSELoss()
        elif args.loss_function == 'mae':
            self.criterion = nn.L1Loss()
        elif args.loss_function == 'ccc':
            self.criterion = ConcordanceCorrelationCoefficient()
        
        # Optimizer
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.args.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f'training_hsv_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self):
        """Train for one epoch with runtime RGB→HSV conversion"""
        self.model.train()
        total_loss = 0.0
        arousal_loss = 0.0
        valence_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training (RGB→HSV)")
        for batch_idx, batch in enumerate(pbar):
            rgb_images = batch['image'].to(self.device)
            arousal_targets = batch['arousal'].to(self.device)
            valence_targets = batch['valence'].to(self.device)
            
            # Convert RGB to HSV on-the-fly
            hsv_images = rgb_to_hsv_tensor(rgb_images)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(hsv_images)
            
            # Extract valence and arousal predictions
            # Handle different model output formats
            if isinstance(outputs, dict):
                if 'emo_feat_2' in outputs:
                    predictions = outputs['emo_feat_2']
                else:
                    # Use the last layer in the dictionary
                    predictions = list(outputs.values())[-1]
            else:
                predictions = outputs
            
            # Extract valence and arousal from the last 2 dimensions
            # Handle different tensor shapes
            if len(predictions.shape) == 1:
                # If 1D tensor, assume it contains both valence and arousal
                if predictions.shape[0] >= 2:
                    valence_pred = predictions[-2].unsqueeze(0).expand(hsv_images.shape[0])
                    arousal_pred = predictions[-1].unsqueeze(0).expand(hsv_images.shape[0])
                else:
                    # Fallback: use the same value for both
                    valence_pred = predictions[0].unsqueeze(0).expand(hsv_images.shape[0])
                    arousal_pred = predictions[0].unsqueeze(0).expand(hsv_images.shape[0])
            elif len(predictions.shape) == 2:
                # Standard 2D tensor
                if predictions.shape[1] >= 2:
                    valence_pred = predictions[:, -2]  # Second to last output
                    arousal_pred = predictions[:, -1]   # Last output
                else:
                    # If only 1 output dimension, duplicate it
                    valence_pred = predictions[:, 0]
                    arousal_pred = predictions[:, 0]
            else:
                # Handle higher dimensional tensors by flattening
                predictions_flat = predictions.view(predictions.shape[0], -1)
                if predictions_flat.shape[1] >= 2:
                    valence_pred = predictions_flat[:, -2]
                    arousal_pred = predictions_flat[:, -1]
                else:
                    valence_pred = predictions_flat[:, 0]
                    arousal_pred = predictions_flat[:, 0]
            
            # Calculate losses
            v_loss = self.criterion(valence_pred, valence_targets)
            a_loss = self.criterion(arousal_pred, arousal_targets)
            loss = v_loss + a_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            arousal_loss += a_loss.item()
            valence_loss += v_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'V_Loss': f'{v_loss.item():.4f}',
                'A_Loss': f'{a_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_arousal_loss = arousal_loss / len(self.train_loader)
        avg_valence_loss = valence_loss / len(self.train_loader)
        
        return avg_loss, avg_valence_loss, avg_arousal_loss
    
    def validate(self):
        """Validate the model with runtime RGB→HSV conversion"""
        self.model.eval()
        total_loss = 0.0
        arousal_loss = 0.0
        valence_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation (RGB→HSV)"):
                rgb_images = batch['image'].to(self.device)
                arousal_targets = batch['arousal'].to(self.device)
                valence_targets = batch['valence'].to(self.device)
                
                # Convert RGB to HSV on-the-fly
                hsv_images = rgb_to_hsv_tensor(rgb_images)
                
                # Forward pass
                outputs = self.model(hsv_images)
                
                # Handle different model output formats
                if isinstance(outputs, dict):
                    if 'emo_feat_2' in outputs:
                        predictions = outputs['emo_feat_2']
                    else:
                        # Use the last layer in the dictionary
                        predictions = list(outputs.values())[-1]
                else:
                    predictions = outputs
                
                # Handle different tensor shapes
                if len(predictions.shape) == 1:
                    # If 1D tensor, assume it contains both valence and arousal
                    if predictions.shape[0] >= 2:
                        valence_pred = predictions[-2].unsqueeze(0).expand(hsv_images.shape[0])
                        arousal_pred = predictions[-1].unsqueeze(0).expand(hsv_images.shape[0])
                    else:
                        # Fallback: use the same value for both
                        valence_pred = predictions[0].unsqueeze(0).expand(hsv_images.shape[0])
                        arousal_pred = predictions[0].unsqueeze(0).expand(hsv_images.shape[0])
                elif len(predictions.shape) == 2:
                    # Standard 2D tensor
                    if predictions.shape[1] >= 2:
                        valence_pred = predictions[:, -2]  # Second to last output
                        arousal_pred = predictions[:, -1]   # Last output
                    else:
                        # If only 1 output dimension, duplicate it
                        valence_pred = predictions[:, 0]
                        arousal_pred = predictions[:, 0]
                else:
                    # Handle higher dimensional tensors by flattening
                    predictions_flat = predictions.view(predictions.shape[0], -1)
                    if predictions_flat.shape[1] >= 2:
                        valence_pred = predictions_flat[:, -2]
                        arousal_pred = predictions_flat[:, -1]
                    else:
                        valence_pred = predictions_flat[:, 0]
                        arousal_pred = predictions_flat[:, 0]
                
                # Calculate losses
                v_loss = self.criterion(valence_pred, valence_targets)
                a_loss = self.criterion(arousal_pred, arousal_targets)
                loss = v_loss + a_loss
                
                total_loss += loss.item()
                arousal_loss += a_loss.item()
                valence_loss += v_loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_arousal_loss = arousal_loss / len(self.val_loader)
        avg_valence_loss = valence_loss / len(self.val_loader)
        
        return avg_loss, avg_valence_loss, avg_arousal_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'args': self.args
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.args.output_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.args.output_dir) / 'best_model_hsv.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best HSV model saved with validation loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting HSV training with runtime RGB→HSV conversion...")
        self.logger.info(f"Total epochs: {self.args.epochs}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(1, self.args.epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{self.args.epochs}")
            
            # Train
            train_loss, train_v_loss, train_a_loss = self.train_epoch()
            
            # Validate
            val_loss, val_v_loss, val_a_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Log results
            self.logger.info(f"Train Loss: {train_loss:.4f} (V: {train_v_loss:.4f}, A: {train_a_loss:.4f})")
            self.logger.info(f"Val Loss: {val_loss:.4f} (V: {val_v_loss:.4f}, A: {val_a_loss:.4f})")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(epoch, is_best)
        
        self.logger.info(f"\nHSV training completed! Best validation loss: {self.best_val_loss:.4f}")


def modify_model_for_training(model, freeze_backbone=True):
    """Modify EMONET model for training"""
    for name, param in model.named_parameters():
        if 'emo_' in name:  # Emotion-related parameters
            param.requires_grad = True
        elif not freeze_backbone:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Finetune EMONET on AFEW-VA with runtime RGB→HSV conversion')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='.',
                        help='Path to AFEW-VA dataset root')
    
    # Model arguments
    parser.add_argument('--pretrained_path', type=str, 
                        default='pretrained/emonet_8.pth',
                        help='Path to pretrained EMONET weights')
    parser.add_argument('--n_expression', type=int, default=8,
                        help='Number of expression classes (5 or 8)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone and only train emotion head')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--loss_function', type=str, default='ccc',
                        choices=['mse', 'mae', 'ccc'], help='Loss function')
    
    # Data split arguments
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output_hsv',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataset
    print("Loading AFEW-VA dataset for runtime HSV conversion...")
    full_dataset = AFEWVADatasetHSV(args.data_root)
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(args.val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders (disable multiprocessing to avoid CUDA issues)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Load model
    print("Loading EMONET model...")
    model = EmoNet(n_expression=args.n_expression)
    
    # Load pretrained weights
    if os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        elif 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            # Assume the entire file is the state dict
            model.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully!")
    else:
        print(f"Warning: Pretrained weights not found at {args.pretrained_path}")
    
    # Modify model for training
    model = modify_model_for_training(model, args.freeze_backbone)
    model = model.to(device)
    
    # Create trainer and start training
    trainer = EmoNetHSVTrainer(model, train_loader, val_loader, device, args)
    trainer.train()


if __name__ == '__main__':
    main()
