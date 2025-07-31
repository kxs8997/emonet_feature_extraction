#!/usr/bin/env python3
"""
Training script for finetuning EMONET on AFEW-VA dataset using pre-extracted features
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


class AFEWVAFeaturesDataset(Dataset):
    """
    AFEW-VA Dataset loader for pre-extracted features
    """
    
    def __init__(self, root_path, manifest_file=None):
        """
        Args:
            root_path: Path to AFEW-VA dataset root
            manifest_file: Path to features_manifest.json
        """
        self.root_path = Path(root_path)
        
        # Load manifest
        if manifest_file is None:
            manifest_file = self.root_path / 'features_manifest.json'
        
        with open(manifest_file, 'r') as f:
            self.manifest = json.load(f)
        
        # Convert to list of samples
        self.samples = []
        for feature_path, labels in self.manifest.items():
            full_path = self.root_path / feature_path
            if full_path.exists():
                self.samples.append((str(full_path), labels))
        
        print(f"Loaded {len(self.samples)} feature samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feature_path, labels = self.samples[idx]
        
        # Load pre-extracted features
        features = np.load(feature_path)['feature']  # Shape: (256,)
        features = torch.from_numpy(features).float()
        
        # Get labels
        arousal = torch.tensor(labels['arousal'], dtype=torch.float32)
        valence = torch.tensor(labels['valence'], dtype=torch.float32)
        
        return {
            'features': features,
            'arousal': arousal,
            'valence': valence,
            'feature_path': feature_path
        }


class SimpleRegressionHead(nn.Module):
    """
    Simple regression head for valence/arousal prediction from features
    """
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=2):
        super(SimpleRegressionHead, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.regressor(x)


class ConcordanceCorrelationCoefficient(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) loss function
    """
    def __init__(self):
        super(ConcordanceCorrelationCoefficient, self).__init__()
    
    def forward(self, predictions, targets):
        # Calculate means
        pred_mean = torch.mean(predictions)
        target_mean = torch.mean(targets)
        
        # Calculate variances and covariance
        pred_var = torch.var(predictions)
        target_var = torch.var(targets)
        covariance = torch.mean((predictions - pred_mean) * (targets - target_mean))
        
        # Calculate CCC
        numerator = 2 * covariance
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
        
        ccc = numerator / (denominator + 1e-8)
        
        # Return 1 - CCC as loss (we want to maximize CCC)
        return 1 - ccc


class FeatureTrainer:
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
        else:
            raise ValueError(f"Unknown loss function: {args.loss_function}")
        
        # Optimizer
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
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
        log_file = log_dir / f'training_{timestamp}.log'
        
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
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        arousal_loss = 0.0
        valence_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            arousal_targets = batch['arousal'].to(self.device)
            valence_targets = batch['valence'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)  # Shape: (batch_size, 2)
            
            # Extract predictions
            valence_pred = outputs[:, 0]  # First output is valence
            arousal_pred = outputs[:, 1]  # Second output is arousal
            
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
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        arousal_loss = 0.0
        valence_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                features = batch['features'].to(self.device)
                arousal_targets = batch['arousal'].to(self.device)
                valence_targets = batch['valence'].to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                valence_pred = outputs[:, 0]
                arousal_pred = outputs[:, 1]
                
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
            best_path = Path(self.args.output_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        epochs = range(1, len(self.train_losses) + 1)
        current_lr = self.optimizer.param_groups[0]['lr']
        plt.plot(epochs, [current_lr] * len(epochs), label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(self.args.output_dir) / 'training_curves.png')
        plt.close()
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
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
            
            # Plot training curves
            if epoch % 5 == 0:
                self.plot_training_curves()
        
        self.logger.info(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train regression head on AFEW-VA features')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, default='.',
                        help='Path to AFEW-VA dataset root')
    parser.add_argument('--manifest_file', type=str, default='features_manifest.json',
                        help='Path to features_manifest.json')
    
    # Model arguments
    parser.add_argument('--input_dim', type=int, default=256,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer dimension')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping threshold (0 to disable)')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--loss_function', type=str, default='mse',
                        choices=['mse', 'mae', 'ccc'], help='Loss function')
    
    # Data split arguments
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output_features',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataset
    print("Loading feature dataset...")
    full_dataset = AFEWVAFeaturesDataset(args.data_root, args.manifest_file)
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(args.val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Create model
    print("Creating regression model...")
    model = SimpleRegressionHead(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=2  # valence and arousal
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer and start training
    trainer = FeatureTrainer(model, train_loader, val_loader, device, args)
    trainer.train()


if __name__ == '__main__':
    main()
