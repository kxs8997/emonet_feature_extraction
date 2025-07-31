#!/usr/bin/env python3
"""
Training script for finetuning EMONET on AFEW-VA dataset
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

# Import EMONET model and dataset
from emonet.models.emonet import EmoNet
from emonet.data.afew_va import AFEWVA, create_afew_va_annotations


class ConcordanceCorrelationCoefficient(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) loss function
    Commonly used for valence/arousal regression
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
        
        ccc = numerator / (denominator + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Return 1 - CCC as loss (we want to maximize CCC, so minimize 1-CCC)
        return 1 - ccc


class EmoNetTrainer:
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
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
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
            images = batch['image'].to(self.device)
            arousal_targets = batch['arousal'].to(self.device)
            valence_targets = batch['valence'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Extract valence and arousal predictions
            # Assuming the model outputs [expression_logits, valence, arousal]
            # The last 2 outputs are valence and arousal
            predictions = outputs['emo_feat_2']  # This should be the final emotion features
            
            # If the model outputs multiple things, we need to extract the regression outputs
            # Based on the model architecture, the last n_reg outputs are valence/arousal
            valence_pred = predictions[:, -2]  # Second to last output
            arousal_pred = predictions[:, -1]   # Last output
            
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
                images = batch['image'].to(self.device)
                arousal_targets = batch['arousal'].to(self.device)
                valence_targets = batch['valence'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                predictions = outputs['emo_feat_2']
                
                valence_pred = predictions[:, -2]
                arousal_pred = predictions[:, -1]
                
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
        plt.plot([lr_group['lr'] for lr_group in self.optimizer.param_groups], label='Learning Rate')
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
            if epoch % 10 == 0:
                self.plot_training_curves()
        
        self.logger.info(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")


def modify_model_for_training(model, freeze_backbone=True):
    """
    Modify EMONET model for training
    
    Args:
        model: EMONET model
        freeze_backbone: If True, freeze the backbone and only train the emotion head
    """
    # Enable gradients for emotion network
    for name, param in model.named_parameters():
        if 'emo_' in name:  # Emotion-related parameters
            param.requires_grad = True
        elif not freeze_backbone:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Finetune EMONET on AFEW-VA')
    
    # Dataset arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to AFEW-VA dataset root')
    parser.add_argument('--manifest_file', type=str, default=None,
                        help='Path to features_manifest.json')
    
    # Model arguments
    parser.add_argument('--pretrained_path', type=str, 
                        default='pretrained/emonet_8.pth',
                        help='Path to pretrained EMONET weights')
    parser.add_argument('--n_expression', type=int, default=8,
                        help='Number of expression classes (5 or 8)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone and only train emotion head')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
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
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Training split ratio')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
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
    
    # Create annotations file if needed
    if not (Path(args.data_root) / 'annotations.json').exists():
        print("Creating annotations file from manifest...")
        create_afew_va_annotations(args.data_root, args.manifest_file)
    
    # Create dataset
    print("Loading dataset...")
    full_dataset = AFEWVA(args.data_root)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(args.train_split * total_size)
    val_size = total_size - train_size
    
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
    
    # Load model
    print("Loading EMONET model...")
    model = EmoNet(n_expression=args.n_expression)
    
    # Load pretrained weights
    if os.path.exists(args.pretrained_path):
        print(f"Loading pretrained weights from {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        print(f"Warning: Pretrained weights not found at {args.pretrained_path}")
    
    # Modify model for training
    model = modify_model_for_training(model, args.freeze_backbone)
    model = model.to(device)
    
    # Create trainer and start training
    trainer = EmoNetTrainer(model, train_loader, val_loader, device, args)
    trainer.train()


if __name__ == '__main__':
    main()
