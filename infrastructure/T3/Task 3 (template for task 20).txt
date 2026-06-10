# task20_cnn_scratch/train_cnn.py
"""
Custom 1D CNN training from scratch for promoter prediction
Owner: Yany
Due: 26 Jun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from tqdm import tqdm
import wandb
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DNA One-hot Encoding
# ============================================================================
class DNAEncoder:
    """Convert DNA sequences to one-hot encoded tensors"""
    
    def __init__(self, max_length=300):
        self.max_length = max_length
        self.nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.num_channels = 5  # A, C, G, T, N
        
    def encode(self, sequences):
        """Encode batch of sequences to one-hot"""
        batch_size = len(sequences)
        encoded = np.zeros((batch_size, self.num_channels, self.max_length), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            seq = seq[:self.max_length].upper()
            for j, char in enumerate(seq):
                if char in self.nucleotide_to_int:
                    idx = self.nucleotide_to_int[char]
                    encoded[i, idx, j] = 1.0
                else:
                    encoded[i, 4, j] = 1.0  # 'N' or unknown base
                    
        return torch.FloatTensor(encoded)

# ============================================================================
# CNN Architecture - Multiple variants for hyperparameter search
# ============================================================================
class PromoterCNN(nn.Module):
    """Custom 1D CNN for promoter prediction"""
    
    def __init__(self, 
                 input_channels=5,
                 sequence_length=300,
                 num_filters=[64, 128, 256],
                 kernel_sizes=[7, 5, 3],
                 dropout_rate=0.3,
                 use_batch_norm=True,
                 use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            block = nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout_rate)
            )
            self.conv_blocks.append(block)
            in_channels = filters
        
        # Calculate output size after convolutions and pooling
        self._calculate_output_size(sequence_length, kernel_sizes, num_filters)
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.flattened_size,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # Binary classification
        )
        
    def _calculate_output_size(self, seq_len, kernel_sizes, num_filters):
        """Calculate flattened size after convolutions"""
        size = seq_len
        for kernel_size in kernel_sizes:
            # After convolution (padding keeps size)
            # After maxpool with stride 2
            size = size // 2
        
        self.flattened_size = num_filters[-1] * size
        
    def forward(self, x):
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Prepare for attention
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Apply attention (optional)
        if self.use_attention:
            x_attn = x_flat.unsqueeze(1)  # Add sequence dimension
            x_attn, _ = self.attention(x_attn, x_attn, x_attn)
            x = x_attn.squeeze(1)
        else:
            x = x_flat
        
        # Fully connected layers
        x = self.fc_layers(x)
        return x

class DilatedCNN(nn.Module):
    """Dilated CNN for capturing longer-range dependencies"""
    
    def __init__(self, input_channels=5, sequence_length=300, dropout_rate=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, dilation=4, padding=4)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, dilation=8, padding=8)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResidualCNN(nn.Module):
    """Residual CNN with skip connections"""
    
    def __init__(self, input_channels=5, sequence_length=300, dropout_rate=0.3):
        super().__init__()
        
        self.initial_conv = nn.Conv1d(input_channels, 64, kernel_size=7, padding=3)
        
        # Residual blocks
        self.res_block1 = self._make_res_block(64, 128, dropout_rate)
        self.res_block2 = self._make_res_block(128, 256, dropout_rate)
        self.res_block3 = self._make_res_block(256, 512, dropout_rate)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        
    def _make_res_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        
        # Residual connections
        identity = x
        x = self.res_block1(x)
        if x.shape[1] == identity.shape[1]:
            x = x + identity[:, :x.shape[1], :x.shape[2]]
        
        identity = x
        x = self.res_block2(x)
        x = x + identity[:, :x.shape[1], :x.shape[2]]
        
        identity = x
        x = self.res_block3(x)
        x = x + identity[:, :x.shape[1], :x.shape[2]]
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# ============================================================================
# Dataset and Training Utilities
# ============================================================================
class CNNDataset(Dataset):
    """Dataset for CNN training"""
    
    def __init__(self, sequences, labels, encoder, augment=False):
        self.sequences = sequences
        self.labels = labels
        self.encoder = encoder
        self.augment = augment
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Data augmentation
        if self.augment and np.random.random() > 0.7:
            seq = self._augment_sequence(seq)
        
        # Encode
        encoded = self.encoder.encode([seq])
        
        return encoded.squeeze(0), torch.tensor(label, dtype=torch.long)
    
    def _augment_sequence(self, seq):
        """Reverse complement or random mutation"""
        if np.random.random() > 0.5:
            # Reverse complement
            complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
            seq = ''.join(complement.get(base, base) for base in seq[::-1])
        else:
            # Random mutation (1% of bases)
            seq_list = list(seq)
            num_mutations = max(1, int(len(seq) * 0.01))
            for _ in range(num_mutations):
                pos = np.random.randint(len(seq_list))
                seq_list[pos] = np.random.choice(['A', 'C', 'G', 'T'])
            seq = ''.join(seq_list)
        return seq

class FocalLoss(nn.Module):
    """Focal loss for imbalanced data"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================================
# Training Functions
# ============================================================================
class CNNTrainer:
    """Trainer for CNN models"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        
        # Loss function
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        return total_loss / len(train_loader), accuracy, f1
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='binary')
        auroc = roc_auc_score(all_labels, all_probs)
        
        return val_loss / len(val_loader), accuracy, f1, auroc
    
    def train(self, train_loader, val_loader, epochs, early_stopping_patience=10):
        """Full training loop"""
        best_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auroc': []}
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1, val_auroc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val AUROC: {val_auroc:.4f}")
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            history['val_auroc'].append(val_auroc)
            
            # Wandb logging
            if self.config.use_wandb:
                wandb.log({
                    'train_loss': train_loss,
                    'train_f1': train_f1,
                    'val_loss': val_loss,
                    'val_f1': val_f1,
                    'val_auroc': val_auroc,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
            
            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f"{self.config.output_dir}/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(f"{self.config.output_dir}/best_model.pt"))
        
        return history, best_f1

# ============================================================================
# Hyperparameter Optimization
# ============================================================================
@dataclass
class CNNConfig:
    model_architecture: str = "standard"  # standard, dilated, residual
    num_filters: List[int] = None
    kernel_sizes: List[int] = None
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    use_attention: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 50
    use_wandb: bool = True
    output_dir: str = "models/cnn_promoter"

def objective(trial, train_loader, val_loader, device, config):
    """Optuna objective for CNN hyperparameters"""
    
    # Suggest hyperparameters
    config.num_filters = trial.suggest_categorical('num_filters', [
        [32, 64, 128], [64, 128, 256], [128, 256, 512]
    ])
    config.kernel_sizes = trial.suggest_categorical('kernel_sizes', [
        [3, 3, 3], [5, 3, 3], [7, 5, 3]
    ])
    config.dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    config.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    config.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    config.use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    config.use_attention = trial.suggest_categorical('use_attention', [True, False])
    
    # Create model based on architecture
    if config.model_architecture == "standard":
        model = PromoterCNN(
            input_channels=5,
            sequence_length=300,
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
            use_attention=config.use_attention
        )
    elif config.model_architecture == "dilated":
        model = DilatedCNN(dropout_rate=config.dropout_rate)
    else:  # residual
        model = ResidualCNN(dropout_rate=config.dropout_rate)
    
    # Create trainer
    trainer = CNNTrainer(model, device, config)
    
    # Train for fewer epochs in trials
    history, best_f1 = trainer.train(
        train_loader, val_loader, 
        epochs=20,  # Fewer epochs for trials
        early_stopping_patience=5
    )
    
    return best_f1

# ============================================================================
# Main execution
# ============================================================================
def main():
    # Load data
    logger.info("Loading training data...")
    train_df = pd.read_parquet("data/train_promoters.parquet")
    val_df = pd.read_parquet("data/val_promoters.parquet")
    test_df = pd.read_parquet("data/test_promoters.parquet")
    
    # Initialize encoder
    encoder = DNAEncoder(max_length=300)
    
    # Create datasets
    train_dataset = CNNDataset(
        train_df['sequence'].tolist(),
        train_df['is_promoter'].tolist(),
        encoder,
        augment=True
    )
    val_dataset = CNNDataset(
        val_df['sequence'].tolist(),
        val_df['is_promoter'].tolist(),
        encoder,
        augment=False
    )
    test_dataset = CNNDataset(
        test_df['sequence'].tolist(),
        test_df['is_promoter'].tolist(),
        encoder,
        augment=False
    )
    
    # Calculate class weights for balanced sampling
    train_labels = train_df['is_promoter'].tolist()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    config = CNNConfig()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Hyperparameter optimization for each architecture
    architectures = ["standard", "dilated", "residual"]
    best_results = {}
    
    for arch in architectures:
        logger.info(f"\n{'='*50}")
        logger.info(f"Optimizing {arch.upper()} CNN architecture")
        logger.info(f"{'='*50}")
        
        config.model_architecture = arch
        
        # Initialize wandb for this architecture
        if config.use_wandb:
            wandb.init(project="cnn_promoter", name=f"{arch}_optimization", config=vars(config))
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f'cnn_{arch}_optimization',
            storage=f'sqlite:///cnn_{arch}_optuna.db',
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader, device, config),
            n_trials=30,
            n_jobs=1
        )
        
        # Train best model for this architecture
        logger.info(f"Training best {arch} model...")
        best_params = study.best_params
        
        # Update config with best params
        for key, value in best_params.items():
            setattr(config, key, value)
        
        # Create final model
        if arch == "standard":
            model = PromoterCNN(
                input_channels=5,
                sequence_length=300,
                num_filters=config.num_filters,
                kernel_sizes=config.kernel_sizes,
                dropout_rate=config.dropout_rate,
                use_batch_norm=config.use_batch_norm,
                use_attention=config.use_attention
            )
        elif arch == "dilated":
            model = DilatedCNN(dropout_rate=config.dropout_rate)
        else:
            model = ResidualCNN(dropout_rate=config.dropout_rate)
        
        # Create trainer
        trainer = CNNTrainer(model, device, config)
        
        # Train on full training+validation
        full_dataset = CNNDataset(
            train_df['sequence'].tolist() + val_df['sequence'].tolist(),
            train_df['is_promoter'].tolist() + val_df['is_promoter'].tolist(),
            encoder,
            augment=True
        )
        
        full_loader = DataLoader(
            full_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Create new validation split from training data
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader_final = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader_final = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
        
        history, best_f1 = trainer.train(
            train_loader_final, val_loader_final,
            epochs=config.num_epochs,
            early_stopping_patience=10
        )
        
        # Evaluate on test set
        test_loss, test_acc, test_f1, test_auroc = trainer.validate(test_loader)
        
        best_results[arch] = {
            "best_params": study.best_params,
            "best_f1": best_f1,
            "test_f1": test_f1,
            "test_auroc": test_auroc,
            "test_accuracy": test_acc,
            "history": history
        }
        
        # Save model
        torch.save(model.state_dict(), f"{config.output_dir}/{arch}_final_model.pt")
        
        # Save predictions
        model.eval()
        test_predictions = []
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)
                probs = F.softmax(output, dim=1)
                test_predictions.extend(probs[:, 1].cpu().numpy())
        
        test_df[f'cnn_{arch}_score'] = test_predictions
        
        if config.use_wandb:
            wandb.finish()
    
    # Save test predictions with all CNN scores
    test_df.to_parquet("results/test_predictions_cnn.parquet")
    
    # Save results
    with open("results/cnn_results.json", "w") as f:
        json.dump(best_results, f, indent=2)
    
    # Select best overall architecture
    best_arch = max(best_results, key=lambda x: best_results[x]['test_f1'])
    logger.info(f"\nBest architecture: {best_arch}")
    logger.info(f"Test F1: {best_results[best_arch]['test_f1']:.4f}")
    logger.info(f"Test AUROC: {best_results[best_arch]['test_auroc']:.4f}")
    
    # Save best model
    import shutil
    shutil.copy(
        f"{config.output_dir}/{best_arch}_final_model.pt",
        f"{config.output_dir}/best_model.pt"
    )
    
    # Save model configuration
    with open(f"{config.output_dir}/model_config.json", "w") as f:
        json.dump({
            "architecture": best_arch,
            "best_results": best_results[best_arch],
            "config": vars(config)
        }, f, indent=2)
    
    logger.info("CNN training complete!")

if __name__ == "__main__":
    main()
