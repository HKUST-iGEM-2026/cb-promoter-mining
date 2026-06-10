# task19_nt_lora_finetune/train_nt_lora.py
"""
Nucleotide Transformer v2 250m fine-tuning with LoRA
Owner: Sanaa
Due: 26 Jun
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
import wandb
import optuna
from dataclasses import dataclass
from typing import Dict, List, Optional
import gc
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class LoRAConfig:
    model_name: str = "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species"
    max_length: int = 300
    batch_size: int = 8  # Smaller due to model size
    learning_rate: float = 1e-4  # Higher LR for LoRA
    num_epochs: int = 15
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # LoRA specific parameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None  # Will set based on model
    
    # Training optimizations
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    use_8bit: bool = True  # Use 8-bit quantization for memory efficiency
    
    output_dir: str = "models/nt_lora_promoter"
    seed: int = 42

class LoRATrainer:
    """Trainer class for LoRA fine-tuning of NT model"""
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.setup_lora_config()
        
    def setup_lora_config(self):
        """Setup LoRA configuration for NT model"""
        
        # NT model uses different layer names
        if self.config.lora_target_modules is None:
            self.config.lora_target_modules = [
                "query", "key", "value", "dense", 
                "dense_h_to_4h", "dense_4h_to_h"
            ]
        
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            inference_mode=False
        )
    
    def load_model_in_8bit(self):
        """Load model with 8-bit quantization for memory efficiency"""
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            quantization_config=bnb_config if self.config.use_8bit else None,
            device_map="auto",
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            trust_remote_code=True
        )
        
        return model
    
    def prepare_model_for_lora(self, model):
        """Prepare model for LoRA training"""
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
        
        # Cast certain layers to fp32 for stability
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)
        
        # Apply LoRA
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='binary'),
            'auroc': roc_auc_score(labels, predictions),
            'mcc': matthews_corrcoef(labels, predictions)
        }
    
    def objective(self, trial, train_dataset, val_dataset):
        """Optuna objective for LoRA hyperparameters"""
        
        # Suggest hyperparameters
        lora_r = trial.suggest_categorical('lora_r', [4, 8, 16, 32])
        lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64])
        lora_dropout = trial.suggest_float('lora_dropout', 0.05, 0.3)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 8])
        
        # Update config
        self.config.lora_r = lora_r
        self.config.lora_alpha = lora_alpha
        self.config.lora_dropout = lora_dropout
        self.config.learning_rate = learning_rate
        self.config.batch_size = batch_size
        
        # Reload model with new LoRA config
        self.setup_lora_config()
        base_model = self.load_model_in_8bit()
        model = self.prepare_model_for_lora(base_model)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}/trial_{trial.number}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,  # Fewer epochs for trials
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=self.config.fp16,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            report_to="wandb",
            run_name=f"nt_lora_trial_{trial.number}"
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        return eval_results['eval_f1']
    
    def cross_validate(self, sequences, labels, n_folds=5):
        """Perform cross-validation with LoRA"""
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.seed)
        cv_scores = []
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Split data
            train_seqs = [sequences[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_seqs = [sequences[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Create datasets
            train_dataset = DNADataset(
                train_seqs, train_labels, tokenizer,
                self.config.max_length, augment=True
            )
            val_dataset = DNADataset(
                val_seqs, val_labels, tokenizer,
                self.config.max_length, augment=False
            )
            
            # Load and prepare model
            base_model = self.load_model_in_8bit()
            model = self.prepare_model_for_lora(base_model)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{self.config.output_dir}/fold_{fold}",
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                num_train_epochs=self.config.num_epochs,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                logging_steps=50,
                evaluation_strategy="steps",
                eval_steps=200,
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                fp16=self.config.fp16,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_checkpointing=self.config.gradient_checkpointing
            )
            
            # Train
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics
            )
            
            trainer.train()
            eval_results = trainer.evaluate()
            cv_scores.append(eval_results['eval_f1'])
            
            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        logger.info(f"Cross-validation F1 scores: {cv_scores}")
        logger.info(f"Mean F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return cv_scores

# ============================================================================
# Inference with Fine-tuned LoRA Model
# ============================================================================
class LoRAPromoterPredictor:
    """Inference class for fine-tuned LoRA model"""
    
    def __init__(self, model_path: str, config: LoRAConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model = self.model.merge_and_unload()  # Merge LoRA weights
        self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def predict(self, sequences: List[str], batch_size: int = 32):
        """Predict promoter probability for sequences"""
        predictions = []
        
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions.extend(probs[:, 1].cpu().numpy())
        
        return np.array(predictions)

# ============================================================================
# Main execution
# ============================================================================
def main():
    # Load data
    logger.info("Loading training data...")
    train_df = pd.read_parquet("data/train_promoters.parquet")
    val_df = pd.read_parquet("data/val_promoters.parquet")
    test_df = pd.read_parquet("data/test_promoters.parquet")
    
    # Initialize config
    config = LoRAConfig()
    
    # Initialize trainer
    trainer = LoRATrainer(config)
    
    # Cross-validation first
    logger.info("Starting cross-validation...")
    all_seqs = train_df['sequence'].tolist() + val_df['sequence'].tolist()
    all_labels = train_df['is_promoter'].tolist() + val_df['is_promoter'].tolist()
    cv_scores = trainer.cross_validate(all_seqs, all_labels, n_folds=5)
    
    # Hyperparameter optimization
    logger.info("Starting hyperparameter optimization...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    train_dataset = DNADataset(
        train_df['sequence'].tolist(),
        train_df['is_promoter'].tolist(),
        tokenizer,
        config.max_length,
        augment=True
    )
    val_dataset = DNADataset(
        val_df['sequence'].tolist(),
        val_df['is_promoter'].tolist(),
        tokenizer,
        config.max_length,
        augment=False
    )
    
    study = optuna.create_study(
        direction='maximize',
        study_name='nt_lora_optimization',
        storage='sqlite:///nt_lora_optuna.db',
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: trainer.objective(trial, train_dataset, val_dataset),
        n_trials=30,
        n_jobs=1
    )
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    best_params = study.best_params
    
    # Update config with best params
    for key, value in best_params.items():
        setattr(config, key, value)
    
    # Train on full dataset
    full_dataset = DNADataset(
        all_seqs, all_labels, tokenizer,
        config.max_length, augment=True
    )
    
    # Final training with best config
    base_model = trainer.load_model_in_8bit()
    model = trainer.prepare_model_for_lora(base_model)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs * 1.5,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        save_total_limit=2
    )
    
    final_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=full_dataset,
        compute_metrics=trainer.compute_metrics
    )
    
    final_trainer.train()
    
    # Save model
    model.save_pretrained(f"{config.output_dir}/final_lora_model")
    tokenizer.save_pretrained(f"{config.output_dir}/final_lora_model")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    predictor = LoRAPromoterPredictor(f"{config.output_dir}/final_lora_model", config)
    test_predictions = predictor.predict(test_df['sequence'].tolist())
    test_metrics = {
        'f1': f1_score(test_df['is_promoter'].tolist(), test_predictions > 0.5),
        'auroc': roc_auc_score(test_df['is_promoter'].tolist(), test_predictions)
    }
    
    # Save results
    results = {
        "best_params": study.best_params,
        "cv_scores": cv_scores,
        "mean_cv_f1": np.mean(cv_scores),
        "std_cv_f1": np.std(cv_scores),
        "test_metrics": test_metrics
    }
    
    import json
    with open("results/nt_lora_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    test_df['nt_lora_score'] = test_predictions
    test_df.to_parquet("results/test_predictions_nt_lora.parquet")
    
    logger.info("NT LoRA fine-tuning complete!")
    
    # Create model card
    create_model_card(config, results)

def create_model_card(config, results):
    """Create model card for Hugging Face Hub"""
    card_content = f"""
    # Nucleotide Transformer v2 250m - LoRA Fine-tuned for Promoter Prediction
    
    ## Model Description
    This model is a fine-tuned version of [InstaDeepAI/nucleotide-transformer-v2-250m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-250m-multi-species)
    using Low-Rank Adaptation (LoRA) for bacterial promoter prediction.
    
    ## Training Details
    - **Base Model**: Nucleotide Transformer v2 250m
    - **Fine-tuning Method**: LoRA
    - **LoRA Rank (r)**: {config.lora_r}
    - **LoRA Alpha**: {config.lora_alpha}
    - **LoRA Dropout**: {config.lora_dropout}
    - **Learning Rate**: {config.learning_rate}
    - **Batch Size**: {config.batch_size}
    - **Training Epochs**: {config.num_epochs}
    
    ## Performance
    - **Cross-validation F1**: {results['mean_cv_f1']:.4f} ± {results['std_cv_f1']:.4f}
    - **Test F1**: {results['test_metrics']['f1']:.4f}
    - **Test AUROC**: {results['test_metrics']['auroc']:.4f}
    
    ## Usage
    ```python
    from train_nt_lora import LoRAPromoterPredictor
    
    predictor = LoRAPromoterPredictor("models/nt_lora_promoter/final_lora_model")
    predictions = predictor.predict(["ATGC...", "CGTA..."])
    ```
    """
