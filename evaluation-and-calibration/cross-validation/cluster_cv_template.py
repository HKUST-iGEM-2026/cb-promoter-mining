"""
Cluster-level 5-fold cross-validation for promoter prediction models
Owner: Yany
Due: 7 Jul
Depends on: T15, T18, T19, T20

This script performs robust cross-validation at the cluster/organism level
to ensure performance estimates are stable and not overfitted to specific strains.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, confusion_matrix, roc_curve
)
from scipy import stats
import json
from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

# Import model classes from previous tasks
import sys
sys.path.append('task18_prokbert_finetune')
sys.path.append('task19_nt_lora_finetune')
sys.path.append('task20_cnn_scratch')

from train_prokbert import ProkBertConfig, LoRAPromoterPredictor
from train_nt_lora import LoRAPromoterPredictor as NTLoRAPredictor
from train_cnn import PromoterCNN, DNAEncoder, CNNDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation"""
    n_folds: int = 5
    random_seed: int = 42
    test_models: List[str] = None  # ['prokbert', 'nt_lora', 'cnn']
    output_dir: str = "results/cross_validation"
    save_predictions: bool = True
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95

    def __post_init__(self):
        if self.test_models is None:
            self.test_models = ['prokbert', 'nt_lora', 'cnn']

class CrossValidationResult:
    """Store cross-validation results for a model"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.fold_metrics = []
        self.fold_predictions = []
        self.fold_true_labels = []
        self.fold_probabilities = []

    def add_fold_result(self, metrics: Dict, predictions: np.ndarray,
                       true_labels: np.ndarray, probabilities: np.ndarray):
        """Add results from one fold"""
        self.fold_metrics.append(metrics)
        self.fold_predictions.append(predictions)
        self.fold_true_labels.append(true_labels)
        self.fold_probabilities.append(probabilities)

    def get_summary_statistics(self, confidence_level: float = 0.95):
        """Calculate summary statistics across folds"""
        metrics_df = pd.DataFrame(self.fold_metrics)

        summary = {}
        for metric in metrics_df.columns:
            values = metrics_df[metric].values
            mean = np.mean(values)
            std = np.std(values)
            sem = stats.sem(values)
            ci = stats.t.interval(confidence_level, len(values) - 1, loc=mean, scale=sem)

            summary[metric] = {
                'mean': float(mean),
                'std': float(std),
                'sem': float(sem),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'values': values.tolist()
            }

        return summary

    def get_aggregate_predictions(self):
        """Combine predictions from all folds"""
        all_predictions = np.concatenate(self.fold_predictions)
        all_true = np.concatenate(self.fold_true_labels)
        all_probs = np.concatenate(self.fold_probabilities)

        return all_predictions, all_true, all_probs

class ClusterCrossValidator:
    """Perform cluster-level cross-validation"""

    def __init__(self, config: CrossValidationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set random seeds
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

        self.results = {}

    def load_data(self, data_path: str = "data/cv_dataset.parquet"):
        """Load dataset with cluster/organism information"""

        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)

        # Expected columns: sequence, is_promoter, organism, cluster_id
        required_cols = ['sequence', 'is_promoter', 'organism', 'cluster_id']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        logger.info(f"Dataset size: {len(df)} sequences")
        logger.info(f"Promoter ratio: {df['is_promoter'].mean():.3f}")
        logger.info(f"Number of organisms: {df['organism'].nunique()}")
        logger.info(f"Number of clusters: {df['cluster_id'].nunique()}")

        organism_stats = df.groupby('organism')['is_promoter'].agg(['count', 'mean'])
        logger.info("\nOrganism distribution:")
        logger.info(organism_stats)

        return df

    def prepare_folds(self, df: pd.DataFrame):
        """Prepare cluster-level stratified folds"""

        unique_clusters = df['cluster_id'].unique()

        cluster_promoter_ratio = df.groupby('cluster_id')['is_promoter'].mean()
        cluster_labels = (cluster_promoter_ratio[unique_clusters] > 0.5).astype(int)

        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )

        fold_indices = []

        for train_clusters, val_clusters in skf.split(unique_clusters, cluster_labels):
            train_idx = df[df['cluster_id'].isin(unique_clusters[train_clusters])].index.values
            val_idx = df[df['cluster_id'].isin(unique_clusters[val_clusters])].index.values

            fold_indices.append((train_idx, val_idx))

            logger.info(f"Fold {len(fold_indices)}: Train clusters={len(train_clusters)}, "
                        f"Val clusters={len(val_clusters)}")
            logger.info(f"  Train size: {len(train_idx)}, Val size: {len(val_idx)}")

        return fold_indices

    def load_model(self, model_name: str, fold: int):
        """Load pre-trained model for the specified fold"""

        model_path = Path(f"models/{model_name}/fold_{fold}")

        if model_name == 'prokbert':
            config = ProkBertConfig()
            predictor = LoRAPromoterPredictor(str(model_path), config)
            return predictor

        elif model_name == 'nt_lora':
            config = ProkBertConfig()  # Adjust if nt_lora has distinct config
            predictor = NTLoRAPredictor(str(model_path), config)
            return predictor

        elif model_name == 'cnn':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PromoterCNN(input_channels=5, sequence_length=300)
            model.load_state_dict(torch.load(model_path / "best_model.pt", map_location=device))
            model.to(device)
            model.eval()
            encoder = DNAEncoder(max_length=300)
            return (model, encoder, device)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def evaluate_model_on_fold(self, model_name: str, fold: int,
                               train_idx, val_idx, df: pd.DataFrame):
        """Evaluate a single model on one fold"""

        logger.info(f"Evaluating {model_name} on fold {fold}")

        val_df = df.iloc[val_idx]
        sequences = val_df['sequence'].tolist()
        true_labels = val_df['is_promoter'].values

        model = self.load_model(model_name, fold)

        if model_name == 'cnn':
            model_obj, encoder, device = model
            probabilities = self._predict_cnn(model_obj, encoder, device, sequences)
        else:
            probabilities = model.predict(sequences)

        predictions = (probabilities > 0.5).astype(int)

        metrics = self.calculate_metrics(true_labels, predictions, probabilities)

        return metrics, predictions, true_labels, probabilities

    def _predict_cnn(self, model, encoder, device, sequences, batch_size=64):
        probabilities = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            encoded = encoder.encode(batch).to(device)

            with torch.no_grad():
                output = model(encoded)
                probs = torch.softmax(output, dim=1)
                probabilities.extend(probs[:, 1].cpu().numpy())

        return np.array(probabilities)

    def calculate_metrics(self, true_labels, predictions, probabilities):
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1_score': f1_score(true_labels, predictions, average='binary'),
            'auroc': roc_auc_score(true_labels, probabilities),
            'average_precision': average_precision_score(true_labels, probabilities),
            'mcc': matthews_corrcoef(true_labels, predictions),
            'specificity': self._calculate_specificity(true_labels, predictions),
            'sensitivity': self._calculate_sensitivity(true_labels, predictions),
            'precision': self._calculate_precision(true_labels, predictions),
            'balanced_accuracy': self._calculate_balanced_accuracy(true_labels, predictions)
        }
        return metrics

    def _calculate_specificity(self, true, pred):
        tn = np.sum((true == 0) & (pred == 0))
        fp = np.sum((true == 0) & (pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def _calculate_sensitivity(self, true, pred):
        tp = np.sum((true == 1) & (pred == 1))
        fn = np.sum((true == 1) & (pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def _calculate_precision(self, true, pred):
        tp = np.sum((true == 1) & (pred == 1))
        fp = np.sum((true == 0) & (pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def _calculate_balanced_accuracy(self, true, pred):
        sensitivity = self._calculate_sensitivity(true, pred)
        specificity = self._calculate_specificity(true, pred)
        return (sensitivity + specificity) / 2

    def run_cross_validation(self, df: pd.DataFrame):
        fold_indices = self.prepare_folds(df)

        for model_name in self.config.test_models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running cross-validation for {model_name}")
            logger.info(f"{'='*60}")

            cv_result = CrossValidationResult(model_name)

            for fold, (train_idx, val_idx) in enumerate(fold_indices, 1):
                logger.info(f"\n--- Fold {fold}/{self.config.n_folds} ---")

                try:
                    metrics, predictions, true_labels, probabilities = self.evaluate_model_on_fold(
                        model_name, fold, train_idx, val_idx, df
                    )

                    cv_result.add_fold_result(metrics, predictions, true_labels, probabilities)

                    logger.info(f"Fold {fold} results: F1={metrics['f1_score']:.4f}, "
                                f"AUROC={metrics['auroc']:.4f}")

                except Exception as e:
                    logger.error(f"Error in fold {fold} for {model_name}: {e}", exc_info=True)
                    continue

            self.results[model_name] = cv_result

            summary = cv_result.get_summary_statistics(self.config.confidence_level)
            logger.info(f"\n{model_name.upper()} Cross-Validation Summary:")
            for metric, stats in summary.items():
                logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                            f"[{self.config.confidence_level*100:.1f}% CI: {stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")

    def generate_plots(self):
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Metric distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        metrics_to_plot = ['f1_score', 'auroc', 'accuracy', 'mcc', 'average_precision', 'balanced_accuracy']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            data_for_models = [ [m[metric] for m in result.fold_metrics] for result in self.results.values()]
            ax.boxplot(data_for_models, labels=list(self.results.keys()))
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel(metric)

        plt.tight_layout()
        path = plot_dir / "metric_distributions.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved metric distribution plot to {path}")

        # Confusion matrices
        fig, axes = plt.subplots(1, len(self.results), figsize=(5*len(self.results), 4))
        if len(self.results) == 1:
            axes = [axes]

        import matplotlib.ticker as mticker
        for ax in axes:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        for idx, (model_name, result) in enumerate(self.results.items()):
            all_pred, all_true, _ = result.get_aggregate_predictions()
            cm = confusion_matrix(all_true, all_pred)

            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{model_name}\nAggregated Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')

        plt.tight_layout()
        path = plot_dir / "confusion_matrices.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrices plot to {path}")

        # ROC curves
        fig, axes = plt.subplots(1, len(self.results), figsize=(5*len(self.results), 4))
        if len(self.results) == 1:
            axes = [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]

            for fold_idx, (probs, true) in enumerate(zip(result.fold_probabilities, result.fold_true_labels)):
                fpr, tpr, _ = roc_curve(true, probs)
                auroc = roc_auc_score(true, probs)
                ax.plot(fpr, tpr, alpha=0.6, label=f'Fold {fold_idx + 1} (AUROC={auroc:.3f})')

            all_probs = np.concatenate(result.fold_probabilities)
            all_true = np.concatenate(result.fold_true_labels)
            fpr, tpr, _ = roc_curve(all_true, all_probs)
            mean_auroc = roc_auc_score(all_true, all_probs)
            ax.plot(fpr, tpr, 'k--', linewidth=2, label=f'Mean (AUROC={mean_auroc:.3f})')

            ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC Curves')
            ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        path = plot_dir / "roc_curves.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved ROC curves plot to {path}")

        # Performance comparison bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['f1_score', 'auroc', 'accuracy', 'mcc']
        x = np.arange(len(metrics))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, (model_name, result) in enumerate(self.results.items()):
            means = [np.mean([m[metric] for m in result.fold_metrics]) for metric in metrics]
            errors = [np.std([m[metric] for m in result.fold_metrics]) for metric in metrics]

            ax.bar(x + idx * width, means, width, label=model_name,
                   yerr=errors, capsize=3, color=colors[idx % len(colors)])

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.set_ylim([0, 1])

        plt.tight_layout()
        path = plot_dir / "model_comparison.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved model comparison plot to {path}")

    def save_results(self):
        summary = {}
        for model_name, result in self.results.items():
            summary[model_name] = result.get_summary_statistics(self.config.confidence_level)

        with open(self.output_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        for model_name, result in self.results.items():
            fold_results = []
            for fold_idx in range(len(result.fold_metrics)):
                fold_results.append({
                    'fold': fold_idx + 1,
                    'metrics': result.fold_metrics[fold_idx],
                    'predictions': result.fold_predictions[fold_idx].tolist(),
                    'true_labels': result.fold_true_labels[fold_idx].tolist(),
                    'probabilities': result.fold_probabilities[fold_idx].tolist()
                })

            with open(self.output_dir / f"{model_name}_fold_results.json", "w") as f:
                json.dump(fold_results, f, indent=2)

        if self.config.save_predictions:
            for model_name, result in self.results.items():
                all_pred, all_true, all_probs = result.get_aggregate_predictions()

                predictions_df = pd.DataFrame({
                    'true_label': all_true,
                    'predicted_label': all_pred,
                    'probability': all_probs
                })
                predictions_df.to_parquet(self.output_dir / f"{model_name}_aggregated_predictions.parquet")

        self.generate_plots()

        logger.info(f"\nResults saved to {self.output_dir}")

    def perform_statistical_tests(self):
        from scipy.stats import wilcoxon, friedmanchisquare

        metrics = ['f1_score', 'auroc']
        statistical_results = {}

        for metric in metrics:
            model_scores = []
            model_names = []

            for model_name, result in self.results.items():
                scores = [m[metric] for m in result.fold_metrics]
                model_scores.append(scores)
                model_names.append(model_name)

            friedman_stat, friedman_p = friedmanchisquare(*model_scores)

            pairwise_results = {}
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    stat, p_val = wilcoxon(model_scores[i], model_scores[j])
                    pairwise_results[f"{model_names[i]} vs {model_names[j]}"] = {
                        'statistic': float(stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    }

            statistical_results[metric] = {
                'friedman_test': {
                    'statistic': float(friedman_stat),
                    'p_value': float(friedman_p),
                    'significant': friedman_p < 0.05
                },
                'pairwise_comparisons': pairwise_results
            }

        with open(self.output_dir / "statistical_tests.json", "w") as f:
            json.dump(statistical_results, f, indent=2)

        return statistical_results

def main():
    config = CrossValidationConfig(
        n_folds=5,
        random_seed=42,
        test_models=['prokbert', 'nt_lora', 'cnn'],
        output_dir="results/cross_validation",
        save_predictions=True,
        compute_confidence_intervals=True
    )

    validator = ClusterCrossValidator(config)
    df = validator.load_data("data/cluster_cv_dataset.parquet")
    validator.run_cross_validation(df)
    validator.save_results()
    statistical_results = validator.perform_statistical_tests()

    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION COMPLETE")
    logger.info("="*60)

    for model_name, result in validator.results.items():
        summary = result.get_summary_statistics(0.95)
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  F1 Score: {summary['f1_score']['mean']:.4f} ± {summary['f1_score']['std']:.4f}")
        logger.info(f"  AUROC: {summary['auroc']['mean']:.4f} ± {summary['auroc']['std']:.4f}")
        logger.info(f"  MCC: {summary['mcc']['mean']:.4f} ± {summary['mcc']['std']:.4f}")

    logger.info("\nStatistical significance tests saved to results/cross_validation/statistical_tests.json")

if __name__ == "__main__":
    main()
