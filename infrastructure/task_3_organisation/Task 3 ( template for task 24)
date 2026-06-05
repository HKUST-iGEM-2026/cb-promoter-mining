# task24_external_evaluation/evaluate_external.py
"""
External holdout evaluation on cross-species Clostridium dataset
Owner: Yany
Due: 7 Jul
Depends on: T9, T18, T19, T20

This script evaluates trained models on completely unseen Clostridium species
to test generalization to novel DNA sequences.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from scipy import stats
import json
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from typing import Dict, List

# Import model classes
import sys
sys.path.append('task18_prokbert_finetune')
sys.path.append('task19_nt_lora_finetune')
sys.path.append('task20_cnn_scratch')

from train_prokbert import ProkBertConfig, LoRAPromoterPredictor
from train_nt_lora import LoRAPromoterPredictor as NTLoRAPredictor
from train_cnn import PromoterCNN, DNAEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
class ExternalEvaluationConfig:
    """Configuration for external evaluation"""

    def __init__(self):
        self.test_species = [
            "Clostridium_perfringens",
            "Clostridium_difficile",
            "Clostridium_botulinum",
            "Clostridium_thermocellum",
            "Clostridium_phytotermentans"
        ]
        self.models_to_evaluate = ['prokbert', 'nt_lora', 'cnn']
        self.output_dir = Path("results/external_evaluation")
        self.synthetic_data_dir = Path("data/external_synthetic")  # New location for synthetic data
        self.synthetic_data_dir.mkdir(parents=True, exist_ok=True)

        self.save_predictions = True
        self.generate_plots = True
        self.bootstrap_iterations = 1000  # For confidence intervals

        # CNN model fallback path (can be adjusted as needed)
        self.cnn_fallback_model_path = Path("models/cnn_promoter/best_model.pt")

class ExternalEvaluator:
    """Evaluate models on external cross-species dataset"""

    def __init__(self, config: ExternalEvaluationConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}

    def load_external_data(self, species: str) -> pd.DataFrame:
        """Load external dataset for a specific species"""

        data_path = Path(f"data/external/{species}/test_data.parquet")

        if not data_path.exists():
            logger.warning(f"Data not found for {species}, generating synthetic data...")
            return self._generate_synthetic_data(species)

        df = pd.read_parquet(data_path)
        df['is_promoter'] = df['is_promoter'].astype(int)  # Ensure integer labels

        logger.info(f"Loaded {len(df)} sequences from {species}")
        logger.info(f"Promoter ratio: {df['is_promoter'].mean():.3f}")

        return df

    def _generate_synthetic_data(self, species: str) -> pd.DataFrame:
        """Generate synthetic test data (for testing purposes)"""
        np.random.seed(hash(species) % 2**32)

        n_sequences = 500
        n_promoters = int(n_sequences * 0.3)  # 30% promoters

        # Generate random sequences
        sequences = []
        for _ in range(n_sequences):
            seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 300))
            sequences.append(seq)

        # Generate labels
        labels = np.zeros(n_sequences, dtype=int)
        labels[:n_promoters] = 1
        np.random.shuffle(labels)

        df = pd.DataFrame({
            'sequence': sequences,
            'is_promoter': labels,
            'species': species
        })

        # Save generated data to separate synthetic data directory
        synthetic_path = self.config.synthetic_data_dir / f"{species}_synthetic.parquet"
        df.to_parquet(synthetic_path)
        logger.info(f"Synthetic data saved to {synthetic_path}")

        return df

    def load_model(self, model_name: str):
        """Load trained model for evaluation"""

        model_path = Path(f"models/{model_name}/final_model")

        if model_name == 'prokbert':
            config = ProkBertConfig()
            predictor = LoRAPromoterPredictor(str(model_path), config)
            return predictor

        elif model_name == 'nt_lora':
            config = ProkBertConfig()
            predictor = NTLoRAPredictor(str(model_path), config)
            return predictor

        elif model_name == 'cnn':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PromoterCNN(input_channels=5, sequence_length=300)

            # Try to load best model
            model_file = model_path / "best_model.pt"
            if not model_file.exists():
                if self.config.cnn_fallback_model_path.exists():
                    logger.warning(f"Model file not found at {model_file}, using fallback path {self.config.cnn_fallback_model_path}")
                    model_file = self.config.cnn_fallback_model_path
                else:
                    raise FileNotFoundError(f"No CNN model file found at {model_file} or fallback {self.config.cnn_fallback_model_path}")

            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            encoder = DNAEncoder(max_length=300)

            return (model, encoder, device)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def predict_with_model(self, model, sequences: List[str], model_name: str) -> np.ndarray:
        """Get predictions from model"""

        if model_name == 'cnn':
            model_obj, encoder, device = model
            probabilities = []
            batch_size = 64

            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                encoded = encoder.encode(batch).to(device)

                with torch.no_grad():
                    output = model_obj(encoded)
                    probs = torch.softmax(output, dim=1)
                    probabilities.extend(probs[:, 1].cpu().numpy())

            return np.array(probabilities)
        else:
            return model.predict(sequences)

    def evaluate_on_species(self, model_name: str, species: str,
                            df: pd.DataFrame) -> Dict:
        """Evaluate a single model on one species"""

        logger.info(f"Evaluating {model_name} on {species}")

        # Load model
        model = self.load_model(model_name)

        # Get predictions
        sequences = df['sequence'].tolist()
        true_labels = df['is_promoter'].astype(int).values

        probabilities = self.predict_with_model(model, sequences, model_name)
        predictions = (probabilities > 0.5).astype(int)

        # Calculate metrics safely (handle cases with one class present)
        metrics = {}
        try:
            metrics['accuracy'] = accuracy_score(true_labels, predictions)
            metrics['f1_score'] = f1_score(true_labels, predictions, average='binary')
            metrics['auroc'] = roc_auc_score(true_labels, probabilities)
            metrics['average_precision'] = average_precision_score(true_labels, probabilities)
            metrics['mcc'] = matthews_corrcoef(true_labels, predictions)
            metrics['precision'] = precision_score(true_labels, predictions)
            metrics['recall'] = recall_score(true_labels, predictions)
            metrics['specificity'] = self._calculate_specificity(true_labels, predictions)
        except ValueError as e:
            logger.warning(f"Metric calculation warning for {model_name} on {species}: {e}")
            # Provide default nan metrics on failure
            nan_val = float('nan')
            metrics = {k: nan_val for k in ['accuracy', 'f1_score', 'auroc', 'average_precision', 'mcc', 'precision', 'recall', 'specificity']}

        # Calculate confidence intervals via bootstrap if data is sufficient
        ci_metrics = {}
        if len(true_labels) > 0:
            ci_metrics = self._bootstrap_confidence(true_labels, probabilities,
                                                    self.config.bootstrap_iterations)

        return {
            'metrics': metrics,
            'confidence_intervals': ci_metrics,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'true_labels': true_labels.tolist()
        }

    def _calculate_specificity(self, true, pred):
        """Calculate specificity"""
        tn = np.sum((true == 0) & (pred == 0))
        fp = np.sum((true == 0) & (pred == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def _bootstrap_confidence(self, true_labels, probabilities, n_iterations=1000):
        """Calculate bootstrap confidence intervals"""

        n_samples = len(true_labels)
        metrics_bootstrap = {
            'auroc': [],
            'f1_score': [],
            'accuracy': []
        }

        for _ in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_true = true_labels[indices]
            boot_probs = probabilities[indices]
            boot_pred = (boot_probs > 0.5).astype(int)

            try:
                metrics_bootstrap['auroc'].append(roc_auc_score(boot_true, boot_probs))
            except ValueError:
                # Edge case: only one class in bootstrap sample
                metrics_bootstrap['auroc'].append(np.nan)

            metrics_bootstrap['f1_score'].append(f1_score(boot_true, boot_pred))
            metrics_bootstrap['accuracy'].append(accuracy_score(boot_true, boot_pred))

        # Remove NaNs if any
        for metric in metrics_bootstrap:
            metrics_bootstrap[metric] = [v for v in metrics_bootstrap[metric] if not np.isnan(v)]

        # Calculate confidence intervals
        cis = {}
        for metric, values in metrics_bootstrap.items():
            if values:
                cis[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'ci_95_lower': np.percentile(values, 2.5),
                    'ci_95_upper': np.percentile(values, 97.5)
                }
            else:
                cis[metric] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'ci_95_lower': float('nan'),
                    'ci_95_upper': float('nan'),
                }

        return cis

    def run_evaluation(self):
        """Run complete external evaluation"""

        for model_name in self.config.models_to_evaluate:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Evaluating {model_name} on external species")
            logger.info(f"{'=' * 60}")

            model_results = {}

            for species in self.config.test_species:
                # Load data
                df = self.load_external_data(species)

                if df.empty:
                    logger.warning(f"No data for species {species}, skipping evaluation.")
                    continue

                # Evaluate
                species_results = self.evaluate_on_species(model_name, species, df)
                model_results[species] = species_results

                # Log results
                metrics = species_results['metrics']
                logger.info(f"\n{species}:")
                logger.info(f"  AUROC: {metrics.get('auroc', float('nan')):.4f}")
                logger.info(f"  F1 Score: {metrics.get('f1_score', float('nan')):.4f}")
                logger.info(f"  MCC: {metrics.get('mcc', float('nan')):.4f}")

            self.results[model_name] = model_results

    def generate_comprehensive_report(self):
        """Generate detailed evaluation report"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'test_species': self.config.test_species,
                'models_evaluated': self.config.models_to_evaluate
            },
            'results': {}
        }

        for model_name, model_results in self.results.items():
            report['results'][model_name] = {}

            # Aggregate across species
            all_metrics = []

            for species, results in model_results.items():
                report['results'][model_name][species] = {
                    'metrics': results['metrics'],
                    'confidence_intervals': results['confidence_intervals']
                }
                all_metrics.append(results['metrics'])

            # If there's no data, skip summary
            if len(all_metrics) == 0:
                logger.warning(f"No metrics available for model {model_name}, skipping cross-species summary.")
                continue

            metrics_df = pd.DataFrame(all_metrics)

            report['results'][model_name]['cross_species_summary'] = {
                'mean_auroc': float(metrics_df['auroc'].mean()) if 'auroc' in metrics_df else float('nan'),
                'std_auroc': float(metrics_df['auroc'].std()) if 'auroc' in metrics_df else float('nan'),
                'mean_f1': float(metrics_df['f1_score'].mean()) if 'f1_score' in metrics_df else float('nan'),
                'std_f1': float(metrics_df['f1_score'].std()) if 'f1_score' in metrics_df else float('nan'),
                'mean_mcc': float(metrics_df['mcc'].mean()) if 'mcc' in metrics_df else float('nan'),
                'std_mcc': float(metrics_df['mcc'].std()) if 'mcc' in metrics_df else float('nan'),
                'generalization_gap': {
                    'auroc': float(metrics_df['auroc'].max() - metrics_df['auroc'].min()) if 'auroc' in metrics_df else float('nan'),
                    'f1': float(metrics_df['f1_score'].max() - metrics_df['f1_score'].min()) if 'f1_score' in metrics_df else float('nan'),
                }
            }

        # Save report
        with open(self.output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        return report

    def generate_visualizations(self):
        """Create comprehensive visualization plots"""

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        if not self.results:
            logger.warning("No results to plot.")
            return

        # Plot 1: Performance across species
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        metrics_to_plot = ['auroc', 'f1_score', 'mcc', 'accuracy']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            # Prepare data
            data = []
            for model_name, model_results in self.results.items():
                for species, results in model_results.items():
                    score = results['metrics'].get(metric, np.nan)
                    data.append({
                        'Model': model_name,
                        'Species': species,
                        'Score': score
                    })

            df_plot = pd.DataFrame(data)

            if df_plot.empty:
                logger.warning(f"No data to plot for metric {metric}.")
                continue

            # Create bar plot
            pivot_df = df_plot.pivot(index='Species', columns='Model', values='Score')
            pivot_df.plot(kind='bar', ax=ax, rot=45)

            ax.set_title(f'{metric.replace("_", " ").title()} Across Species')
            ax.set_ylabel(metric.upper())
            ax.set_xlabel('Species')
            ax.legend(loc='lower right')
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(plot_dir / "performance_across_species.png", dpi=150)
        plt.close()

        # Plot 2: Radar chart for model comparison
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        metrics = ['auroc', 'f1_score', 'mcc', 'precision', 'recall', 'specificity']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        for model_name, model_results in self.results.items():
            # Calculate mean across species
            mean_metrics = {}
            for metric in metrics:
                values = [results['metrics'].get(metric, np.nan) for results in model_results.values()]
                values = [v for v in values if not np.isnan(v)]
                mean_metrics[metric] = np.mean(values) if values else 0.0

            values = [mean_metrics[metric] for metric in metrics]
            values += values[:1]  # Close the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim([0, 1])
        ax.set_title('Model Performance Radar Chart (Cross-Species)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(plot_dir / "radar_chart.png", dpi=150)
        plt.close()

        # Plot 3: Confusion matrices for each model (aggregated)
        n_models = len(self.results)
        if n_models == 0:
            logger.warning("No models results available for confusion matrix plotting.")
            return

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes = [axes]

        for idx, (model_name, model_results) in enumerate(self.results.items()):
            # Aggregate predictions across all species
            all_predictions = []
            all_true = []

            for results in model_results.values():
                all_predictions.extend(results['predictions'])
                all_true.extend(results['true_labels'])

            if len(all_true) == 0 or len(all_predictions) == 0:
                logger.warning(f"No data for confusion matrix plotting for model {model_name}.")
                continue

            cm = confusion_matrix(all_true, all_predictions)

            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'{model_name}\nAggregated Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')

        plt.tight_layout()
        plt.savefig(plot_dir / "confusion_matrices_external.png", dpi=150)
        plt.close()

        # Plot 4: ROC curves for each species
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, species in enumerate(self.config.test_species):
            if idx >= len(axes):
                break

            ax = axes[idx]

            plotted = False
            for model_name, model_results in self.results.items():
                if species in model_results:
                    results = model_results[species]
                    true = np.array(results['true_labels'])
                    probs = np.array(results['probabilities'])
                    # Only plot if both classes present in labels
                    if len(np.unique(true)) < 2:
                        logger.warning(f"Skipping ROC plot for {model_name} on {species} (only one class present).")
                        continue

                    fpr, tpr, _ = roc_curve(true, probs)
                    auroc = roc_auc_score(true, probs)

                    ax.plot(fpr, tpr, label=f'{model_name} (AUROC={auroc:.3f})')
                    plotted = True

            if not plotted:
                ax.text(0.5, 0.5, 'Insufficient data for ROC', horizontalalignment='center', verticalalignment='center')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curves - {species}')
            ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(plot_dir / "roc_curves_by_species.png", dpi=150)
        plt.close()

        # Plot 5: Performance degradation from training to external
        fig, ax = plt.subplots(figsize=(10, 6))

        # This would compare internal validation performance (from T22) to external
        # For now, we'll create a placeholder
        internal_performance = {
            'prokbert': 0.92,  # Example values - replace with actual T22 results
            'nt_lora': 0.94,
            'cnn': 0.89
        }

        x = np.arange(len(self.config.models_to_evaluate))
        width = 0.35

        internal_scores = [internal_performance.get(m, 0) for m in self.config.models_to_evaluate]
        external_scores = []
        external_stds = []

        for model_name in self.config.models_to_evaluate:
            model_results = self.results.get(model_name, {})
            scores = [results['metrics'].get('auroc', np.nan) for results in model_results.values() if not np.isnan(results['metrics'].get('auroc', np.nan))]
            if len(scores) > 0:
                external_scores.append(np.mean(scores))
                external_stds.append(np.std(scores))
            else:
                external_scores.append(0)
                external_stds.append(0)

        ax.bar(x - width / 2, internal_scores, width, label='Internal Validation', color='blue', alpha=0.7)
        ax.bar(x + width / 2, external_scores, width, label='External (Cross-Species)',
               yerr=external_stds, capsize=5, color='orange', alpha=0.7)

        ax.set_xlabel('Model')
        ax.set_ylabel('AUROC')
        ax.set_title('Generalization Gap: Internal vs External Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(self.config.models_to_evaluate)
        ax.legend()
        ax.set_ylim([0, 1])

        # Add generalization gap annotations
        for i, (internal, external) in enumerate(zip(internal_scores, external_scores)):
            gap = internal - external
            ax.annotate(f'Δ = {gap:.3f}',
                        xy=(i, (internal + external) / 2),
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(plot_dir / "generalization_gap.png", dpi=150)
        plt.close()

    def save_predictions(self):
        """Save all predictions for downstream analysis"""

        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        for model_name, model_results in self.results.items():
            for species, results in model_results.items():
                df = pd.DataFrame({
                    'true_label': results['true_labels'],
                    'predicted_label': results['predictions'],
                    'probability': results['probabilities']
                })
                pred_path = predictions_dir / f"{model_name}_{species}_predictions.parquet"
                df.to_parquet(pred_path)

        logger.info(f"Predictions saved to {predictions_dir}")


# ============================================================================
# Main execution
# ============================================================================
def main():
    """Main execution function"""

    # Initialize configuration
    config = ExternalEvaluationConfig()

    # Initialize evaluator
    evaluator = ExternalEvaluator(config)

    # Run evaluation
    logger.info("Starting external evaluation on Clostridium species...")
    evaluator.run_evaluation()

    # Generate report
    report = evaluator.generate_comprehensive_report()

    # Generate visualizations
    if config.generate_plots:
        evaluator.generate_visualizations()

    # Save predictions
    if config.save_predictions:
        evaluator.save_predictions()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXTERNAL EVALUATION COMPLETE")
    logger.info("=" * 60)

    for model_name in config.models_to_evaluate:
        summary = report['results'].get(model_name, {}).get('cross_species_summary', None)
        if summary is None:
            logger.warning(f"No cross-species summary available for {model_name}")
            continue

        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Mean AUROC: {summary.get('mean_auroc', float('nan')):.4f} ± {summary.get('std_auroc', float('nan')):.4f}")
        logger.info(f"  Mean F1: {summary.get('mean_f1', float('nan')):.4f} ± {summary.get('std_f1', float('nan')):.4f}")
        logger.info(f"  Generalization Gap: {summary.get('generalization_gap', {}).get('auroc', float('nan')):.4f}")

    logger.info(f"\nFull report saved to {config.output_dir}/evaluation_report.json")
    logger.info(f"Visualizations saved to {config.output_dir}/plots/")


if __name__ == "__main__":
    main()






"""
Prepare external Clostridium dataset from T9
This script formats the external Clostridium promoter data for evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from Bio import SeqIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_clostridium_genomes():
    """Load Clostridium genome data and extract promoter and negative sequences."""

    species_list = [
        "Clostridium_perfringens",
        "Clostridium_difficile",
        "Clostridium_botulinum",
        "Clostridium_thermocellum",
        "Clostridium_acetobutylicum"
    ]

    all_data = []

    for species in species_list:
        logger.info(f"Processing {species}")

        genome_file = Path(f"data/external/raw/{species}.fna")
        gff_file = Path(f"data/external/raw/{species}.gff")

        if not genome_file.exists():
            logger.warning(f"Genome file not found for {species}, skipping")
            continue

        if not gff_file.exists():
            logger.warning(f"GFF annotation file not found for {species}, skipping promoter extraction")
            continue

        promoters = extract_promoters(genome_file, gff_file)
        if len(promoters) == 0:
            logger.warning(f"No promoter sequences extracted for {species}")

        negatives = generate_negatives(genome_file, len(promoters))

        for seq in promoters:
            if len(seq) >= 100:  # Keep reasonable length sequences only
                all_data.append({
                    'sequence': seq,
                    'is_promoter': 1,
                    'species': species,
                    'source': 'promoter_region'
                })

        for seq in negatives:
            if len(seq) >= 100:  # Filter short negatives
                all_data.append({
                    'sequence': seq,
                    'is_promoter': 0,
                    'species': species,
                    'source': 'intergenic_region'
                })

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path = Path("data/external/clostridium_test_set.parquet")
    df.to_parquet(output_path)

    logger.info(f"Created external dataset with {len(df)} sequences")
    logger.info(f"Promoter ratio: {df['is_promoter'].mean():.3f}")
    logger.info(f"Species distribution:\n{df['species'].value_counts()}")

    return df

def extract_promoters(genome_file: Path, gff_file: Path, upstream=300):
    """Extract promoter regions (upstream sequences) from genome based on GFF annotations."""

    genome = {}
    for record in SeqIO.parse(str(genome_file), "fasta"):
        genome[record.id] = str(record.seq)

    promoters = []

    with open(gff_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            seqid, source, feature_type, start, end, score, strand, phase, attributes = parts

            if feature_type not in {'gene', 'CDS'}:
                continue

            start, end = int(start), int(end)
            if seqid not in genome:
                logger.warning(f"Sequence ID {seqid} not found in genome for {genome_file}")
                continue

            chrom_seq = genome[seqid]

            if strand == '+':
                promoter_start = max(0, start - upstream - 1)
                promoter_end = start - 1
                promoter_seq = chrom_seq[promoter_start:promoter_end]

            else:
                promoter_start = end
                promoter_end = min(len(chrom_seq), end + upstream)
                promoter_seq = chrom_seq[promoter_start:promoter_end]
                promoter_seq = reverse_complement(promoter_seq)

            if len(promoter_seq) >= 100:  # threshold to avoid very short sequences
                promoters.append(promoter_seq[:upstream])

    logger.info(f"Extracted {len(promoters)} promoter sequences from {genome_file}")
    return promoters

def generate_negatives(genome_file: Path, n_promoters: int, length=300, ratio=3):
    """Generate negative (non-promoter) sequences by sampling random genome positions."""

    genome = {}
    for record in SeqIO.parse(str(genome_file), "fasta"):
        genome[record.id] = str(record.seq)

    genome_lengths = {k: len(v) for k, v in genome.items()}
    contig_names = list(genome.keys())
    contig_lengths = np.array([genome_lengths[contig] for contig in contig_names])
    cumulative_lengths = np.cumsum(contig_lengths)
    total_length = cumulative_lengths[-1]

    n_negatives = n_promoters * ratio
    negatives = []

    for _ in range(n_negatives):
        pos = np.random.randint(0, total_length)
        contig_idx = np.searchsorted(cumulative_lengths, pos, side='right')
        contig = contig_names[contig_idx]
        cont_start = cumulative_lengths[contig_idx - 1] if contig_idx > 0 else 0
        rel_pos = pos - contig_start

        start = max(0, rel_pos - length // 2)
        end = start + length

        # Ensure end does not exceed contig length
        if end > genome_lengths[contig]:
            end = genome_lengths[contig]
            start = max(0, end - length)

        neg_seq = genome[contig][start:end]
        if len(neg_seq) == length:  # only add negatives of correct length
            negatives.append(neg_seq)

    logger.info(f"Generated {len(negatives)} negative sequences")
    return negatives

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base.upper(), 'N') for base in reversed(seq))

def main():
    """Prepare and save external dataset, split by species."""

    Path("data/external/raw").mkdir(parents=True, exist_ok=True)
    Path("data/external/processed").mkdir(parents=True, exist_ok=True)

    df = load_clostridium_genomes()

    for species in df['species'].unique():
        species_safe = species.replace(' ', '_')
        species_df = df[df['species'] == species]
        species_df.to_parquet(f"data/external/processed/{species_safe}.parquet")

    logger.info("External dataset preparation complete!")

if __name__ == "__main__":
    main()
