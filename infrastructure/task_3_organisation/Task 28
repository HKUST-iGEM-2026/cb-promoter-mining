# task28_sliding_window/sliding_window_engine.py
"""
Sliding-window inference engine for Clostridium genome-wide promoter prediction
Owner: Swasti
Due: 11 Jul
Depends on: PT6, T13, T20, T21

This engine runs all three trained models across Clostridium genomes using
a sliding window approach, scoped to upstream regions from T13.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from tqdm import tqdm
import json
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d

# Import model classes from previous tasks
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
@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window inference"""
    
    # Window parameters
    window_size: int = 300  # bp
    stride: int = 10  # bp (overlap for smooth predictions)
    
    # Model ensemble
    models_to_use: List[str] = field(default_factory=lambda: ['prokbert', 'nt_lora', 'cnn'])
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'prokbert': 0.33,
        'nt_lora': 0.34,
        'cnn': 0.33
    })
    
    # Post-processing
    smoothing_sigma: float = 2.0  # Gaussian smoothing sigma
    min_peak_distance: int = 50  # Minimum distance between promoter peaks (bp)
    min_peak_prominence: float = 0.3  # Minimum peak prominence
    promoter_threshold: float = 0.5  # Binary classification threshold
    
    # Output options
    output_dir: str = "results/sliding_window"
    save_predictions: bool = True
    save_track_files: bool = True  # For genome browsers
    generate_plots: bool = True
    
    # Performance
    batch_size: int = 64
    num_workers: int = 4
    use_gpu: bool = True
    
    # Upstream regions only (from T13)
    upstream_only: bool = True
    upstream_buffer: int = 500  # bp before and after upstream regions

@dataclass
class PromoterPrediction:
    """Store a single promoter prediction"""
    contig: str
    start: int
    end: int
    center: int
    sequence: str
    prokbert_score: float
    nt_lora_score: float
    cnn_score: float
    ensemble_score: float
    confidence: str  # 'high', 'medium', 'low'
    motif_match: Optional[Dict] = None

# ============================================================================
# Genome Scanner
# ============================================================================
class GenomeScanner:
    """Scan genome with sliding window"""
    
    def __init__(self, genome_path: str, config: SlidingWindowConfig):
        self.genome_path = Path(genome_path)
        self.config = config
        self.contigs = {}
        self.load_genome()
        
    def load_genome(self):
        """Load genome from FASTA file"""
        logger.info(f"Loading genome from {self.genome_path}")
        
        for record in SeqIO.parse(self.genome_path, "fasta"):
            self.contigs[record.id] = str(record.seq).upper()
            
        logger.info(f"Loaded {len(self.contigs)} contigs")
        self.total_length = sum(len(seq) for seq in self.contigs.values())
        logger.info(f"Total genome length: {self.total_length:,} bp")
    
    def get_upstream_regions(self, upstream_regions_file: str) -> List[Dict]:
        """Load upstream regions from T13"""
        
        if not upstream_regions_file or not Path(upstream_regions_file).exists():
            logger.warning("Upstream regions file not found, scanning entire genome")
            return None
        
        logger.info(f"Loading upstream regions from {upstream_regions_file}")
        df = pd.read_parquet(upstream_regions_file)
        
        regions = []
        for _, row in df.iterrows():
            regions.append({
                'contig': row['contig'],
                'start': row['start'],
                'end': row['end'],
                'gene_id': row.get('gene_id', 'unknown'),
                'strand': row.get('strand', '+')
            })
        
        logger.info(f"Loaded {len(regions)} upstream regions")
        return regions
    
    def sliding_window_generator(self, contig_id: str, sequence: str, 
                                 upstream_regions: Optional[List[Dict]] = None) -> Generator:
        """Generate sliding windows across genome or specific regions"""
        
        if upstream_regions and self.config.upstream_only:
            # Only scan specified upstream regions
            region_windows = []
            for region in upstream_regions:
                if region['contig'] != contig_id:
                    continue
                
                # Add buffer around region
                start = max(0, region['start'] - self.config.upstream_buffer)
                end = min(len(sequence), region['end'] + self.config.upstream_buffer)
                
                # Generate windows within this region
                for pos in range(start, end - self.config.window_size + 1, self.config.stride):
                    region_windows.append(pos)
            
            # Remove duplicates and sort
            region_windows = sorted(set(region_windows))
            
            for pos in region_windows:
                window_seq = sequence[pos:pos + self.config.window_size]
                if len(window_seq) == self.config.window_size:
                    yield pos, pos + self.config.window_size, window_seq
        
        else:
            # Scan entire contig
            for pos in range(0, len(sequence) - self.config.window_size + 1, self.config.stride):
                window_seq = sequence[pos:pos + self.config.window_size]
                if len(window_seq) == self.config.window_size:
                    yield pos, pos + self.config.window_size, window_seq

# ============================================================================
# Model Loader and Predictor
# ============================================================================
class ModelEnsemble:
    """Load and run ensemble of models"""
    
    def __init__(self, config: SlidingWindowConfig):
        self.config = config
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        
        # Load ProkBERT
        if 'prokbert' in self.config.models_to_use:
            logger.info("Loading ProkBERT model...")
            try:
                model_path = Path("models/prokbert_promoter/final_model")
                prokbert_config = ProkBertConfig()
                self.models['prokbert'] = LoRAPromoterPredictor(str(model_path), prokbert_config)
                logger.info("ProkBERT loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ProkBERT: {e}")
                self.models['prokbert'] = None
        
        # Load Nucleotide Transformer + LoRA
        if 'nt_lora' in self.config.models_to_use:
            logger.info("Loading Nucleotide Transformer + LoRA...")
            try:
                model_path = Path("models/nt_lora_promoter/final_lora_model")
                nt_config = ProkBertConfig()
                self.models['nt_lora'] = NTLoRAPredictor(str(model_path), nt_config)
                logger.info("NT+LoRA loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NT+LoRA: {e}")
                self.models['nt_lora'] = None
        
        # Load CNN
        if 'cnn' in self.config.models_to_use:
            logger.info("Loading CNN model...")
            try:
                model_path = Path("models/cnn_promoter/best_model.pt")
                model = PromoterCNN(input_channels=5, sequence_length=300)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                encoder = DNAEncoder(max_length=300)
                self.models['cnn'] = (model, encoder, self.device)
                logger.info("CNN loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load CNN: {e}")
                self.models['cnn'] = None
    
    def predict_batch(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """Run batch prediction with all models"""
        
        predictions = defaultdict(list)
        
        for model_name, model in self.models.items():
            if model is None:
                continue
            
            if model_name == 'cnn':
                model_obj, encoder, device = model
                batch_probs = []
                
                for i in range(0, len(sequences), self.config.batch_size):
                    batch = sequences[i:i+self.config.batch_size]
                    encoded = encoder.encode(batch).to(device)
                    
                    with torch.no_grad():
                        output = model_obj(encoded)
                        probs = torch.softmax(output, dim=1)
                        batch_probs.extend(probs[:, 1].cpu().numpy())
                
                predictions[model_name] = np.array(batch_probs)
            
            else:
                # Transformer models
                batch_probs = model.predict(sequences)
                predictions[model_name] = np.array(batch_probs)
        
        # Calculate ensemble score
        ensemble_score = np.zeros(len(sequences))
        total_weight = 0
        
        for model_name, weight in self.config.ensemble_weights.items():
            if model_name in predictions and predictions[model_name] is not None:
                ensemble_score += predictions[model_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_score /= total_weight
        
        predictions['ensemble'] = ensemble_score
        
        return predictions

# ============================================================================
# Post-processing and Peak Detection
# ============================================================================
class PromoterDetector:
    """Detect promoter regions from sliding window scores"""
    
    def __init__(self, config: SlidingWindowConfig):
        self.config = config
    
    def smooth_scores(self, positions: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to prediction scores"""
        
        # Create continuous array
        min_pos = positions.min()
        max_pos = positions.max()
        step = self.config.stride
        
        # Interpolate to continuous positions
        continuous_pos = np.arange(min_pos, max_pos + step, step)
        continuous_scores = np.interp(continuous_pos, positions, scores)
        
        # Apply Gaussian filter
        smoothed = gaussian_filter1d(continuous_scores, sigma=self.config.smoothing_sigma)
        
        return continuous_pos, smoothed
    
    def find_promoter_peaks(self, positions: np.ndarray, scores: np.ndarray) -> List[Dict]:
        """Identify promoter peaks in sliding window scores"""
        
        # Smooth scores
        cont_pos, smoothed = self.smooth_scores(positions, scores)
        
        # Find peaks
        peaks, properties = find_peaks(
            smoothed,
            distance=self.config.min_peak_distance / self.config.stride,
            prominence=self.config.min_peak_prominence,
            height=self.config.promoter_threshold
        )
        
        promoter_regions = []
        for peak_idx in peaks:
            peak_pos = cont_pos[peak_idx]
            peak_score = smoothed[peak_idx]
            peak_prominence = properties['prominences'][peak_idx] if 'prominences' in properties else 0
            
            # Define promoter region around peak
            half_width = self.config.window_size // 2
            start = max(0, int(peak_pos - half_width))
            end = start + self.config.window_size
            
            # Determine confidence based on peak properties
            if peak_score > 0.8 and peak_prominence > 0.5:
                confidence = 'high'
            elif peak_score > 0.6 and peak_prominence > 0.3:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            promoter_regions.append({
                'center': int(peak_pos),
                'start': start,
                'end': end,
                'score': float(peak_score),
                'prominence': float(peak_prominence),
                'confidence': confidence
            })
        
        return promoter_regions
    
    def merge_overlapping_promoters(self, promoters: List[Dict], merge_distance: int = 100) -> List[Dict]:
        """Merge overlapping or nearby promoter predictions"""
        
        if not promoters:
            return []
        
        # Sort by start position
        promoters.sort(key=lambda x: x['start'])
        
        merged = []
        current = promoters[0].copy()
        
        for next_promoter in promoters[1:]:
            if next_promoter['start'] - current['end'] <= merge_distance:
                # Merge
                current['end'] = max(current['end'], next_promoter['end'])
                current['score'] = max(current['score'], next_promoter['score'])
                current['confidence'] = max(current['confidence'], next_promoter['confidence'],
                                           key=lambda c: {'high': 2, 'medium': 1, 'low': 0}[c])
            else:
                merged.append(current)
                current = next_promoter.copy()
        
        merged.append(current)
        return merged

# ============================================================================
# Main Inference Engine
# ============================================================================
class SlidingWindowEngine:
    """Main engine for genome-wide promoter prediction"""
    
    def __init__(self, config: SlidingWindowConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scanner = None
        self.ensemble = None
        self.detector = None
        
    def initialize(self, genome_path: str, upstream_regions_file: Optional[str] = None):
        """Initialize engine with genome and models"""
        
        self.scanner = GenomeScanner(genome_path, self.config)
        self.ensemble = ModelEnsemble(self.config)
        self.detector = PromoterDetector(self.config)
        
        if upstream_regions_file:
            self.upstream_regions = self.scanner.get_upstream_regions(upstream_regions_file)
        else:
            self.upstream_regions = None
    
    def scan_contig(self, contig_id: str, sequence: str) -> Dict:
        """Scan a single contig with sliding window"""
        
        logger.info(f"Scanning contig {contig_id} (length: {len(sequence):,} bp)")
        
        positions = []
        sequences = []
        window_info = []
        
        # Generate windows
        for start, end, seq in self.scanner.sliding_window_generator(
            contig_id, sequence, self.upstream_regions
        ):
            positions.append(start)
            sequences.append(seq)
            window_info.append({
                'contig': contig_id,
                'start': start,
                'end': end
            })
        
        if not sequences:
            logger.warning(f"No windows generated for {contig_id}")
            return {}
        
        # Batch prediction
        logger.info(f"Running inference on {len(sequences)} windows...")
        all_predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(sequences), self.config.batch_size), desc=f"Predicting {contig_id}"):
            batch_seqs = sequences[i:i+self.config.batch_size]
            batch_preds = self.ensemble.predict_batch(batch_seqs)
            all_predictions.append(batch_preds)
        
        # Aggregate predictions
        aggregated = defaultdict(list)
        for pred in all_predictions:
            for model_name, scores in pred.items():
                aggregated[model_name].extend(scores)
        
        # Convert to arrays
        for model_name in aggregated:
            aggregated[model_name] = np.array(aggregated[model_name])
        
        # Detect promoters
        ensemble_scores = aggregated['ensemble']
        promoter_peaks = self.detector.find_promoter_peaks(
            np.array(positions), ensemble_scores
        )
        
        # Merge overlapping promoters
        merged_promoters = self.detector.merge_overlapping_promoters(promoter_peaks)
        
        # Create detailed promoter objects
        promoters = []
        for peak in merged_promoters:
            # Get the window scores for this region
            region_mask = (np.array(positions) >= peak['start']) & (np.array(positions) <= peak['end'])
            
            promoter = PromoterPrediction(
                contig=contig_id,
                start=peak['start'],
                end=peak['end'],
                center=peak['center'],
                sequence=sequence[peak['start']:peak['end']],
                prokbert_score=float(np.mean(aggregated['prokbert'][region_mask])) if 'prokbert' in aggregated else 0,
                nt_lora_score=float(np.mean(aggregated['nt_lora'][region_mask])) if 'nt_lora' in aggregated else 0,
                cnn_score=float(np.mean(aggregated['cnn'][region_mask])) if 'cnn' in aggregated else 0,
                ensemble_score=float(np.mean(ensemble_scores[region_mask])),
                confidence=peak['confidence']
            )
            promoters.append(promoter)
        
        return {
            'contig': contig_id,
            'positions': positions,
            'predictions': aggregated,
            'promoters': promoters,
            'window_info': window_info
        }
    
    def run_genome_scan(self, genome_path: str, upstream_file: Optional[str] = None):
        """Run complete genome scan"""
        
        self.initialize(genome_path, upstream_file)
        
        all_results = {}
        all_promoters = []
        
        logger.info(f"\n{'='*60}")
        logger.info("Starting genome-wide promoter scan")
        logger.info(f"{'='*60}")
        
        for contig_id, sequence in self.scanner.contigs.items():
            try:
                result = self.scan_contig(contig_id, sequence)
                if result:
                    all_results[contig_id] = result
                    all_promoters.extend(result['promoters'])
                    logger.info(f"Found {len(result['promoters'])} promoter regions in {contig_id}")
            except Exception as e:
                logger.error(f"Error scanning {contig_id}: {e}")
                continue
        
        # Save results
        self.save_results(all_results, all_promoters)
        
        # Generate reports
        if self.config.generate_plots:
            self.generate_visualizations(all_results, all_promoters)
        
        logger.info(f"\nScan complete! Found {len(all_promoters)} promoter regions total")
        
        return all_results, all_promoters
    
    def save_results(self, all_results: Dict, all_promoters: List[PromoterPrediction]):
        """Save all results to disk"""
        
        # Save raw predictions
        if self.config.save_predictions:
            for contig_id, result in all_results.items():
                # Save scores
                scores_df = pd.DataFrame({
                    'position': result['positions'],
                    **{f'{model}_score': scores for model, scores in result['predictions'].items()}
                })
                scores_df.to_parquet(self.output_dir / f"{contig_id}_scores.parquet")
                
                # Save window info
                windows_df = pd.DataFrame(result['window_info'])
                windows_df.to_parquet(self.output_dir / f"{contig_id}_windows.parquet")
        
        # Save promoter predictions
        promoters_data = []
        for promoter in all_promoters:
            promoters_data.append({
                'contig': promoter.contig,
                'start': promoter.start,
                'end': promoter.end,
                'center': promoter.center,
                'sequence': promoter.sequence,
                'prokbert_score': promoter.prokbert_score,
                'nt_lora_score': promoter.nt_lora_score,
                'cnn_score': promoter.cnn_score,
                'ensemble_score': promoter.ensemble_score,
                'confidence': promoter.confidence,
                'length': len(promoter.sequence)
            })
        
        promoters_df = pd.DataFrame(promoters_data)
        promoters_df.to_parquet(self.output_dir / "predicted_promoters.parquet")
        promoters_df.to_csv(self.output_dir / "predicted_promoters.csv", index=False)
        
        # Save summary statistics
        summary = {
            'total_promoters': len(all_promoters),
            'confidence_distribution': promoters_df['confidence'].value_counts().to_dict(),
            'score_statistics': {
                'ensemble_score': {
                    'mean': float(promoters_df['ensemble_score'].mean()),
                    'std': float(promoters_df['ensemble_score'].std()),
                    'min': float(promoters_df['ensemble_score'].min()),
                    'max': float(promoters_df['ensemble_score'].max())
                }
            },
            'config': {
                'window_size': self.config.window_size,
                'stride': self.config.stride,
                'smoothing_sigma': self.config.smoothing_sigma,
                'promoter_threshold': self.config.promoter_threshold
            }
        }
        
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save track files for genome browser
        if self.config.save_track_files:
            self.save_bed_track(all_promoters)
            self.save_gff3_annotation(all_promoters)
            self.save_bigwig_track(all_results)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def save_bed_track(self, promoters: List[PromoterPrediction]):
        """Save BED file for genome browser visualization"""
        
        bed_lines = []
        for promoter in promoters:
            # BED format: chrom, start, end, name, score, strand
            name = f"promoter_{promoter.center}"
            score = int(promoter.ensemble_score * 1000)
            bed_lines.append(f"{promoter.contig}\t{promoter.start}\t{promoter.end}\t{name}\t{score}\t.")
        
        with open(self.output_dir / "promoters.bed", "w") as f:
            f.write("\n".join(bed_lines))
        
        logger.info(f"BED track saved to {self.output_dir}/promoters.bed")
    
    def save_gff3_annotation(self, promoters: List[PromoterPrediction]):
        """Save GFF3 annotation file"""
        
        gff_lines = [
            "##gff-version 3",
            f"##date {pd.Timestamp.now().strftime('%Y-%m-%d')}",
            "##source-version PromoterPrediction_v1.0"
        ]
        
        for i, promoter in enumerate(promoters, 1):
            attributes = f"ID=promoter_{i};Name=PredictedPromoter_{i};score={promoter.ensemble_score:.3f};confidence={promoter.confidence}"
            gff_lines.append(
                f"{promoter.contig}\tSlidingWindowEngine\tpromoter\t{promoter.start+1}\t{promoter.end}\t"
                f"{promoter.ensemble_score:.3f}\t.\t.\t{attributes}"
            )
        
        with open(self.output_dir / "promoters.gff3", "w") as f:
            f.write("\n".join(gff_lines))
        
        logger.info(f"GFF3 annotation saved to {self.output_dir}/promoters.gff3")
    
    def save_bigwig_track(self, all_results: Dict):
        """Save BigWig track for continuous scores"""
        
        # This requires pyBigWig package
        try:
            import pyBigWig
            
            for contig_id, result in all_results.items():
                bw_path = self.output_dir / f"{contig_id}_scores.bw"
                bw = pyBigWig.open(str(bw_path), 'w')
                
                # Create chromosome header
                contig_length = len(self.scanner.contigs[contig_id])
                bw.addHeader([(contig_id, contig_length)])
                
                # Add scores
                positions = result['positions']
                ensemble_scores = result['predictions']['ensemble']
                
                # Convert to list of intervals
                intervals = [(contig_id, int(pos), int(pos + self.config.window_size), 
                             float(score)) for pos, score in zip(positions, ensemble_scores)]
                
                bw.addEntries(intervals, type='bedGraph')
                bw.close()
                
            logger.info(f"BigWig tracks saved to {self.output_dir}/")
        except ImportError:
            logger.warning("pyBigWig not installed. Skipping BigWig track generation.")
    
    def generate_visualizations(self, all_results: Dict, all_promoters: List[PromoterPrediction]):
        """Generate comprehensive visualizations"""
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Plot 1: Genome-wide promoter score distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score distribution histogram
        all_scores = []
        for result in all_results.values():
            all_scores.extend(result['predictions']['ensemble'])
        
        axes[0, 0].hist(all_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.config.promoter_threshold, color='red', linestyle='--', 
                          label=f'Threshold ({self.config.promoter_threshold})')
        axes[0, 0].set_xlabel('Ensemble Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Promoter Scores')
        axes[0, 0].legend()
        
        # Confidence pie chart
        confidence_counts = pd.Series([p.confidence for p in all_promoters]).value_counts()
        axes[0, 1].pie(confidence_counts.values, labels=confidence_counts.index, 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title(f'Predicted Promoter Confidence ({len(all_promoters)} total)')
        
        # Score vs length scatter
        promoters_df = pd.DataFrame([{
            'score': p.ensemble_score,
            'length': len(p.sequence),
            'confidence': p.confidence
        } for p in all_promoters])
        
        scatter = axes[1, 0].scatter(promoters_df['length'], promoters_df['score'], 
                                     c=promoters_df['score'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_xlabel('Promoter Region Length (bp)')
        axes[1, 0].set_ylabel('Ensemble Score')
        axes[1, 0].set_title('Promoter Score vs Length')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Model correlation heatmap
        model_scores = []
        for result in all_results.values():
            if all(model in result['predictions'] for model in ['prokbert', 'nt_lora', 'cnn']):
                model_scores.append(pd.DataFrame({
                    'ProkBERT': result['predictions']['prokbert'],
                    'NT+LoRA': result['predictions']['nt_lora'],
                    'CNN': result['predictions']['cnn']
                }))
        
        if model_scores:
            all_model_scores = pd.concat(model_scores)
            corr_matrix = all_model_scores.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
            axes[1, 1].set_title('Model Prediction Correlations')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "genome_wide_statistics.png", dpi=150)
        plt.close()
        
        # Plot 2: Per-contig score profiles (top 5 contigs)
        n_contigs = min(5, len(all_results))
        fig, axes = plt.subplots(n_contigs, 1, figsize=(15, 3*n_contigs))
        if n_contigs == 1:
            axes = [axes]
        
        for idx, (contig_id, result) in enumerate(list(all_results.items())[:n_contigs]):
            positions = np.array(result['positions'])
            scores = result['predictions']['ensemble']
            
            axes[idx].plot(positions, scores, 'b-', alpha=0.7, linewidth=1)
            axes[idx].axhline(self.config.promoter_threshold, color='r', linestyle='--', alpha=0.5)
            axes[idx].set_ylabel('Promoter Score')
            axes[idx].set_title(f'{contig_id} - Promoter Score Profile')
            axes[idx].set_xlim([positions.min(), positions.max()])
            
            # Mark detected promoters
            for promoter in result['promoters']:
                axes[idx].axvspan(promoter.start, promoter.end, alpha=0.2, color='green')
                axes[idx].axvline(promoter.center, color='green', linestyle='-', alpha=0.5)
        
        axes[-1].set_xlabel('Genomic Position (bp)')
        plt.tight_layout()
        plt.savefig(viz_dir / "contig_score_profiles.png", dpi=150)
        plt.close()
        
        # Plot 3: Promoter score heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        all_positions = []
        all_scores = []
        contig_labels = []
        
        for contig_id, result in all_results.items():
            positions = np.array(result['positions'])
            scores = result['predictions']['ensemble']
            
            # Normalize positions within contig
            norm_positions = (positions - positions.min()) / (positions.max() - positions.min())
            
            all_positions.extend(norm_positions)
            all_scores.extend(scores)
            contig_labels.extend([contig_id] * len(positions))
        
        # Create dataframe for heatmap
        heatmap_df = pd.DataFrame({
            'position': all_positions,
            'score': all_scores,
            'contig': contig_labels
        })
        
        # Pivot for heatmap
        pivot_df = heatmap_df.pivot_table(index='contig', columns='position', values='score')
        
        sns.heatmap(pivot_df, ax=ax, cmap='YlOrRd', cbar_kws={'label': 'Promoter Score'})
        ax.set_title('Genome-wide Promoter Score Heatmap')
        ax.set_xlabel('Normalized Genomic Position')
        ax.set_ylabel('Contig')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "promoter_heatmap.png", dpi=150)
        plt.close()
        
        # Plot 4: Top promoter regions with sequence logos
        top_promoters = sorted(all_promoters, key=lambda x: x.ensemble_score, reverse=True)[:10]
        
        # For the top promoter, create a detailed view
        if top_promoters:
            top = top_promoters[0]
            fig, axes = plt.subplots(2, 1, figsize=(15, 8))
            
            # Get scores for this region
            for contig_id, result in all_results.items():
                if contig_id == top.contig:
                    positions = np.array(result['positions'])
                    scores = result['predictions']['ensemble']
                    
                    # Subset to region around promoter
                    mask = (positions >= top.start - 200) & (positions <= top.end + 200)
                    plot_positions = positions[mask]
                    plot_scores = scores[mask]
                    
                    axes[0].plot(plot_positions, plot_scores, 'b-', linewidth=2)
                    axes[0].axhline(self.config.promoter_threshold, color='r', linestyle='--', alpha=0.5)
                    axes[0].axvspan(top.start, top.end, alpha=0.3, color='green')
                    axes[0].axvline(top.center, color='green', linestyle='-', linewidth=2)
                    axes[0].set_ylabel('Promoter Score')
                    axes[0].set_title(f'Top Promoter Region: {top.contig}:{top.start}-{top.end} (Score: {top.ensemble_score:.3f})')
                    axes[0].set_xlabel('Genomic Position (bp)')
                    
                    # Add sequence text
                    axes[1].text(0.5, 0.5, top.sequence[:200] + '...', 
                                transform=axes[1].transAxes, 
                                ha='center', va='center',
                                fontfamily='monospace', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    axes[1].set_title('Promoter Sequence (first 200 bp)')
                    axes[1].axis('off')
                    
                    break
            
            plt.tight_layout()
            plt.savefig(viz_dir / "top_promoter_detail.png", dpi=150)
            plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")

# ============================================================================
# Motif Analysis
# ============================================================================
class MotifAnalyzer:
    """Analyze detected promoters for known motifs"""
    
    def __init__(self):
        self.sigmaA_pwm = self._load_sigmaA_pwm()
        self.sigma70_pwm = self._load_sigma70_pwm()
        
    def _load_sigmaA_pwm(self):
        """Load sigmaA PWM matrix"""
        # -10 and -35 box motifs for sigmaA
        # This is a simplified version - replace with actual PWM from literature
        return {
            '-35': np.array([
                [0.1, 0.7, 0.1, 0.1],  # TTGACA motif
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.7],
                [0.7, 0.1, 0.1, 0.1]
            ]),
            '-10': np.array([
                [0.1, 0.1, 0.1, 0.7],  # TATAAT motif
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.7, 0.1, 0.1, 0.1],
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1]
            ])
        }
    
    def _load_sigma70_pwm(self):
        """Load sigma70 PWM matrix for E. coli (reference)"""
        # Placeholder for sigma70 motifs
        return self._load_sigmaA_pwm()
    
    def scan_for_motifs(self, sequence: str) -> Dict:
        """Scan sequence for promoter motifs"""
        
        motifs = {
            'sigmaA_-35': None,
            'sigmaA_-10': None,
            'sigma70_-35': None,
            'sigma70_-10': None
        }
        
        # Scan for sigmaA -35 box
        for i in range(len(sequence) - 6):
            subseq = sequence[i:i+6]
            score = self._score_motif(subseq, self.sigmaA_pwm['-35'])
            if score > 5.0:  # Threshold
                motifs['sigmaA_-35'] = {
                    'position': i,
                    'sequence': subseq,
                    'score': score
                }
                break
        
        # Scan for sigmaA -10 box (typically 17-18 bp downstream)
        if motifs['sigmaA_-35']:
            start = motifs['sigmaA_-35']['position'] + 17
            for i in range(start, min(start + 5, len(sequence) - 6)):
                subseq = sequence[i:i+6]
                score = self._score_motif(subseq, self.sigmaA_pwm['-10'])
                if score > 5.0:
                    motifs['sigmaA_-10'] = {
                        'position': i,
                        'sequence': subseq,
                        'score': score,
                        'spacing': i - motifs['sigmaA_-35']['position']
                    }
                    break
        
        return motifs
    
    def _score_motif(self, sequence: str, pwm: np.ndarray) -> float:
        """Score a sequence against a PWM"""
        if len(sequence) != pwm.shape[0]:
            return 0.0
        
        score = 0.0
        nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        for i, nt in enumerate(sequence):
            if nt in nucleotides:
                score += np.log2(pwm[i, nucleotides[nt]] / 0.25)
        
        return score

# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function"""
    
    # Configuration
    config = SlidingWindowConfig(
        window_size=300,
        stride=10,
        models_to_use=['prokbert', 'nt_lora', 'cnn'],
        smoothing_sigma=2.0,
        min_peak_distance=50,
        promoter_threshold=0.5,
        output_dir="results/sliding_window",
        save_predictions=True,
        save_track_files=True,
        generate_plots=True,
        batch_size=64,
        upstream_only=True,
        upstream_buffer=500
    )
    
    # Initialize engine
    engine = SlidingWindowEngine(config)
    
    # Run genome scan
    # Update paths to your actual genome and upstream regions files
    genome_path = "data/genomes/clostridium_acetobutylicum_ATCC824.fna"
    upstream_file = "results/upstream_regions/upstream_300bp.parquet"  # From T13
    
    results, promoters = engine.run_genome_scan(genome_path, upstream_file)
    
    # Run motif analysis on top promoters
    motif_analyzer = MotifAnalyzer()
    
    # Add motif information to promoters
    for promoter in promoters[:50]:  # Top 50 promoters
        motifs = motif_analyzer.scan_for_motifs(promoter.sequence)
        promoter.motif_match = motifs
    
    # Save enriched promoters with motif information
    enriched_promoters = []
    for promoter in promoters:
        if promoter.motif_match and any(promoter.motif_match.values()):
            enriched_promoters.append({
                'contig': promoter.contig,
                'start': promoter.start,
                'end': promoter.end,
                'ensemble_score': promoter.ensemble_score,
                'has_sigmaA_motif': bool(promoter.motif_match.get('sigmaA_-35')),
                'motif_details': str(promoter.motif_match)
            })
    
    enriched_df = pd.DataFrame(enriched_promoters)
    enriched_df.to_parquet(config.output_dir / "enriched_promoters_with_motifs.parquet")
    
    logger.info(f"Found {len(enriched_promoters)} promoters with known motifs")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SLIDING WINDOW INFERENCE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total promoters detected: {len(promoters)}")
    logger.info(f"High confidence: {sum(1 for p in promoters if p.confidence == 'high')}")
    logger.info(f"Medium confidence: {sum(1 for p in promoters if p.confidence == 'medium')}")
    logger.info(f"Low confidence: {sum(1 for p in promoters if p.confidence == 'low')}")
    logger.info(f"Promoters with sigmaA motifs: {len(enriched_promoters)}")
    
    # Output top 10 promoters
    logger.info("\nTop 10 Predicted Promoters:")
    top_promoters = sorted(promoters, key=lambda x: x.ensemble_score, reverse=True)[:10]
    for i, promoter in enumerate(top_promoters, 1):
        logger.info(f"{i}. {promoter.contig}:{promoter.start}-{promoter.end} "
                   f"(Score: {promoter.ensemble_score:.3f}, Confidence: {promoter.confidence})")

if __name__ == "__main__":
    main()





# task28_sliding_window/export_for_genome_browser.py
"""
Export predictions for genome browser visualization (JBrowse/IGV)
"""

import pandas as pd
import json
from pathlib import Path

def create_jbrowse_config(promoters_df, output_dir: str):
    """Create JBrowse 2 configuration files"""
    
    jbrowse_dir = Path(output_dir) / "jbrowse"
    jbrowse_dir.mkdir(parents=True, exist_ok=True)
    
    # Create track configuration
    track_config = {
        "type": "FeatureTrack",
        "trackId": "promoter_predictions",
        "name": "Predicted Promoters",
        "assemblyNames": ["Clostridium_acetobutylicum"],
        "category": ["Machine Learning Predictions"],
        "displays": [
            {
                "type": "LinearFeatureDisplay",
                "displayId": "promoter_predictions_display",
                "renderer": {
                    "type": "SvgFeatureRenderer",
                    "color1": "green"
                }
            }
        ]
    }
    
    with open(jbrowse_dir / "track_config.json", "w") as f:
        json.dump(track_config, f, indent=2)
    
    # Export as JSON lines for JBrowse
    features = []
    for _, row in promoters_df.iterrows():
        feature = {
            "refName": row['contig'],
            "start": row['start'],
            "end": row['end'],
            "type": "promoter",
            "score": row['ensemble_score'],
            "name": f"promoter_{row['center']}",
            "confidence": row['confidence']
        }
        features.append(feature)
    
    with open(jbrowse_dir / "promoters.jsonl", "w") as f:
        for feature in features:
            f.write(json.dumps(feature) + "\n")
    
    print(f"JBrowse files saved to {jbrowse_dir}")

def main():
    """Export for genome browsers"""
    
    # Load predictions
    promoters_df = pd.read_parquet("results/sliding_window/predicted_promoters.parquet")
    
    # Create JBrowse config
    create_jbrowse_config(promoters_df, "results/sliding_window")
    
    # Export as GFF3 (already done in main engine)
    print("Export complete!")

if __name__ == "__main__":
    main()
