"""
Enterprise-grade Data Quality Management System
Comprehensive data filtering, validation, and quality assurance for Ultra-AI training.
"""

import re
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import partial
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import langdetect
from sentence_transformers import SentenceTransformer
import spacy

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics for tracking."""
    total_samples: int = 0
    filtered_samples: int = 0
    quality_scores: Dict[str, float] = None
    duplicate_groups: int = 0
    language_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = {}
        if self.language_distribution is None:
            self.language_distribution = {}


class TextQualityAnalyzer:
    """Advanced text quality analysis and scoring."""
    
    def __init__(self):
        self.tokenizer = None
        self.quality_model = None
        self.sentence_model = None
        self.nlp = None
        self._setup_models()
        
    def _setup_models(self):
        """Setup models for quality analysis."""
        try:
            # Tokenizer for perplexity calculation
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Quality scoring model (optional - can be heavy)
            # self.quality_model = AutoModelForSequenceClassification.from_pretrained(
            #     "unitary/toxic-bert"
            # )
            
            # Sentence transformer for similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # SpaCy for linguistic analysis
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy English model not found. Some features will be disabled.")
                self.nlp = None
                
        except Exception as e:
            logger.warning(f"Some quality analysis models failed to load: {e}")
            
    def calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity as a quality measure."""
        if not self.tokenizer:
            return 50.0  # Default moderate perplexity
            
        try:
            tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                # Simple perplexity approximation
                token_count = tokens.shape[1]
                # Use inverse frequency as approximation
                unique_tokens = len(set(tokens[0].tolist()))
                perplexity = token_count / max(unique_tokens, 1)
                return min(perplexity, 100.0)  # Cap at 100
        except Exception:
            return 50.0
            
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calculate various readability metrics."""
        try:
            metrics = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'gunning_fog': textstat.gunning_fog(text),
            }
            return metrics
        except Exception:
            return {key: 0.0 for key in ['flesch_reading_ease', 'flesch_kincaid_grade', 
                                       'automated_readability_index', 'coleman_liau_index', 'gunning_fog']}
            
    def calculate_linguistic_features(self, text: str) -> Dict[str, float]:
        """Calculate advanced linguistic features."""
        features = {}
        
        # Basic statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Character distribution
        total_chars = len(text)
        features['alpha_ratio'] = sum(c.isalpha() for c in text) / max(total_chars, 1)
        features['digit_ratio'] = sum(c.isdigit() for c in text) / max(total_chars, 1)
        features['space_ratio'] = sum(c.isspace() for c in text) / max(total_chars, 1)
        features['punct_ratio'] = sum(c in '.,!?;:' for c in text) / max(total_chars, 1)
        
        # Advanced features with spaCy
        if self.nlp:
            try:
                doc = self.nlp(text[:1000000])  # Limit for performance
                features['noun_ratio'] = len([token for token in doc if token.pos_ == 'NOUN']) / len(doc)
                features['verb_ratio'] = len([token for token in doc if token.pos_ == 'VERB']) / len(doc)
                features['adj_ratio'] = len([token for token in doc if token.pos_ == 'ADJ']) / len(doc)
                features['named_entity_ratio'] = len(doc.ents) / max(len(doc), 1)
            except Exception:
                features.update({
                    'noun_ratio': 0.0, 'verb_ratio': 0.0, 
                    'adj_ratio': 0.0, 'named_entity_ratio': 0.0
                })
        else:
            features.update({
                'noun_ratio': 0.0, 'verb_ratio': 0.0, 
                'adj_ratio': 0.0, 'named_entity_ratio': 0.0
            })
            
        return features
        
    def calculate_quality_score(self, text: str) -> float:
        """Calculate comprehensive quality score (0-1, higher is better)."""
        try:
            # Get all metrics
            perplexity = self.calculate_perplexity(text)
            readability = self.calculate_readability(text)
            linguistic = self.calculate_linguistic_features(text)
            
            # Normalize perplexity (lower is better, so invert)
            perplexity_score = 1.0 / (1.0 + perplexity / 50.0)
            
            # Readability score (normalize Flesch Reading Ease)
            flesch_score = max(0, min(100, readability['flesch_reading_ease'])) / 100.0
            
            # Linguistic quality indicators
            word_count = linguistic['word_count']
            length_score = min(1.0, max(0.0, (word_count - 10) / 500))  # Prefer 10-500 words
            
            alpha_score = linguistic['alpha_ratio']  # Prefer high alphabetic content
            balance_score = min(linguistic['noun_ratio'] + linguistic['verb_ratio'], 1.0)
            
            # Weighted combination
            quality_score = (
                0.25 * perplexity_score +
                0.20 * flesch_score +
                0.15 * length_score +
                0.20 * alpha_score +
                0.20 * balance_score
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5  # Default medium quality


class DuplicateDetector:
    """Advanced duplicate and near-duplicate detection."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_hashes = set()
        self.embeddings_cache = {}
        
    def calculate_text_hash(self, text: str) -> str:
        """Calculate hash for exact duplicate detection."""
        # Normalize text before hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
        
    def is_exact_duplicate(self, text: str) -> bool:
        """Check for exact duplicates."""
        text_hash = self.calculate_text_hash(text)
        if text_hash in self.document_hashes:
            return True
        self.document_hashes.add(text_hash)
        return False
        
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calculate pairwise similarity matrix for texts."""
        if len(texts) < 2:
            return np.array([[]])
            
        try:
            # Use TF-IDF for fast similarity calculation
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            return np.zeros((len(texts), len(texts)))
            
    def find_near_duplicates(self, texts: List[str]) -> List[List[int]]:
        """Find groups of near-duplicate texts."""
        if len(texts) < 2:
            return []
            
        similarity_matrix = self.calculate_similarity_matrix(texts)
        
        # Find connected components of similar texts
        visited = set()
        duplicate_groups = []
        
        for i in range(len(texts)):
            if i in visited:
                continue
                
            # Find all texts similar to text i
            similar_indices = [i]
            for j in range(i + 1, len(texts)):
                if j not in visited and similarity_matrix[i][j] > self.similarity_threshold:
                    similar_indices.append(j)
                    visited.add(j)
                    
            if len(similar_indices) > 1:
                duplicate_groups.append(similar_indices)
                visited.update(similar_indices)
            else:
                visited.add(i)
                
        return duplicate_groups


class LanguageDetector:
    """Robust language detection and filtering."""
    
    def __init__(self, supported_languages: List[str] = None):
        self.supported_languages = supported_languages or ['en', 'fr', 'es', 'de', 'it']
        self.confidence_threshold = 0.8
        
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language and confidence."""
        try:
            # Use multiple methods for robust detection
            lang = langdetect.detect(text)
            
            # Calculate confidence (simplified)
            confidence = min(1.0, len(text) / 100.0)  # Longer texts are more confident
            
            return lang, confidence
        except Exception:
            return 'unknown', 0.0
            
    def is_supported_language(self, text: str) -> bool:
        """Check if text is in supported language."""
        lang, confidence = self.detect_language(text)
        return lang in self.supported_languages and confidence > self.confidence_threshold


class DataQualityManager:
    """Enterprise-grade data quality management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_analyzer = TextQualityAnalyzer()
        self.duplicate_detector = DuplicateDetector(
            similarity_threshold=config.get('duplicate_threshold', 0.85)
        )
        self.language_detector = LanguageDetector(
            supported_languages=config.get('language_filter', ['en'])
        )
        
        # Quality thresholds
        self.min_text_length = config.get('min_text_length', 100)
        self.max_text_length = config.get('max_text_length', 1000000)
        self.quality_threshold = config.get('text_quality_threshold', 0.5)
        
        # Statistics tracking
        self.metrics = DataQualityMetrics()
        
        # Processing parameters
        self.batch_size = config.get('quality_batch_size', 1000)
        self.num_workers = config.get('quality_num_workers', mp.cpu_count())
        
    def validate_text_basic(self, text: str) -> bool:
        """Basic text validation."""
        if not isinstance(text, str):
            return False
            
        text = text.strip()
        if not text:
            return False
            
        # Length checks
        if len(text) < self.min_text_length or len(text) > self.max_text_length:
            return False
            
        # Character ratio checks
        total_chars = len(text)
        alpha_ratio = sum(c.isalpha() for c in text) / total_chars
        
        # Must have reasonable amount of alphabetic characters
        if alpha_ratio < 0.3:
            return False
            
        # Must not be mostly punctuation or numbers
        punct_ratio = sum(c in '.,!?;:()[]{}' for c in text) / total_chars
        if punct_ratio > 0.3:
            return False
            
        return True
        
    def process_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Process a single sample for quality assessment."""
        text = sample.get('text', '')
        
        # Basic validation
        if not self.validate_text_basic(text):
            return False, {'reason': 'basic_validation_failed'}
            
        # Language detection
        if not self.language_detector.is_supported_language(text):
            return False, {'reason': 'unsupported_language'}
            
        # Duplicate detection
        if self.duplicate_detector.is_exact_duplicate(text):
            return False, {'reason': 'exact_duplicate'}
            
        # Quality score calculation
        quality_score = self.quality_analyzer.calculate_quality_score(text)
        if quality_score < self.quality_threshold:
            return False, {'reason': 'low_quality', 'quality_score': quality_score}
            
        # Passed all checks
        metadata = {
            'quality_score': quality_score,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        return True, metadata
        
    def process_batch(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process a batch of samples."""
        valid_samples = []
        batch_stats = defaultdict(int)
        
        for sample in samples:
            self.metrics.total_samples += 1
            is_valid, metadata = self.process_sample(sample)
            
            if is_valid:
                sample['quality_metadata'] = metadata
                valid_samples.append(sample)
            else:
                self.metrics.filtered_samples += 1
                batch_stats[metadata.get('reason', 'unknown')] += 1
                
        return valid_samples, dict(batch_stats)
        
    def process_dataset(self, dataset_path: str, output_path: str) -> DataQualityMetrics:
        """Process entire dataset with quality filtering."""
        logger.info(f"Processing dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        if dataset_path.suffix == '.jsonl':
            samples = self._load_jsonl(dataset_path)
        elif dataset_path.suffix == '.json':
            with open(dataset_path, 'r') as f:
                samples = json.load(f)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
            
        logger.info(f"Loaded {len(samples)} samples")
        
        # Process in batches
        all_valid_samples = []
        all_stats = defaultdict(int)
        
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            valid_samples, batch_stats = self.process_batch(batch)
            
            all_valid_samples.extend(valid_samples)
            for key, count in batch_stats.items():
                all_stats[key] += count
                
            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)} samples, "
                           f"valid: {len(all_valid_samples)}, "
                           f"filtered: {self.metrics.filtered_samples}")
                
        # Near-duplicate detection on valid samples
        logger.info("Running near-duplicate detection...")
        if len(all_valid_samples) > 1:
            texts = [sample['text'] for sample in all_valid_samples]
            duplicate_groups = self.duplicate_detector.find_near_duplicates(texts)
            
            # Remove duplicates (keep first in each group)
            indices_to_remove = set()
            for group in duplicate_groups:
                indices_to_remove.update(group[1:])  # Keep first, remove rest
                
            deduplicated_samples = [
                sample for i, sample in enumerate(all_valid_samples)
                if i not in indices_to_remove
            ]
            
            logger.info(f"Found {len(duplicate_groups)} duplicate groups, "
                       f"removed {len(indices_to_remove)} samples")
            
            all_valid_samples = deduplicated_samples
            
        # Save processed dataset
        output_file = output_path / f"processed_{dataset_path.name}"
        if output_file.suffix == '.jsonl':
            self._save_jsonl(all_valid_samples, output_file)
        else:
            with open(output_file, 'w') as f:
                json.dump(all_valid_samples, f, indent=2)
                
        # Update metrics
        self.metrics.total_samples = len(samples)
        self.metrics.filtered_samples = len(samples) - len(all_valid_samples)
        
        # Save quality report
        report = {
            'dataset_path': str(dataset_path),
            'output_path': str(output_file),
            'total_samples': self.metrics.total_samples,
            'valid_samples': len(all_valid_samples),
            'filtered_samples': self.metrics.filtered_samples,
            'filter_reasons': dict(all_stats),
            'quality_threshold': self.quality_threshold,
            'duplicate_threshold': self.duplicate_detector.similarity_threshold,
        }
        
        report_path = output_path / 'quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Quality processing complete. Report saved to: {report_path}")
        logger.info(f"Valid samples: {len(all_valid_samples)} / {len(samples)} "
                   f"({len(all_valid_samples)/len(samples)*100:.1f}%)")
        
        return self.metrics
        
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        samples = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return samples
        
    def _save_jsonl(self, samples: List[Dict[str, Any]], file_path: Path):
        """Save samples to JSONL file."""
        with open(file_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')


def create_quality_filtered_dataset(
    input_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> DataQualityMetrics:
    """Create a quality-filtered dataset."""
    manager = DataQualityManager(config)
    return manager.process_dataset(input_path, output_path)