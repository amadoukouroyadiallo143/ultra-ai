import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator, Tuple
import logging
import random
from PIL import Image
import torchaudio
import cv2
from transformers import AutoTokenizer
import h5py

logger = logging.getLogger(__name__)


class ProcessedDataset(Dataset):
    """
    Dataset pour charger les données déjà préprocessées (fichiers .pt).
    Utilise les données tokenisées sauvées par le script preprocess_hf_data.py.
    """
    
    def __init__(self, processed_data_path: str):
        self.processed_data_path = Path(processed_data_path)
        self.samples = []
        self._load_processed_data()
        logger.info(f"Loaded ProcessedDataset with {len(self.samples)} samples")
        
    def _load_processed_data(self):
        """Charge les données préprocessées depuis les fichiers .pt"""
        # Charger le manifest d'entraînement
        manifest_path = self.processed_data_path / "training_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        # Vérifier si c'est le manifest traité (avec 'datasets') ou non traité (avec 'modalities')
        datasets_key = 'datasets' if 'datasets' in manifest else 'modalities'
        
        # Charger tous les chunks de tous les datasets
        for modality, datasets in manifest[datasets_key].items():
            for dataset_name, dataset_info in datasets.items():
                chunk_files = dataset_info.get('chunk_files', [])
                
                for chunk_file in chunk_files:
                    # Convertir le chemin Windows en chemin correct
                    chunk_path = self.processed_data_path / chunk_file.replace('\\', '/').replace('data/processed/', '')
                    
                    if chunk_path.exists():
                        try:
                            chunk_data = torch.load(chunk_path, map_location='cpu')
                            self.samples.extend(chunk_data)
                            logger.info(f"Loaded {len(chunk_data)} samples from {chunk_path}")
                        except Exception as e:
                            logger.error(f"Error loading {chunk_path}: {e}")
                    else:
                        logger.warning(f"Chunk file not found: {chunk_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retourne un échantillon préprocessé"""
        sample = self.samples[idx]
        
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(sample['input_ids'], dtype=torch.long)  # Pour language modeling
        }


class UltraDataset(Dataset):
    """
    Ultra-AI dataset supporting multimodal data (text, image, audio, video).
    Designed for ultra-long context training up to 100M tokens.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "microsoft/DialoGPT-large",
        max_seq_length: int = 100000,
        modalities: List[str] = ["text", "image", "audio", "video"],
        image_size: int = 224,
        audio_sample_rate: int = 16000,
        video_frames: int = 8,
        cache_size: int = 1000,
    ):
        self.data_path = Path(data_path)
        self.max_seq_length = max_seq_length
        self.modalities = modalities
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        self.video_frames = video_frames
        self.cache_size = cache_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load dataset metadata
        self.samples = self._load_dataset_metadata()
        
        # Initialize cache
        self.cache = {}
        self.cache_order = []
        
        logger.info(f"Loaded UltraDataset with {len(self.samples)} samples")
        
    def _load_dataset_metadata(self) -> List[Dict[str, Any]]:
        """Load dataset metadata from JSON files."""
        samples = []
        
        # Look for metadata files
        metadata_files = list(self.data_path.glob("*.json"))
        
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                file_samples = json.load(f)
                
            # Validate and filter samples based on available modalities
            for sample in file_samples:
                validated_sample = self._validate_sample(sample)
                if validated_sample:
                    samples.append(validated_sample)
                    
        return samples
        
    def _validate_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate that sample has required modality data."""
        validated = {"id": sample.get("id", len(self.samples))}
        
        # Check text data
        if "text" in self.modalities and "text" in sample:
            validated["text"] = sample["text"]
            
        # Check image data
        if "image" in self.modalities and "image_path" in sample:
            image_path = self.data_path / sample["image_path"]
            if image_path.exists():
                validated["image_path"] = str(image_path)
                
        # Check audio data
        if "audio" in self.modalities and "audio_path" in sample:
            audio_path = self.data_path / sample["audio_path"]
            if audio_path.exists():
                validated["audio_path"] = str(audio_path)
                
        # Check video data
        if "video" in self.modalities and "video_path" in sample:
            video_path = self.data_path / sample["video_path"]
            if video_path.exists():
                validated["video_path"] = str(video_path)
                
        # Return sample if it has at least one modality
        if len(validated) > 1:
            return validated
        else:
            return None
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single multimodal sample."""
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
            
        sample_meta = self.samples[idx]
        sample = {}
        
        # Process text
        if "text" in sample_meta:
            text_data = self._process_text(sample_meta["text"])
            sample.update(text_data)
            
        # Process image
        if "image_path" in sample_meta:
            image_data = self._process_image(sample_meta["image_path"])
            sample["image"] = image_data
            
        # Process audio
        if "audio_path" in sample_meta:
            audio_data = self._process_audio(sample_meta["audio_path"])
            sample["audio"] = audio_data
            
        # Process video
        if "video_path" in sample_meta:
            video_data = self._process_video(sample_meta["video_path"])
            sample["video"] = video_data
            
        # Add to cache
        self._add_to_cache(idx, sample)
        
        return sample
        
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text data with tokenization."""
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
    def _process_image(self, image_path: str) -> torch.Tensor:
        """Process image data."""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Resize image
            image = image.resize((self.image_size, self.image_size))
            
            # Convert to tensor and normalize
            image_array = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # CHW
            
            return image_tensor
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return dummy image
            return torch.zeros(3, self.image_size, self.image_size)
            
    def _process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio data."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.audio_sample_rate)
                waveform = resampler(waveform)
                
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # Compute mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.audio_sample_rate,
                n_mels=80,
                n_fft=1024,
                hop_length=320,
            )
            
            mel_spec = mel_transform(waveform)
            
            return mel_spec.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            # Return dummy audio
            return torch.zeros(80, 1000)  # Dummy mel spectrogram
            
    def _process_video(self, video_path: str) -> torch.Tensor:
        """Process video data."""
        try:
            # Load video
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Extract frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, frame_count - 1, self.video_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                    # Normalize
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                else:
                    # Add dummy frame if reading fails
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.float32))
                    
            cap.release()
            
            # Convert to tensor (frames, channels, height, width)
            video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
            
            return video_tensor
            
        except Exception as e:
            logger.warning(f"Error loading video {video_path}: {e}")
            # Return dummy video
            return torch.zeros(self.video_frames, 3, self.image_size, self.image_size)
            
    def _add_to_cache(self, idx: int, sample: Dict[str, torch.Tensor]):
        """Add sample to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
            
        self.cache[idx] = sample
        self.cache_order.append(idx)


class MultimodalDataLoader:
    """
    Advanced data loader for multimodal ultra-long context data.
    Supports dynamic batching and memory-efficient loading.
    """
    
    def __init__(
        self,
        dataset: UltraDataset,
        batch_size: int = 1,  # Start with small batch size for ultra-long sequences
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn: Optional[callable] = None,
        sequence_bucketing: bool = True,
        max_tokens_per_batch: int = 1000000,  # 1M tokens per batch
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.sequence_bucketing = sequence_bucketing
        self.max_tokens_per_batch = max_tokens_per_batch
        
        # Custom collate function
        if collate_fn is None:
            self.collate_fn = self._multimodal_collate
        else:
            self.collate_fn = collate_fn
            
        # Create buckets for sequence length if enabled
        if self.sequence_bucketing:
            self.buckets = self._create_buckets()
        else:
            self.buckets = None
            
    def _create_buckets(self) -> Dict[int, List[int]]:
        """Create buckets based on sequence lengths."""
        buckets = {}
        
        # Sample sequence lengths (this is expensive, so we sample)
        sample_indices = random.sample(range(len(self.dataset)), min(1000, len(self.dataset)))
        
        for idx in sample_indices:
            sample = self.dataset[idx]
            
            if "input_ids" in sample:
                seq_len = sample["attention_mask"].sum().item()
                bucket_key = (seq_len // 1000) * 1000  # Round to nearest 1000
                
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(idx)
                
        logger.info(f"Created {len(buckets)} sequence buckets")
        return buckets
        
    def _multimodal_collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for multimodal data with proper padding."""
        collated = {}
        
        # Collate text data with padding
        if "input_ids" in batch[0]:
            # Find max sequence length in this batch
            max_len = max(item["input_ids"].size(0) for item in batch)
            
            # Pad all sequences to max length
            input_ids = []
            attention_masks = []
            
            for item in batch:
                input_id = item["input_ids"]
                attention_mask = item["attention_mask"]
                
                # Pad if needed
                if input_id.size(0) < max_len:
                    pad_len = max_len - input_id.size(0)
                    input_id = torch.cat([input_id, torch.zeros(pad_len, dtype=input_id.dtype)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
            
            collated.update({
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_masks),
                "labels": torch.stack(input_ids),  # Pour language modeling
            })
            
        # Collate image data
        if "image" in batch[0]:
            images = torch.stack([item["image"] for item in batch])
            collated["images"] = images
            
        # Collate audio data
        if "audio" in batch[0]:
            # Pad audio to same length
            max_audio_len = max(item["audio"].shape[-1] for item in batch)
            padded_audio = []
            
            for item in batch:
                audio = item["audio"]
                if audio.shape[-1] < max_audio_len:
                    padding = torch.zeros(audio.shape[0], max_audio_len - audio.shape[-1])
                    audio = torch.cat([audio, padding], dim=-1)
                padded_audio.append(audio)
                
            collated["audio"] = torch.stack(padded_audio)
            
        # Collate video data
        if "video" in batch[0]:
            videos = torch.stack([item["video"] for item in batch])
            collated["videos"] = videos
            
        return collated
        
    def __iter__(self):
        """Create data loader iterator."""
        if self.sequence_bucketing and self.buckets:
            return self._bucketed_iterator()
        else:
            return self._standard_iterator()
            
    def _standard_iterator(self):
        """Standard data loader iterator."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn,
        )
        
        return iter(dataloader)
        
    def _bucketed_iterator(self):
        """Sequence bucketing iterator for efficient padding."""
        bucket_keys = list(self.buckets.keys())
        if self.shuffle:
            random.shuffle(bucket_keys)
            
        for bucket_key in bucket_keys:
            bucket_indices = self.buckets[bucket_key].copy()
            if self.shuffle:
                random.shuffle(bucket_indices)
                
            # Create batches from bucket
            for i in range(0, len(bucket_indices), self.batch_size):
                batch_indices = bucket_indices[i:i + self.batch_size]
                
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    batch = [self.dataset[idx] for idx in batch_indices]
                    yield self.collate_fn(batch)
                    
    def __len__(self):
        """Get number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class StreamingDataset(IterableDataset):
    """
    Streaming dataset for extremely large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        data_files: List[str],
        tokenizer_name: str = "microsoft/DialoGPT-large",
        max_seq_length: int = 100000,
        buffer_size: int = 10000,
    ):
        self.data_files = data_files
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through streaming data."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process
            file_list = self.data_files
        else:
            # Multi-process: split files across workers
            per_worker = len(self.data_files) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.data_files)
            file_list = self.data_files[start:end]
            
        for file_path in file_list:
            yield from self._process_file(file_path)
            
    def _process_file(self, file_path: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Process a single data file."""
        if file_path.endswith('.jsonl'):
            yield from self._process_jsonl(file_path)
        elif file_path.endswith('.h5'):
            yield from self._process_hdf5(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            
    def _process_jsonl(self, file_path: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Process JSONL file."""
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "text" in data:
                        yield self._tokenize_text(data["text"])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")
                    
    def _process_hdf5(self, file_path: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Process HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            if 'text' in f:
                texts = f['text']
                for i in range(len(texts)):
                    text = texts[i].decode('utf-8') if isinstance(texts[i], bytes) else texts[i]
                    yield self._tokenize_text(text)
                    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text data."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


class PretrainDataset(Dataset):
    """
    Dataset specifically for pretraining with ultra-long context.
    Implements dynamic document concatenation and context packing.
    """
    
    def __init__(
        self,
        text_files: List[str],
        tokenizer_name: str = "microsoft/DialoGPT-large",
        max_seq_length: int = 100000,
        min_seq_length: int = 1000,
        concatenate_documents: bool = True,
        shuffle_documents: bool = True,
    ):
        self.text_files = text_files
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.concatenate_documents = concatenate_documents
        self.shuffle_documents = shuffle_documents
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load and prepare data
        self.sequences = self._prepare_sequences()
        
        logger.info(f"Prepared {len(self.sequences)} sequences for pretraining")
        
    def _prepare_sequences(self) -> List[List[int]]:
        """Prepare tokenized sequences with document concatenation."""
        all_texts = []
        
        # Load all text files
        for file_path in self.text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                all_texts.append(data['text'])
                        except json.JSONDecodeError:
                            continue
                else:
                    # Plain text file
                    content = f.read()
                    # Split by paragraphs or chunks
                    chunks = content.split('\n\n')
                    all_texts.extend([chunk.strip() for chunk in chunks if chunk.strip()])
                    
        if self.shuffle_documents:
            random.shuffle(all_texts)
            
        # Tokenize and concatenate
        sequences = []
        current_sequence = []
        current_length = 0
        
        for text in all_texts:
            # Tokenize text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if self.concatenate_documents:
                # Add to current sequence if it fits
                if current_length + len(tokens) + 1 <= self.max_seq_length:  # +1 for separator
                    if current_sequence:  # Add separator between documents
                        current_sequence.append(self.tokenizer.eos_token_id)
                        current_length += 1
                    current_sequence.extend(tokens)
                    current_length += len(tokens)
                else:
                    # Save current sequence if it's long enough
                    if current_length >= self.min_seq_length:
                        sequences.append(current_sequence)
                        
                    # Start new sequence
                    current_sequence = tokens
                    current_length = len(tokens)
            else:
                # Each document is a separate sequence
                if len(tokens) >= self.min_seq_length:
                    sequences.append(tokens[:self.max_seq_length])
                    
        # Add final sequence
        if current_sequence and len(current_sequence) >= self.min_seq_length:
            sequences.append(current_sequence)
            
        return sequences
        
    def __len__(self) -> int:
        return len(self.sequences)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized sequence."""
        sequence = self.sequences[idx]
        
        # Pad sequence
        if len(sequence) < self.max_seq_length:
            padding_length = self.max_seq_length - len(sequence)
            sequence = sequence + [self.tokenizer.pad_token_id] * padding_length
            
        # Create attention mask
        attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in sequence]
        
        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }