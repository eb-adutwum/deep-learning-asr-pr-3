"""
Dataset class for AfriSpeech200.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from .preprocessing import AudioPreprocessor, TextPreprocessor
from .augmentation import SpecAugment, TimeMask, FreqMask

class AfriSpeechDataset(Dataset):
    """Dataset class for AfriSpeech200 Twi data."""
    
    def __init__(
        self,
        csv_path: str,
        audio_base_dir: str,
        audio_preprocessor: AudioPreprocessor,
        text_preprocessor: Optional[TextPreprocessor] = None,
        processor=None,
        model_type: str = 'whisper',
        max_audio_length: Optional[float] = None,
        min_audio_length: float = 0.5,
        use_augmentation: bool = False,
        split: str = 'train'
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file with transcripts
            audio_base_dir: Base directory for audio files
            audio_preprocessor: Audio preprocessor instance
            text_preprocessor: Text preprocessor instance
            max_audio_length: Maximum audio length in seconds (None = no limit)
            min_audio_length: Minimum audio length in seconds
            use_augmentation: Whether to apply augmentation
            split: Dataset split ('train', 'dev', 'test')
        """
        self.csv_path = csv_path
        self.audio_base_dir = audio_base_dir
        self.audio_preprocessor = audio_preprocessor
        self.text_preprocessor = text_preprocessor
        self.processor = processor
        self.model_type = model_type
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.use_augmentation = use_augmentation and (split == 'train')
        self.split = split
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Filter by audio length if specified
        if 'duration' in self.df.columns:
            if min_audio_length is not None:
                self.df = self.df[self.df['duration'] >= min_audio_length]
            if max_audio_length is not None:
                self.df = self.df[self.df['duration'] <= max_audio_length]
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        # Initialize augmentation
        if self.use_augmentation:
            self.spec_augment = SpecAugment(
                freq_mask_param=27,
                time_mask_param=100,
                num_freq_masks=2,
                num_time_masks=2
            )
            self.time_mask = TimeMask(mask_param=50)
            self.freq_mask = FreqMask(mask_param=27)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def _get_audio_path(self, row: pd.Series) -> str:
        """Get full path to audio file."""
        # Handle different path formats
        audio_path = row.get('audio_paths', '')
        if pd.isna(audio_path) or audio_path == '':
            # Try constructing from other fields
            audio_id = row.get('audio_ids', '')
            split = row.get('split', self.split)  # Use dataset split as fallback
            if split:
                return os.path.join(
                    self.audio_base_dir,
                    split,
                    'data',
                    'data',
                    f"{audio_id}.wav"
                )
            return None
        
        # Remove leading slash if present
        if audio_path.startswith('/'):
            audio_path = audio_path[1:]
        
        split = row.get('split', self.split or 'train')
        parts = audio_path.split('/')
        parent_dir = parts[-2] if len(parts) >= 2 else ''
        filename = parts[-1]
        
        candidate_paths = [
            os.path.join(self.audio_base_dir, audio_path),
            os.path.join(
                self.audio_base_dir,
                split,
                'data',
                'data',
                parent_dir,
                filename
            ),
            os.path.join(
                self.audio_base_dir,
                split,
                'data',
                'data',
                'intron',
                parent_dir,
                filename
            )
        ]
        
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return candidate
        
        return candidate_paths[0]
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with 'audio', 'transcript', 'text_ids', 'audio_length', 'text_length'
        """
        row = self.df.iloc[idx]
        
        # Get audio path and load
        audio_path = self._get_audio_path(row)
        if audio_path is None or not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        # Load waveform
        waveform, sample_rate = self.audio_preprocessor.load_audio(audio_path)
        waveform = self.audio_preprocessor.resample(waveform, sample_rate)
        waveform_np = waveform.squeeze(0).cpu().numpy()
        
        # Feature extraction
        attention_mask = None
        if self.processor is not None and hasattr(self.processor, 'feature_extractor'):
            features = self.processor.feature_extractor(
                waveform_np,
                sampling_rate=self.audio_preprocessor.sample_rate,
                return_tensors="pt"
            )
            if self.model_type == 'wav2vec2':
                audio_features = features.input_values[0]
                attention_mask = features.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask[0]
            else:
                audio_features = features.input_features[0]
                attention_mask = features.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask[0]
        else:
            if self.model_type == 'wav2vec2':
                if self.audio_preprocessor.normalize:
                    waveform = self.audio_preprocessor.normalize_waveform(waveform)
                audio_features = waveform.squeeze(0)
            else:
                audio_features = self.audio_preprocessor.preprocess_waveform(
                    waveform,
                    self.audio_preprocessor.sample_rate
                )
        
        # Apply augmentation if training on spectrogram features
        if self.use_augmentation and audio_features.dim() == 2:
            if torch.rand(1).item() > 0.5:  # 50% chance
                audio_features = self.spec_augment(audio_features)
        
        # Get transcript
        transcript = str(row.get('transcript', '')).strip()
        if not transcript:
            transcript = ' '  # Empty string fallback
        
        # Convert text to IDs
        if self.processor is not None and hasattr(self.processor, 'tokenizer'):
            if self.model_type == 'wav2vec2':
                with self.processor.as_target_processor():
                    tokenized = self.processor(
                        transcript,
                        return_tensors='pt'
                    )
                    text_ids = tokenized.input_ids[0]
            else:
                tokenized = self.processor.tokenizer(
                    transcript,
                    return_tensors='pt'
                )
                text_ids = tokenized.input_ids[0]
        elif self.text_preprocessor is not None:
            text_ids = torch.tensor(
                self.text_preprocessor.text_to_ids(transcript),
                dtype=torch.long
            )
        else:
            raise ValueError("No tokenizer available. Provide a processor or text_preprocessor.")
        
        if attention_mask is None:
            mask_length = audio_features.shape[-1]
            attention_mask = torch.zeros(mask_length, dtype=torch.long)
            attention_mask[:mask_length] = 1
        
        return {
            'audio': audio_features,
            'transcript': transcript,
            'text_ids': text_ids.long(),
            'audio_length': audio_features.shape[-1],
            'text_length': len(text_ids),
            'attention_mask': attention_mask.long()
        }


def collate_fn(
    batch: list,
    pad_token_id: int = 0,
    label_pad_token_id: int = -100
) -> dict:
    """
    Collate function for DataLoader.
    Pads sequences to same length.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched dictionary
    """
    # Find max lengths
    max_audio_length = max([item['audio_length'] for item in batch])
    max_text_length = max([item['text_length'] for item in batch])
    
    # Get dimensions
    sample_audio = batch[0]['audio']
    batch_size = len(batch)
    
    # Initialize tensors
    if sample_audio.dim() == 1:
        audio_batch = torch.zeros(batch_size, max_audio_length, dtype=sample_audio.dtype)
    else:
        n_mels = sample_audio.shape[0]
        audio_batch = torch.zeros(
            batch_size,
            n_mels,
            max_audio_length,
            dtype=sample_audio.dtype
        )
    text_batch = torch.full(
        (batch_size, max_text_length),
        fill_value=pad_token_id,
        dtype=torch.long
    )
    label_batch = torch.full(
        (batch_size, max_text_length),
        fill_value=label_pad_token_id,
        dtype=torch.long
    )
    audio_lengths = torch.zeros(batch_size, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, dtype=torch.long)
    attention_batch = torch.zeros(batch_size, max_audio_length, dtype=torch.long)
    transcripts = []
    
    for i, item in enumerate(batch):
        audio = item['audio']
        text_ids = item['text_ids']
        
        # Pad audio
        audio_length = audio.shape[-1]
        if audio.dim() == 1:
            audio_batch[i, :audio_length] = audio
        else:
            audio_batch[i, :, :audio_length] = audio
        audio_lengths[i] = audio_length
        
        # Attention mask
        attention = item.get('attention_mask')
        if attention is None:
            attention = torch.ones(audio_length, dtype=torch.long)
        attn_len = min(attention.shape[-1], max_audio_length)
        attention_batch[i, :attn_len] = attention[:attn_len]
        
        # Pad text
        text_length = len(text_ids)
        if text_length > 0:
            text_batch[i, :text_length] = text_ids
            label_batch[i, :text_length] = text_ids
        text_lengths[i] = text_length
        
        transcripts.append(item['transcript'])
    
    return {
        'audio': audio_batch,
        'text_ids': text_batch,
        'labels': label_batch,
        'audio_lengths': audio_lengths,
        'text_lengths': text_lengths,
        'attention_mask': attention_batch,
        'transcripts': transcripts
    }

