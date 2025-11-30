"""
Configuration management for pretrained transformer ASR system.
Initial version - before any tweaks.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import torch


def _detect_device() -> str:
    """Detect the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

@dataclass
class DataConfig:
    """Configuration for data processing."""
    accent: str = 'twi'
    train_csv: str = '../../afrispeech200/transcripts/twi/train_main.csv'
    dev_csv: str = '../../afrispeech200/transcripts/twi/val_expanded.csv'
    test_csv: str = '../../afrispeech200/transcripts/twi/test.csv'
    audio_base_dir: str = '../../afrispeech200/audio/twi'
    
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    
    tokenizer_type: str = 'char'
    max_audio_length: Optional[float] = None
    min_audio_length: Optional[float] = None
    
    use_augmentation: bool = True
    speed_perturbation: bool = True
    noise_injection: bool = False
    time_mask: bool = True
    freq_mask: bool = True

@dataclass
class ModelConfig:
    """Configuration for pretrained model."""
    model_name: str = 'openai/whisper-base'
    freeze_feature_extractor: bool = True
    freeze_encoder: bool = True
    dropout: float = 0.1
    
    language: str = 'en'
    vocab_size: Optional[int] = None

@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 5e-5  # Initial learning rate
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    
    lr_scheduler_type: str = 'cosine'
    
    output_dir: str = 'checkpoints/pretrained_transformer1'
    save_steps: int = 400
    eval_steps: int = 200
    logging_steps: int = 50
    train_eval_subset: int = 0
    
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    fp16: bool = False
    gradient_accumulation_steps: int = 1

@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    seed: int = 42
    device: str = field(default_factory=_detect_device)
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        def _resolve(path: str) -> str:
            if path is None:
                return None
            if os.path.isabs(path):
                return path
            return os.path.abspath(os.path.join(base_dir, path))
        
        train_candidate = _resolve(self.data.train_csv)
        if not os.path.exists(train_candidate):
            train_candidate = _resolve('../../afrispeech200/transcripts/twi/train.csv')
        self.data.train_csv = train_candidate
        
        dev_candidate = _resolve(self.data.dev_csv)
        if not os.path.exists(dev_candidate):
            dev_candidate = _resolve('../../afrispeech200/transcripts/twi/dev.csv')
        self.data.dev_csv = dev_candidate
        self.data.test_csv = _resolve(self.data.test_csv)
        self.data.audio_base_dir = _resolve(self.data.audio_base_dir)
