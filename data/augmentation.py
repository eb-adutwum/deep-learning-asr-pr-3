"""
Data augmentation utilities for audio.
"""
import torch
import numpy as np
from typing import Tuple

class SpeedPerturbation:
    """Apply speed perturbation to audio."""
    
    def __init__(self, min_speed: float = 0.9, max_speed: float = 1.1):
        """
        Initialize speed perturbation.
        
        Args:
            min_speed: Minimum speed factor
            max_speed: Maximum speed factor
        """
        self.min_speed = min_speed
        self.max_speed = max_speed
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply speed perturbation.
        
        Args:
            waveform: Input waveform
            sample_rate: Sample rate
            
        Returns:
            Speed-perturbed waveform
        """
        speed = np.random.uniform(self.min_speed, self.max_speed)
        # Resample to simulate speed change
        new_length = int(len(waveform) / speed)
        indices = torch.linspace(0, len(waveform) - 1, new_length)
        perturbed = waveform[indices.long()]
        return perturbed


class SpecAugment:
    """Apply SpecAugment to mel spectrograms."""
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        """
        Initialize SpecAugment.
        
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to mel spectrogram.
        
        Args:
            mel_spec: Input mel spectrogram [n_mels, time]
            
        Returns:
            Augmented mel spectrogram
        """
        augmented = mel_spec.clone()
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, augmented.shape[0] - f)
            augmented[f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, augmented.shape[1] - t)
            augmented[:, t0:t0+t] = 0
        
        return augmented


class TimeMask:
    """Apply time masking to mel spectrograms."""
    
    def __init__(self, mask_param: int = 50):
        """
        Initialize time mask.
        
        Args:
            mask_param: Maximum width of time mask
        """
        self.mask_param = mask_param
    
    def __call__(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply time masking.
        
        Args:
            mel_spec: Input mel spectrogram [n_mels, time]
            
        Returns:
            Masked mel spectrogram
        """
        augmented = mel_spec.clone()
        t = np.random.randint(0, min(self.mask_param, augmented.shape[1]))
        t0 = np.random.randint(0, augmented.shape[1] - t)
        augmented[:, t0:t0+t] = 0
        return augmented


class FreqMask:
    """Apply frequency masking to mel spectrograms."""
    
    def __init__(self, mask_param: int = 27):
        """
        Initialize frequency mask.
        
        Args:
            mask_param: Maximum width of frequency mask
        """
        self.mask_param = mask_param
    
    def __call__(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking.
        
        Args:
            mel_spec: Input mel spectrogram [n_mels, time]
            
        Returns:
            Masked mel spectrogram
        """
        augmented = mel_spec.clone()
        f = np.random.randint(0, min(self.mask_param, augmented.shape[0]))
        f0 = np.random.randint(0, augmented.shape[0] - f)
        augmented[f0:f0+f, :] = 0
        return augmented

