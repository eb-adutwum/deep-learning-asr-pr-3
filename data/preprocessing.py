"""
Audio preprocessing utilities for ASR.
"""
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Tuple, Optional

class AudioPreprocessor:
    """Handles audio preprocessing for transformer models."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        normalize: bool = True
    ):
        """
        Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks
            f_min: Minimum frequency for mel scale
            f_max: Maximum frequency for mel scale
            normalize: Whether to normalize audio
        """
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )
        
        # Amplitude to DB transform (for Whisper-like features)
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
    def resample(self, waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
        """
        Resample audio to target sample rate.
        
        Args:
            waveform: Input audio waveform
            orig_sample_rate: Original sample rate
            
        Returns:
            Resampled waveform
        """
        if orig_sample_rate != self.sample_rate:
            resampler = T.Resample(orig_sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        return waveform
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except ImportError:
            waveform, sample_rate = self._load_with_soundfile(audio_path)
        except Exception as e:
            if 'TorchCodec' in str(e):
                waveform, sample_rate = self._load_with_soundfile(audio_path)
            else:
                raise ValueError(f"Error loading audio from {audio_path}: {e}")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform, sample_rate
    
    def _load_with_soundfile(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Fallback loader using soundfile to avoid torchcodec dependency."""
        import soundfile as sf
        
        data, sample_rate = sf.read(audio_path)
        waveform = torch.tensor(data, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
        return waveform, sample_rate
    
    def preprocess(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file for model input.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed features (mel spectrogram)
        """
        # Load audio
        waveform, sample_rate = self.load_audio(audio_path)
        
        # Resample
        waveform = self.resample(waveform, sample_rate)
        
        # Normalize
        if self.normalize:
            waveform = self.normalize_waveform(waveform)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        return mel_spec_db.squeeze(0)  # Remove channel dimension
    
    def preprocess_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess waveform tensor.
        
        Args:
            waveform: Input waveform
            sample_rate: Sample rate of waveform
            
        Returns:
            Preprocessed features
        """
        # Resample
        waveform = self.resample(waveform, sample_rate)
        
        # Normalize
        if self.normalize:
            waveform = self.normalize_waveform(waveform)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        return mel_spec_db.squeeze(0)

    def normalize_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize waveform to zero mean/unit variance.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            Normalized waveform
        """
        std = waveform.std()
        if std == 0:
            std = std + 1e-8
        return (waveform - waveform.mean()) / (std + 1e-8)


class TextPreprocessor:
    """Handles text preprocessing for ASR."""
    
    def __init__(self, tokenizer_type: str = 'char'):
        """
        Initialize text preprocessor.
        
        Args:
            tokenizer_type: 'char' for character-level or 'word' for word-level
        """
        self.tokenizer_type = tokenizer_type
        self.vocab = None
        self.vocab_size = None
    
    def build_vocab(self, texts: list) -> dict:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Vocabulary dictionary mapping tokens to indices
        """
        if self.tokenizer_type == 'char':
            # Collect all unique characters
            all_chars = set()
            for text in texts:
                all_chars.update(text.lower())
            
            # Build vocab with special tokens
            special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
            vocab = {token: idx for idx, token in enumerate(special_tokens)}
            
            # Add characters (sorted for consistency)
            for char in sorted(all_chars):
                if char not in vocab:
                    vocab[char] = len(vocab)
        
        else:  # word-level
            # Collect all words
            all_words = set()
            for text in texts:
                words = text.lower().split()
                all_words.update(words)
            
            # Build vocab with special tokens
            special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
            vocab = {token: idx for idx, token in enumerate(special_tokens)}
            
            # Add words (sorted for consistency)
            for word in sorted(all_words):
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        return vocab
    
    def text_to_ids(self, text: str, add_special_tokens: bool = True) -> list:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        text = text.lower()
        ids = []
        
        if add_special_tokens:
            ids.append(self.vocab['<sos>'])
        
        if self.tokenizer_type == 'char':
            for char in text:
                ids.append(self.vocab.get(char, self.vocab['<unk>']))
        else:
            for word in text.split():
                ids.append(self.vocab.get(word, self.vocab['<unk>']))
        
        if add_special_tokens:
            ids.append(self.vocab['<eos>'])
        
        return ids
    
    def ids_to_text(self, ids: list) -> str:
        """
        Convert token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Reconstructed text
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        # Reverse vocab for lookup
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for id in ids:
            token = id_to_token.get(id, '<unk>')
            # Skip special tokens
            if token not in ['<pad>', '<sos>', '<eos>', '<unk>']:
                tokens.append(token)
        
        if self.tokenizer_type == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)

