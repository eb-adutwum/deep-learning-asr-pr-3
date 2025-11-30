"""
Pretrained transformer models for ASR.
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
import warnings

try:
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        AutoFeatureExtractor,
        AutoProcessor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers library not available. Install it to use pretrained models.")


class PretrainedASRModel(nn.Module):
    """Wrapper for pretrained ASR models."""
    
    def __init__(
        self,
        model_name: str = 'openai/whisper-small',
        freeze_feature_extractor: bool = True,
        freeze_encoder: bool = False,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize pretrained ASR model.
        
        Args:
            model_name: Model name (e.g., 'openai/whisper-small', 'facebook/wav2vec2-base-960h')
            freeze_feature_extractor: Whether to freeze feature extractor
            freeze_encoder: Whether to freeze encoder layers
            vocab_size: Vocabulary size for CTC models (if needed)
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install it with: pip install transformers")
        
        self.model_name = model_name
        self.model_type = 'whisper' if 'whisper' in model_name.lower() else 'wav2vec2'
        
        # Load model and processor
        if self.model_type == 'whisper':
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            # Set label smoothing in model config if available
            if hasattr(self.model.config, 'label_smoothing'):
                self.model.config.label_smoothing = 0.1
            try:
                self.processor = WhisperProcessor.from_pretrained(model_name)
            except:
                self.processor = AutoProcessor.from_pretrained(model_name)
            
            if hasattr(self, 'processor') and self.processor is not None:
                tokenizer = getattr(self.processor, 'tokenizer', None)
                if tokenizer is not None and tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            
            # Disable forced language tokens for multilingual fine-tuning
            if hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'forced_decoder_ids'):
                    self.model.config.forced_decoder_ids = None
                if hasattr(self.model.config, 'suppress_tokens'):
                    self.model.config.suppress_tokens = []
            if hasattr(self.model, 'generation_config'):
                if hasattr(self.model.generation_config, 'forced_decoder_ids'):
                    self.model.generation_config.forced_decoder_ids = None
                if hasattr(self.model.generation_config, 'suppress_tokens'):
                    self.model.generation_config.suppress_tokens = []
            
            # Apply dropout to decoder for regularization
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                for layer in self.model.model.decoder.layers:
                    if hasattr(layer, 'dropout') and hasattr(layer.dropout, 'p'):
                        layer.dropout.p = 0.2
                    if hasattr(layer, 'activation_dropout') and hasattr(layer.activation_dropout, 'p'):
                        layer.activation_dropout.p = 0.2
                    # Also set dropout in attention and feedforward
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'dropout'):
                        if hasattr(layer.self_attn.dropout, 'p'):
                            layer.self_attn.dropout.p = 0.2
                    if hasattr(layer, 'encoder_attn') and hasattr(layer.encoder_attn, 'dropout'):
                        if hasattr(layer.encoder_attn.dropout, 'p'):
                            layer.encoder_attn.dropout.p = 0.2
            
            # Freeze layers if requested
            if freeze_feature_extractor:
                if hasattr(self.model, 'model'):
                    if hasattr(self.model.model, 'encoder'):
                        for param in self.model.model.encoder.embed_positions.parameters():
                            param.requires_grad = False
                    if hasattr(self.model.model, 'feature_extractor'):
                        for param in self.model.model.feature_extractor.parameters():
                            param.requires_grad = False
            
            if freeze_encoder:
                if hasattr(self.model, 'model'):
                    if hasattr(self.model.model, 'encoder'):
                        for param in self.model.model.encoder.parameters():
                            param.requires_grad = False
        
        else:  # wav2vec2
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            except:
                self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Resize token embeddings if vocab_size is provided
            if vocab_size and vocab_size != self.model.config.vocab_size:
                self.model.resize_token_embeddings(vocab_size)
            
            # Freeze layers if requested
            if freeze_feature_extractor:
                if hasattr(self.model, 'wav2vec2'):
                    for param in self.model.wav2vec2.feature_extractor.parameters():
                        param.requires_grad = False
            
            if freeze_encoder:
                if hasattr(self.model, 'wav2vec2'):
                    for param in self.model.wav2vec2.encoder.parameters():
                        param.requires_grad = False
    
    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_features: Input features (for Whisper) [batch, n_mels, time]
            input_values: Input values (for Wav2Vec2) [batch, time]
            labels: Labels for training [batch, seq_len]
            
        Returns:
            Model outputs
        """
        if self.model_type == 'whisper':
            if input_features is None:
                raise ValueError("input_features required for Whisper")
            
            # Convert mel spectrogram to expected format
            # Whisper expects [batch, n_mels, time]
            if input_features.dim() == 3:
                # Already in correct format
                pass
            else:
                raise ValueError(f"Unexpected input_features shape: {input_features.shape}")
            
            # Create attention mask if not provided
            if 'attention_mask' not in kwargs and labels is not None:
                # Create attention mask from labels (ignore padding tokens -100)
                attention_mask = (labels != -100).long()
                kwargs['decoder_attention_mask'] = attention_mask
            
            outputs = self.model(
                input_features=input_features,
                labels=labels,
                **kwargs
            )
        else:  # wav2vec2
            if input_values is None:
                raise ValueError("input_values required for Wav2Vec2")
            
            outputs = self.model(
                input_values=input_values,
                labels=labels,
                **kwargs
            )
        
        return outputs
    
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        input_values: Optional[torch.Tensor] = None,
        max_length: int = 256,  # Reduced to prevent over-generation
        **kwargs
    ) -> torch.Tensor:
        """
        Generate predictions.
        
        Args:
            input_features: Input features (for Whisper)
            input_values: Input values (for Wav2Vec2)
            max_length: Maximum generation length
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        if self.model_type == 'whisper':
            if input_features is None:
                raise ValueError("input_features required for Whisper")
            
            # Suppress deprecation warnings by using new cache format
            generation_config = self.model.generation_config
            if hasattr(generation_config, 'use_cache') and generation_config.use_cache:
                kwargs.setdefault('use_cache', True)
            
            return self.model.generate(
                input_features=input_features,
                max_length=max_length,
                num_beams=5,  # Beam search for better quality
                do_sample=False,
                length_penalty=1.0,  # Penalize longer sequences
                early_stopping=True,
                no_repeat_ngram_size=2,  # Prevent repetition
                **kwargs
            )
        else:  # wav2vec2
            if input_values is None:
                raise ValueError("input_values required for Wav2Vec2")
            
            # Wav2Vec2 uses CTC decoding, so we get logits and decode
            with torch.no_grad():
                outputs = self.model(input_values=input_values)
                logits = outputs.logits
                # CTC greedy decoding
                predicted_ids = torch.argmax(logits, dim=-1)
            return predicted_ids

