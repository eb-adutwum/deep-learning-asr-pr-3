"""
Training utilities for pretrained transformer ASR.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
from .metrics import calculate_cer, calculate_wer, calculate_bleu

class Trainer:
    """Trainer class for pretrained ASR model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        train_eval_loader: DataLoader = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = 'checkpoints',
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = False,
        gradient_accumulation_steps: int = 1,
        label_smoothing: float = 0.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            output_dir: Output directory for checkpoints
            save_steps: Steps between saves
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            max_grad_norm: Maximum gradient norm for clipping
            fp16: Whether to use mixed precision
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_eval_loader = train_eval_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_smoothing = label_smoothing
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize scaler for mixed precision
        if fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_cer = float('inf')
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.val_cers = []
        self.val_wers = []
        self.val_bleus = []
        self.train_eval_losses = []
        self.train_cers = []
        self.train_wers = []
        self.train_bleus = []
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare inputs
            inputs, labels = self._prepare_batch(batch)
            
            # Forward pass
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    # Apply label smoothing manually if needed
                    if self.label_smoothing > 0 and hasattr(outputs, 'logits'):
                        # Label smoothing is handled by the model's loss function
                        # For Whisper, we can't easily modify it, so we'll rely on model config
                        pass
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                if self.label_smoothing > 0 and hasattr(outputs, 'logits'):
                    # Label smoothing handled by model config
                    pass
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Logging
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            if self.global_step % self.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': loss.item() * self.gradient_accumulation_steps,
                    'lr': current_lr,
                    'step': self.global_step
                })
            
            # Evaluation
            if self.global_step % self.eval_steps == 0 and self.global_step > 0:
                val_metrics = self.evaluate()
                # Save if best
                if val_metrics['cer'] < self.best_val_cer:
                    self.best_val_cer = val_metrics['cer']
                    self.save_checkpoint('best_model.pt')
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0 and self.global_step > 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate_loader(
        self,
        data_loader: DataLoader,
        desc: str = "Evaluating",
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """Evaluate on specified loader."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                inputs, labels = self._prepare_batch(batch)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate with proper parameters to prevent over-generation
                predictions = self.model.generate(
                    **inputs,
                    max_length=256  # Limit generation length (other params set in model.generate)
                )
                predictions_text = self._decode_batch(predictions, batch)
                references_text = batch['transcripts']
                
                all_predictions.extend(predictions_text)
                all_references.extend(references_text)
        
        avg_loss = total_loss / max(len(data_loader), 1)
        cer = calculate_cer(all_predictions, all_references)
        wer = calculate_wer(all_predictions, all_references)
        bleu = calculate_bleu(all_predictions, all_references)
        
        results = {
            'loss': avg_loss,
            'cer': cer,
            'wer': wer,
            'bleu': bleu
        }
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['references'] = all_references
        
        return results
    
    def evaluate(self, return_predictions: bool = False) -> Dict[str, float]:
        """Evaluate on validation loader."""
        return self.evaluate_loader(
            self.val_loader,
            desc="Evaluating (val)",
            return_predictions=return_predictions
        )
    
    def _prepare_batch(self, batch: Dict) -> tuple:
        """
        Prepare batch for model input.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            Tuple of (inputs dict, labels tensor)
        """
        audio = batch['audio'].to(self.device, dtype=torch.float32)
        labels = batch.get('labels')
        if labels is None:
            text_ids = batch['text_ids']
            labels = text_ids.clone()
        labels = labels.to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Determine model type and prepare inputs
        if hasattr(self.model, 'model_type'):
            if self.model.model_type == 'whisper':
                # Whisper expects input_features
                inputs = {
                    'input_features': audio,
                    'attention_mask': attention_mask
                }
            else:  # wav2vec2
                # Wav2Vec2 expects input_values (waveform)
                # For now, pass mel spectrogram (will need to convert if using Wav2Vec2)
                inputs = {
                    'input_values': audio,
                    'attention_mask': attention_mask
                }
        else:
            # Default to Whisper format
            inputs = {'input_features': audio, 'attention_mask': attention_mask}
        
        return inputs, labels
    
    def _decode_batch(self, predictions: torch.Tensor, batch: Dict) -> list:
        """
        Decode predictions to text.
        
        Args:
            predictions: Predicted token IDs
            batch: Original batch
            
        Returns:
            List of decoded texts
        """
        # This should use the processor/tokenizer
        # For now, return placeholder - will be implemented based on actual model
        if hasattr(self.model, 'processor'):
            texts = self.model.processor.batch_decode(
                predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            # Filter out blank or invalid predictions
            cleaned_texts = []
            for text in texts:
                # Remove excessive dashes and whitespace
                text = text.strip()
                # If text is mostly dashes or very short, try to recover
                if len(text) < 3 or text.count('-') > len(text) * 0.5:
                    text = ""  # Mark as empty, will be handled by metrics
                # Remove non-printable characters except spaces
                text = ''.join(c for c in text if c.isprintable() or c.isspace())
                cleaned_texts.append(text)
            return cleaned_texts
        else:
            # Fallback: use text preprocessor from dataset
            # This is a placeholder - actual implementation depends on model
            return [f"prediction_{i}" for i in range(len(batch['transcripts']))]
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'best_val_cer': self.best_val_cer,
            'train_losses': self.train_losses,
             'train_eval_losses': self.train_eval_losses,
             'train_cers': self.train_cers,
             'train_wers': self.train_wers,
             'train_bleus': self.train_bleus,
            'val_losses': self.val_losses,
            'val_cers': self.val_cers,
            'val_wers': self.val_wers,
            'val_bleus': self.val_bleus
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_cer = checkpoint.get('best_val_cer', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_eval_losses = checkpoint.get('train_eval_losses', [])
        self.train_cers = checkpoint.get('train_cers', [])
        self.train_wers = checkpoint.get('train_wers', [])
        self.train_bleus = checkpoint.get('train_bleus', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_cers = checkpoint.get('val_cers', [])
        self.val_wers = checkpoint.get('val_wers', [])
        self.val_bleus = checkpoint.get('val_bleus', [])
        print(f"Checkpoint loaded from {checkpoint_path}")

