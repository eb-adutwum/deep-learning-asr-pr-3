"""
Main script for training pretrained transformer ASR model.
Initial version - before any tweaks.
"""
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from utils.config import Config
from data.dataset import AfriSpeechDataset, collate_fn
from data.preprocessing import AudioPreprocessor, TextPreprocessor
from models.pretrained_asr import PretrainedASRModel
from training.trainer import Trainer
from training.evaluator import Evaluator

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_vocab(csv_path: str) -> list:
    """Build vocabulary from transcripts."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    texts = df['transcript'].astype(str).tolist()
    return texts


class TeeLogger:
    """Tee stdout to both console and log file."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'w', buffering=1)
        self._stdout = sys.stdout
    
    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        self.log_file.close()
    
    def write(self, data):
        self._stdout.write(data)
        self.log_file.write(data)
    
    def flush(self):
        self._stdout.flush()
        self.log_file.flush()

def main():
    """Main entry point."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = Config()
    set_seed(config.seed)
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    log_path = os.path.join(config.training.output_dir, 'training1.log')
    
    with TeeLogger(log_path):
        _run_training(config)

def _run_training(config: Config):
    """Main training routine."""
    print(f"Using device: {config.device}")
    
    try:
        import transformers
        print("transformers library is available")
    except ImportError:
        print("ERROR: transformers library not available. Please install it:")
        print("pip install transformers")
        return
    
    # Initialize preprocessors
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        n_mels=config.data.n_mels,
        f_min=config.data.f_min,
        f_max=config.data.f_max
    )
    
    # Initialize model
    print(f"Initializing model: {config.model.model_name}...")
    model = PretrainedASRModel(
        model_name=config.model.model_name,
        freeze_feature_extractor=config.model.freeze_feature_extractor,
        freeze_encoder=config.model.freeze_encoder,
        vocab_size=config.model.vocab_size
    )
    processor = getattr(model, 'processor', None)
    
    pad_token_id = 0
    text_preprocessor = None
    if processor and hasattr(processor, 'tokenizer'):
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        pad_token_id = processor.tokenizer.pad_token_id
    else:
        print("Building custom vocabulary...")
        train_texts = build_vocab(config.data.train_csv)
        text_preprocessor = TextPreprocessor(tokenizer_type=config.data.tokenizer_type)
        text_preprocessor.build_vocab(train_texts)
        pad_token_id = text_preprocessor.vocab.get('<pad>', 0)
        print(f"Vocabulary size: {text_preprocessor.vocab_size}")
    
    # Create datasets
    print("Loading datasets...")
    dataset_kwargs = {
        'audio_preprocessor': audio_preprocessor,
        'text_preprocessor': text_preprocessor,
        'processor': processor,
        'model_type': model.model_type,
        'max_audio_length': config.data.max_audio_length,
        'min_audio_length': config.data.min_audio_length
    }
    
    train_dataset = AfriSpeechDataset(
        csv_path=config.data.train_csv,
        audio_base_dir=config.data.audio_base_dir,
        use_augmentation=config.data.use_augmentation,
        split='train',
        **dataset_kwargs
    )
    
    val_dataset = AfriSpeechDataset(
        csv_path=config.data.dev_csv,
        audio_base_dir=config.data.audio_base_dir,
        use_augmentation=False,
        split='dev',
        **dataset_kwargs
    )
    
    test_dataset = AfriSpeechDataset(
        csv_path=config.data.test_csv,
        audio_base_dir=config.data.audio_base_dir,
        use_augmentation=False,
        split='test',
        **dataset_kwargs
    )
    
    from functools import partial
    collate = partial(collate_fn, pad_token_id=pad_token_id if pad_token_id is not None else 0, label_pad_token_id=-100)
    
    pin_memory = config.device == 'cuda'
    num_workers = min(4, os.cpu_count() or 2)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print(f"Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.98)
    )
    
    # Initialize scheduler
    total_steps = len(train_loader) * config.training.num_epochs
    warmup_steps = config.training.warmup_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_eval_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.device,
        output_dir=config.training.output_dir,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        max_grad_norm=config.training.max_grad_norm,
        fp16=config.training.fp16,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        label_smoothing=config.training.label_smoothing
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
        train_loss = trainer.train_epoch()
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate()
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val CER: {val_metrics['cer']:.4f}")
        print(f"Val WER: {val_metrics['wer']:.4f}")
        print(f"Val BLEU: {val_metrics['bleu']:.4f}")
        
        # Store metrics for visualization
        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_metrics['loss'])
        trainer.val_cers.append(val_metrics['cer'])
        trainer.val_wers.append(val_metrics['wer'])
        trainer.val_bleus.append(val_metrics['bleu'])
    
    # Save final checkpoint
    trainer.save_checkpoint('final_model.pt')
    print("\nFinal model saved.")
    
    # Create visualization directory
    viz_dir = os.path.join(os.path.dirname(__file__), 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    from utils.visualization import plot_training_history, save_metric_curves
    
    history_path = os.path.join(viz_dir, 'training_history.png')
    plot_training_history(
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses,
        train_eval_losses=None,
        train_cers=None,
        val_cers=trainer.val_cers,
        train_wers=None,
        val_wers=trainer.val_wers,
        train_bleus=None,
        val_bleus=trainer.val_bleus,
        save_path=history_path
    )
    
    metrics_dict = {
        'loss': (trainer.train_losses, trainer.val_losses),
        'cer': (None, trainer.val_cers),
        'wer': (None, trainer.val_wers),
        'bleu': (None, trainer.val_bleus)
    }
    save_metric_curves(metrics_dict, viz_dir, 'training1')
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    evaluator = Evaluator(trainer.model, device=config.device)
    test_metrics = evaluator.evaluate(test_loader, decode_fn=None)
    
    print(f"\nTest Results:")
    print(f"Test CER: {test_metrics['cer']:.4f}")
    print(f"Test WER: {test_metrics['wer']:.4f}")
    print(f"Test BLEU: {test_metrics['bleu']:.4f}")
    
    # Show sample predictions vs ground truth
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS vs GROUND TRUTH (First 20 examples)")
    print("="*80)
    predictions = test_metrics.get('predictions', [])
    references = test_metrics.get('references', [])
    
    for i, (pred, ref) in enumerate(zip(predictions[:20], references[:20])):
        print(f"\nExample {i+1}:")
        print(f"GROUND TRUTH: {ref}")
        print(f"PREDICTION:   {pred}")
        print("-" * 80)
    
    print(f"\nVisualizations saved to: {viz_dir}")
    print(f"Training log saved to: {os.path.join(config.training.output_dir, 'training1.log')}")

if __name__ == '__main__':
    main()
