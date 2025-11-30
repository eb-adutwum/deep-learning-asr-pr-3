"""
Evaluation script for pretrained transformer ASR model.
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from utils.config import Config
from data.dataset import AfriSpeechDataset, collate_fn
from data.preprocessing import AudioPreprocessor, TextPreprocessor
from models.pretrained_asr import PretrainedASRModel
from training.evaluator import Evaluator
from training.metrics import calculate_cer, calculate_wer
from utils.visualization import plot_examples

def build_vocab(csv_path: str) -> list:
    """Build vocabulary from transcripts."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    texts = df['transcript'].astype(str).tolist()
    return texts

def main():
    """Main evaluation function."""
    # Load configuration
    config = Config()
    
    # Checkpoint path
    checkpoint_path = os.path.join(config.training.output_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(config.training.output_dir, 'final_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using main_pretrained.py")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Build vocabulary
    print("Building vocabulary...")
    train_texts = build_vocab(config.data.train_csv)
    
    # Initialize preprocessors
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config.data.sample_rate,
        n_fft=config.data.n_fft,
        hop_length=config.data.hop_length,
        n_mels=config.data.n_mels,
        f_min=config.data.f_min,
        f_max=config.data.f_max
    )
    
    text_preprocessor = TextPreprocessor(tokenizer_type=config.data.tokenizer_type)
    text_preprocessor.build_vocab(train_texts)
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = AfriSpeechDataset(
        csv_path=config.data.test_csv,
        audio_base_dir=config.data.audio_base_dir,
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        use_augmentation=False,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print(f"Initializing model: {config.model.model_name}...")
    model = PretrainedASRModel(
        model_name=config.model.model_name,
        freeze_feature_extractor=config.model.freeze_feature_extractor,
        freeze_encoder=config.model.freeze_encoder,
        vocab_size=config.model.vocab_size if 'wav2vec2' in config.model.model_name.lower() else None
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    print("Checkpoint loaded successfully")
    
    # Create decode function
    def decode_fn(predictions):
        """Decode predictions to text."""
        if hasattr(model, 'processor'):
            try:
                texts = model.processor.batch_decode(
                    predictions,
                    skip_special_tokens=True
                )
                return texts
            except:
                pass
        # Fallback: use text preprocessor
        texts = []
        for pred in predictions:
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().tolist()
            text = text_preprocessor.ids_to_text(pred)
            texts.append(text)
        return texts
    
    # Evaluate
    print("\nEvaluating on test set...")
    evaluator = Evaluator(model, device=config.device)
    test_metrics = evaluator.evaluate(test_loader, decode_fn=decode_fn)
    
    print(f"\n{'='*50}")
    print(f"Test Set Results:")
    print(f"{'='*50}")
    print(f"Character Error Rate (CER): {test_metrics['cer']:.4f} ({test_metrics['cer']*100:.2f}%)")
    print(f"Word Error Rate (WER): {test_metrics['wer']:.4f} ({test_metrics['wer']*100:.2f}%)")
    print(f"{'='*50}\n")
    
    # Generate and save examples
    print("Generating example predictions...")
    examples = evaluator.generate_examples(test_loader, num_examples=20, decode_fn=decode_fn)
    
    # Print examples
    print("\nExample Predictions:")
    print("-" * 80)
    for i, ex in enumerate(examples[:10]):
        print(f"\nExample {i+1}:")
        print(f"  Reference:  {ex['reference']}")
        print(f"  Prediction: {ex['prediction']}")
        # Calculate edit distance for this example
        ref = ex['reference'].lower().strip()
        pred = ex['prediction'].lower().strip()
        errors = sum(a != b for a, b in zip(ref, pred))
        max_len = max(len(ref), len(pred))
        cer_example = errors / max_len if max_len > 0 else 1.0
        print(f"  CER: {cer_example:.4f}")
    
    # Plot examples
    output_dir = config.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_examples(
        examples,
        save_path=os.path.join(output_dir, 'test_examples.png'),
        max_examples=20
    )
    
    print(f"\nExamples saved to {os.path.join(output_dir, 'test_examples.png')}")
    print("\nEvaluation completed!")

if __name__ == '__main__':
    main()

