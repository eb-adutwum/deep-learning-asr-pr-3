"""
Evaluation utilities for ASR models.
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from .metrics import calculate_cer, calculate_wer, calculate_bleu

class Evaluator:
    """Evaluator for ASR models."""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader, decode_fn=None) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            decode_fn: Optional function to decode predictions
            
        Returns:
            Dictionary with metrics and predictions
        """
        all_predictions = []
        all_references = []
        all_audio_paths = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Prepare inputs
                audio = batch['audio'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Determine model type
                if hasattr(self.model, 'model_type'):
                    if self.model.model_type == 'whisper':
                        inputs = {'input_features': audio, 'attention_mask': attention_mask}
                    else:
                        inputs = {'input_values': audio, 'attention_mask': attention_mask}
                else:
                    inputs = {'input_features': audio, 'attention_mask': attention_mask}
                
                # Generate predictions with proper parameters
                predictions = self.model.generate(
                    **inputs,
                    max_length=256  # Limit generation length (other params set in model.generate)
                )
                
                # Decode
                if decode_fn:
                    predictions_text = decode_fn(predictions)
                else:
                    predictions_text = self._decode_default(predictions, batch)
                
                references_text = batch['transcripts']
                
                all_predictions.extend(predictions_text)
                all_references.extend(references_text)
        
        # Calculate metrics
        cer = calculate_cer(all_predictions, all_references)
        wer = calculate_wer(all_predictions, all_references)
        bleu = calculate_bleu(all_predictions, all_references)
        
        return {
            'cer': cer,
            'wer': wer,
            'bleu': bleu,
            'predictions': all_predictions,
            'references': all_references
        }
    
    def _decode_default(self, predictions: torch.Tensor, batch: Dict) -> List[str]:
        """Default decoding method."""
        if hasattr(self.model, 'processor'):
            try:
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
                    # If text is mostly dashes or very short, mark as empty
                    if len(text) < 3 or text.count('-') > len(text) * 0.5:
                        text = ""  # Mark as empty, will be handled by metrics
                    # Remove non-printable characters except spaces
                    text = ''.join(c for c in text if c.isprintable() or c.isspace())
                    cleaned_texts.append(text)
                return cleaned_texts
            except:
                pass
        
        # Fallback
        return [f"pred_{i}" for i in range(len(batch['transcripts']))]
    
    def generate_examples(
        self,
        test_loader: DataLoader,
        num_examples: int = 10,
        decode_fn=None
    ) -> List[Dict]:
        """
        Generate example predictions.
        
        Args:
            test_loader: Test data loader
            num_examples: Number of examples to generate
            decode_fn: Optional decode function
            
        Returns:
            List of example dictionaries
        """
        examples = []
        count = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if count >= num_examples:
                    break
                
                # Prepare inputs
                audio = batch['audio'].to(self.device)
                
                if hasattr(self.model, 'model_type'):
                    if self.model.model_type == 'whisper':
                        inputs = {'input_features': audio}
                    else:
                        inputs = {'input_values': audio}
                else:
                    inputs = {'input_features': audio}
                
                # Generate with proper parameters
                predictions = self.model.generate(
                    **inputs,
                    max_length=256  # Limit generation length (other params set in model.generate)
                )
                
                # Decode
                if decode_fn:
                    predictions_text = decode_fn(predictions)
                else:
                    predictions_text = self._decode_default(predictions, batch)
                
                references_text = batch['transcripts']
                
                for pred, ref in zip(predictions_text, references_text):
                    if count >= num_examples:
                        break
                    examples.append({
                        'prediction': pred,
                        'reference': ref
                    })
                    count += 1
        
        return examples

