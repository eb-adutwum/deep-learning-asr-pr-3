"""
Visualization utilities for training and evaluation.
"""
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple

def _plot_metric(ax, title: str, train_values: Optional[List[float]], val_values: Optional[List[float]], ylabel: str):
    """Helper to plot train/val curves."""
    if train_values:
        ax.plot(train_values, label='Train')
    if val_values:
        ax.plot(val_values, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_eval_losses: Optional[List[float]] = None,
    train_cers: Optional[List[float]] = None,
    val_cers: Optional[List[float]] = None,
    train_wers: Optional[List[float]] = None,
    val_wers: Optional[List[float]] = None,
    train_bleus: Optional[List[float]] = None,
    val_bleus: Optional[List[float]] = None,
    save_path: str = 'training_history.png'
):
    """
    Plot training history with train vs validation curves.
    """
    metrics: List[Tuple[str, Optional[List[float]], Optional[List[float]], str]] = [
        ('Loss', train_eval_losses or train_losses, val_losses, 'Loss')
    ]
    
    if train_cers or val_cers:
        metrics.append(('CER', train_cers, val_cers, 'CER'))
    if train_wers or val_wers:
        metrics.append(('WER', train_wers, val_wers, 'WER'))
    if train_bleus or val_bleus:
        metrics.append(('BLEU', train_bleus, val_bleus, 'BLEU'))
    
    rows = math.ceil(len(metrics) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 4))
    axes = axes.flatten()
    
    for idx, (title, train_vals, val_vals, ylabel) in enumerate(metrics):
        _plot_metric(axes[idx], title, train_vals, val_vals, ylabel)
    
    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

def save_metric_curves(
    metrics: Dict[str, Tuple[Optional[List[float]], Optional[List[float]]]],
    save_dir: str,
    prefix: str
):
    """Save individual metric plots."""
    os.makedirs(save_dir, exist_ok=True)
    for metric_name, (train_vals, val_vals) in metrics.items():
        if not train_vals and not val_vals:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_metric(ax, f"{metric_name.upper()} vs Epoch", train_vals, val_vals, metric_name.upper())
        file_path = os.path.join(save_dir, f"{prefix}_{metric_name.lower()}.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{metric_name.upper()} curve saved to {file_path}")


def plot_attention_weights(
    attention_weights: np.ndarray,
    input_text: str = None,
    output_text: str = None,
    save_path: str = 'attention_weights.png'
):
    """
    Plot attention weights.
    
    Args:
        attention_weights: Attention weights [num_heads, seq_len, seq_len] or [seq_len, seq_len]
        input_text: Input text (optional)
        output_text: Output text (optional)
        save_path: Path to save figure
    """
    # Handle multi-head attention
    if attention_weights.ndim == 3:
        num_heads = attention_weights.shape[0]
        # Average across heads for visualization
        attention_weights = attention_weights.mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    ax.set_xlabel('Input Position')
    ax.set_ylabel('Output Position')
    ax.set_title('Attention Weights')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention weights saved to {save_path}")


def plot_examples(
    examples: List[Dict],
    save_path: str = 'examples.png',
    max_examples: int = 10
):
    """
    Plot example predictions.
    
    Args:
        examples: List of example dictionaries with 'prediction' and 'reference'
        save_path: Path to save figure
        max_examples: Maximum number of examples to plot
    """
    num_examples = min(len(examples), max_examples)
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(15, num_examples * 2))
    if num_examples == 1:
        axes = [axes]
    
    for i, example in enumerate(examples[:num_examples]):
        pred = example['prediction']
        ref = example['reference']
        
        axes[i].axis('off')
        axes[i].text(0.05, 0.7, f"Reference: {ref}", 
                    transform=axes[i].transAxes, fontsize=10, wrap=True)
        axes[i].text(0.05, 0.3, f"Prediction: {pred}", 
                    transform=axes[i].transAxes, fontsize=10, 
                    color='blue', wrap=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Examples saved to {save_path}")

