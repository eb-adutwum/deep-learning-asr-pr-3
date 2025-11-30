"""
Evaluation metrics for ASR.
"""
from jiwer import wer, cer as jiwer_cer
from sacrebleu.metrics import BLEU
from typing import List

bleu_scorer = BLEU(effective_order=True)

def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        WER score
    """
    return wer(references, predictions)

def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        CER score
    """
    return jiwer_cer(references, predictions)

def calculate_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Calculate corpus BLEU score (0-1 scale).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        BLEU score between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0
    score = bleu_scorer.corpus_score(predictions, [references]).score
    return score / 100.0

def edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    return dp[m][n]

def character_error_rate(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate using edit distance.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        CER score
    """
    total_chars = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        ref = ref.lower().strip()
        pred = pred.lower().strip()
        total_chars += len(ref) if len(ref) > 0 else 1
        total_errors += edit_distance(pred, ref)
    
    return total_errors / total_chars if total_chars > 0 else 1.0

def word_error_rate(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate using edit distance.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        WER score
    """
    total_words = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        ref_words = ref.lower().strip().split()
        pred_words = pred.lower().strip().split()
        total_words += len(ref_words) if len(ref_words) > 0 else 1
        # Calculate word-level edit distance
        total_errors += edit_distance(' '.join(ref_words), ' '.join(pred_words))
    
    return total_errors / total_words if total_words > 0 else 1.0

