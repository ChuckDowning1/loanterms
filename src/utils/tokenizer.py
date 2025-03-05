"""
Utility functions for tokenizing text and handling token-based operations.
"""
import tiktoken
import pandas as pd
from typing import List, Tuple, Dict, Any

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text using the cl100k_base encoding.
    
    Args:
        text: The text to tokenize.
        
    Returns:
        The number of tokens in the text.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    return len(tokens)

def tokenize_text(text: str) -> List[int]:
    """
    Tokenize the given text using the cl100k_base encoding.
    
    Args:
        text: The text to tokenize.
        
    Returns:
        A list of token IDs.
    """
    encoder = tiktoken.get_encoding("cl100k_base")
    return encoder.encode(text)

def chunk_text(text: str, chunk_size: int, overlap_size: int) -> List[str]:
    """
    Split text into overlapping chunks based on token count.
    
    Args:
        text: The text to chunk.
        chunk_size: The maximum number of tokens per chunk.
        overlap_size: The number of tokens to overlap between chunks.
        
    Returns:
        A list of text chunks.
    """
    if not text or text.isspace():
        return []
        
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(tokens):
        # Get chunk tokens
        end_idx = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end_idx]
        
        # Decode to text
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move to next position, accounting for overlap
        i += chunk_size - overlap_size
        if i >= len(tokens):
            break
    
    return chunks

def calculate_token_statistics(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
    """
    Calculate token statistics for a text column in a DataFrame.
    
    Args:
        df: DataFrame containing the text data.
        text_column: The name of the column containing text.
        
    Returns:
        Dictionary with token statistics.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Remove empty texts
    df_filtered = df[df[text_column].notna() & (df[text_column] != "")]
    
    # Count tokens for each text
    token_counts = df_filtered[text_column].apply(count_tokens)
    
    return {
        "total_documents": len(df_filtered),
        "total_tokens": token_counts.sum(),
        "average_tokens_per_document": token_counts.mean(),
        "median_tokens": token_counts.median(),
        "min_tokens": token_counts.min(),
        "max_tokens": token_counts.max(),
    } 