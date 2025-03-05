"""
Main script to extract unique terms from loan agreements using Google Gemini API.

This script:
1. Loads loan data from CSV
2. Generates descriptive statistics
3. Tokenizes text and calculates token statistics
4. Processes loan texts through Gemini API to extract terms
5. Standardizes and deduplicates terms
6. Saves the final list of unique terms to CSV
"""
import os
import json
import pandas as pd
from pathlib import Path
import time
from typing import Optional

from src.config.settings import (
    LOANS_CSV, DESCRIPTIVE_STATS_FILE, TOKEN_STATS_FILE, UNIQUE_TERMS_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP, OUTPUT_DIR
)
from src.utils.data_utils import (
    load_loan_data, generate_descriptive_stats, save_stats_to_json, plot_loans_over_time
)
from src.utils.tokenizer import calculate_token_statistics
from src.processors.gemini_processor import process_all_loans, standardize_terms, save_terms_to_csv

def main(limit: Optional[int] = None):
    """
    Main function to run the term extraction process.
    
    Args:
        limit: Optional limit on the number of loans to process (for testing).
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Step 1: Loading loan data...")
    df = load_loan_data(LOANS_CSV)
    
    # Shuffle the dataframe to randomize processing order
    df = df.sample(frac=1.0, random_state=None).reset_index(drop=True)
    print(f"Shuffled {len(df)} loans to randomize processing order")
    
    # Apply limit if specified
    if limit and limit > 0:
        print(f"Limiting processing to {limit} randomly selected loans (out of {len(df)} total)")
        df = df.head(limit)
    
    print("\nStep 2: Generating descriptive statistics...")
    descriptive_stats = generate_descriptive_stats(df)
    save_stats_to_json(descriptive_stats, DESCRIPTIVE_STATS_FILE)
    
    # Create visualization of loans over time
    print("\nStep 3: Creating visualizations...")
    plot_loans_over_time(df, OUTPUT_DIR)
    
    print("\nStep 4: Calculating token statistics...")
    token_stats = calculate_token_statistics(df, 'text')
    save_stats_to_json(token_stats, TOKEN_STATS_FILE)
    
    print(f"\nTotal documents: {token_stats['total_documents']}")
    print(f"Total tokens: {token_stats['total_tokens']:,}")
    print(f"Average tokens per document: {token_stats['average_tokens_per_document']:.2f}")
    
    print("\nStep 5: Processing loans through Gemini API...")
    all_terms = process_all_loans(df, CHUNK_SIZE, CHUNK_OVERLAP)
    
    print("\nStep 6: Standardizing and deduplicating terms...")
    unique_terms = standardize_terms(all_terms)
    
    print("\nStep 7: Saving final unique terms to CSV...")
    save_terms_to_csv(unique_terms, UNIQUE_TERMS_FILE)
    
    # Print summary of credit risk relevant terms
    credit_risk_terms = [t for t in unique_terms if t.get("credit_risk_relevant", False)]
    print(f"\nIdentified {len(unique_terms)} unique terms in total")
    print(f"Of these, {len(credit_risk_terms)} were identified as relevant to credit risk")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main() 