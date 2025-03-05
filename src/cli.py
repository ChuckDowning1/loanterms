"""
Command-line interface for the loan term extraction project.

This script provides a convenient CLI to run the term extraction process
with optional parameters for customization.
"""
import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path to help with imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import main
from src.config import settings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract unique terms from loan agreements using Google Gemini API."
    )
    
    parser.add_argument(
        "--input", 
        type=str,
        help="Path to input CSV file (default: data/sample_loans.csv)",
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Directory to save output files (default: src/output)",
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=int,
        help=f"Token chunk size for processing (default: {settings.CHUNK_SIZE})",
    )
    
    parser.add_argument(
        "--chunk-overlap", 
        type=int,
        help=f"Overlap between chunks (default: {settings.CHUNK_OVERLAP})",
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        help=f"Gemini model to use (default: {settings.MODEL_NAME})",
    )
    
    parser.add_argument(
        "--api-key", 
        type=str,
        help="Google API key (overrides .env file)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int,
        help="Limit number of loans to process (useful for testing)"
    )
    
    return parser.parse_args()

def update_settings(args):
    """Update settings based on command line arguments."""
    if args.input:
        settings.LOANS_CSV = Path(args.input)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        settings.OUTPUT_DIR = output_dir
        settings.DESCRIPTIVE_STATS_FILE = output_dir / "descriptive_stats.json"
        settings.TOKEN_STATS_FILE = output_dir / "token_stats.json"
        settings.UNIQUE_TERMS_FILE = output_dir / "unique_terms.csv"
    
    if args.chunk_size:
        settings.CHUNK_SIZE = args.chunk_size
    
    if args.chunk_overlap:
        settings.CHUNK_OVERLAP = args.chunk_overlap
    
    if args.model:
        settings.MODEL_NAME = args.model
    
    if args.api_key:
        os.environ["GOOGLE_API_KEY"] = args.api_key
        settings.GOOGLE_API_KEY = args.api_key

if __name__ == "__main__":
    args = parse_args()
    update_settings(args)
    main(limit=args.limit if args.limit else None) 