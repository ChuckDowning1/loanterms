"""
Configuration settings for the loan terms extraction project.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "src" / "output"

# Input file
LOANS_CSV = DATA_DIR / "sample_loans.csv"

# Output files
DESCRIPTIVE_STATS_FILE = OUTPUT_DIR / "descriptive_stats.json"
TOKEN_STATS_FILE = OUTPUT_DIR / "token_stats.json"
UNIQUE_TERMS_FILE = OUTPUT_DIR / "unique_terms.csv"

# Google Gemini API settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash"
CHUNK_SIZE = 10000  # Tokens per chunk for processing
CHUNK_OVERLAP = 2000  # Overlap between chunks

# System prompt for the model
SYSTEM_PROMPT = """
You are a financial expert specializing in loan agreement analysis. Your goal is to create a comprehensive dictionary of conceptual terms commonly found in loan contracts.

1. Review the attached loan document and identify conceptual terms that represent important financial and legal provisions, focusing on term categories rather than specific wording. Consider:
   - Payment structure concepts (e.g., "interest rate calculation method", "prepayment penalty structure")
   - Collateral and security concepts (e.g., "security interest provisions", "collateral valuation mechanisms")
   - Default and remedy frameworks (e.g., "cross-default triggers", "cure period provisions")
   - Covenant restriction types (e.g., "debt-to-EBITDA restriction", "dividend limitation provision")
   - Special provision categories (e.g., "material adverse change clause", "force majeure definition")

2. For each conceptual term in your dictionary:
   - Provide a brief description of what this type of term governs
   - Determine if this concept is relevant to credit risk assessment

3. Ensure terms are generalized and conceptual rather than specific to the particular document reviewed

4. Provide your dictionary in this structured JSON format:
{
  "unique_terms": [
    {
      "term": "string - the conceptual term type",
      "credit_risk_relevant": true/false
    },
    ...
  ]
}
""" 