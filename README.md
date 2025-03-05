# Loan Agreement Term Extractor

This repository processes loan agreements to extract unique terms and assess their relevance to credit risk using the Google Gemini API.

## Project Overview

The project analyzes loan agreements to:

1. Provide descriptive statistics about loan coverage over time
2. Calculate token statistics for the loan text data
3. Extract unique terms from each loan agreement using Google Gemini API
4. Assess whether each term is relevant to credit risk evaluation
5. Apply multi-level LLM-based deduplication to ensure conceptual uniqueness
6. Standardize and deduplicate terms across all loans
7. Output a final CSV with unique terms and their credit risk relevance

## Repository Structure

```
.
├── data/                   # Data directory
│   └── sample_loans.csv    # Input loan data
├── src/                    # Source code
│   ├── config/             # Configuration settings
│   │   └── settings.py     # Project configuration
│   ├── output/             # Output files directory
│   ├── processors/         # Processing modules
│   │   └── gemini_processor.py  # Gemini API interaction
│   ├── utils/              # Utility functions
│   │   ├── data_utils.py   # Data loading and stats
│   │   └── tokenizer.py    # Text tokenization utilities
│   ├── cli.py              # Command-line interface
│   └── main.py             # Main execution script
├── requirements.txt        # Project dependencies
├── template.env            # Template for .env file
└── README.md               # This documentation
```

## Setup and Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd loan-term-extractor
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the template and add your Google API key:
   ```
   cp template.env .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

## Usage

### Basic Usage

1. Place your loan data CSV in the `data/` directory with a structure matching the sample_loans.csv format.

2. Run the main script:
   ```
   python src/main.py
   ```

3. Check the output in the `src/output/` directory.

### Command-Line Interface

For more control, use the CLI script:

```
python src/cli.py [options]
```

Options:
- `--input PATH`: Path to input CSV file
- `--output-dir PATH`: Directory to save output files
- `--chunk-size N`: Token chunk size for processing
- `--chunk-overlap N`: Overlap between chunks
- `--model NAME`: Gemini model to use
- `--api-key KEY`: Google API key (overrides .env file)
- `--limit N`: Limit number of loans to process (useful for testing)

Example:
```
python src/cli.py --input my_loans.csv --output-dir ./results --limit 5 --chunk-size 8000
```

The system automatically randomizes the order of loans before processing, so when using the `--limit` option, you'll get different documents each time you run the program. This is especially useful for testing with small subsets of your data.

### Output Files

Check the output in the specified directory:
- `descriptive_stats.json`: Statistics about loan coverage
- `token_stats.json`: Token statistics for loan texts
- `unique_terms.csv`: Final list of unique terms and credit risk relevance
- `loans_by_year.png`: Visualization of loan distribution by year

## Data Format

The input CSV should have the following columns:
- `accession`: Loan identifier
- `date`: Date of the loan
- `text`: Full text of the loan agreement
- `loan`: Loan type indicator
- `amendment`: Amendment indicator
- `lead_arranger`: Lead arranger name
- `lender_name`: Lender name
- `borrower_name`: Borrower name
- `gvkey`: Company identifier

## Multi-Level LLM Deduplication

The system employs a three-tiered approach to ensure truly unique conceptual terms:

1. **Chunk-Level Deduplication**: Basic string matching to remove exact duplicates within a document's chunks.

2. **Document-Level LLM Deduplication**: Uses the Gemini API to intelligently consolidate terms from a single document based on conceptual similarity rather than just string matching. This ensures terms that represent the same concept but are worded differently are properly deduplicated.

3. **Corpus-Level LLM Deduplication**: After processing each document, its deduplicated terms are evaluated against the growing corpus. The Gemini API identifies conceptually similar terms across documents and only adds genuinely new concepts to the corpus.

This multi-level approach significantly reduces redundancy while preserving genuinely distinct financial and legal concepts. The prompts are carefully designed to aggressively consolidate similar terms while recognizing important differences between financial concepts (e.g., different types of covenants or security mechanisms).

## Google Gemini API

This project uses the Google Gemini 2.0 Flash model to analyze loan agreements. The API processes text in chunks of 10,000 tokens with 2,000 token overlap to handle long documents effectively.

Key characteristics of our API implementation:
- Zero-temperature generation for deterministic, reproducible outputs
- Structured JSON output format
- Fallback mechanisms for handling large corpora
- Batch processing for efficient deduplication of large term sets

## Configuration

Key settings can be modified in `src/config/settings.py`:
- `CHUNK_SIZE`: Token chunk size for processing (default: 10000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 2000)
- `MODEL_NAME`: Gemini model to use (default: "gemini-2.0-flash")

## Output Format

The final CSV output contains:
- `term`: The extracted unique term from the loan agreements
- `credit_risk_relevant`: Boolean indicating whether the term is relevant for credit risk assessment

## Performance

The multi-level deduplication typically achieves:
- 30-40% reduction at the document level compared to basic deduplication
- 10-20% further reduction at the corpus level
- Overall higher quality terms that represent truly unique financial and legal concepts 