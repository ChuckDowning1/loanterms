"""
Module for interacting with the Google Gemini API to extract unique terms from loan agreements.
"""
import json
import os
from google import genai
from google.genai import types
from tqdm import tqdm
import time
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config.settings import GOOGLE_API_KEY, MODEL_NAME, SYSTEM_PROMPT
from src.utils.tokenizer import chunk_text

# Configure the Google Gemini API client
client = genai.Client(api_key=GOOGLE_API_KEY)

def extract_terms_from_chunk(text_chunk: str) -> List[Dict[str, Any]]:
    """
    Extract unique terms from a text chunk using the Gemini model.
    
    Args:
        text_chunk: A chunk of text to process.
        
    Returns:
        List of dictionaries containing terms and their credit risk relevance.
    """
    if not text_chunk or text_chunk.isspace():
        return []
    
    try:
        # Create prompt with instructions
        prompt = f"""You are a financial expert specializing in loan agreement analysis. Your goal is to create a comprehensive dictionary of conceptual terms commonly found in loan contracts.

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


Here is the text from the loan agreement:
{text_chunk}

4. Provide your dictionary in this structured JSON format:
{{
  "unique_terms": [
    {{
      "term": "string - the conceptual term type (generalized)",
      "credit_risk_relevant": true/false
    }},
    ...
  ]
}}
"""
        
        # Create content structure for the API request
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        # Configure generation parameters
        generate_config = types.GenerateContentConfig(
            temperature=0.0,  # Set to 0 for deterministic output
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )
        
        # Call the API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_config,
        )
        
        # Parse JSON from response
        try:
            result = json.loads(response.text)
            if "unique_terms" in result:
                return result["unique_terms"]
            else:
                print("Warning: Response missing 'unique_terms' key:", result)
                return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON from response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error in extract_terms_from_chunk: {str(e)}")
        # Sleep to avoid rate limiting
        time.sleep(2)
        return []

def deduplicate_terms_at_document_level(terms: List[Dict[str, Any]], loan_id: str) -> List[Dict[str, Any]]:
    """
    Second round of LLM processing: De-duplicate terms at the document level.
    
    Args:
        terms: List of terms extracted from a single document.
        loan_id: The ID of the loan document.
        
    Returns:
        De-duplicated list of terms.
    """
    if not terms:
        return []
    
    try:
        # Create prompt with instructions
        prompt = f"""You are a financial expert specializing in loan agreement analysis. Your task is to AGGRESSIVELY review a list of terms extracted from a single loan document (ID: {loan_id}) and create a consolidated, de-duplicated list of unique conceptual terms.

Be extremely thorough in identifying conceptually similar terms, even if they appear different on the surface. Your goal is significant reduction through intelligent consolidation while preserving truly distinct terms.

INSTRUCTIONS:
1. Aggressively consolidate terms that represent the same or similar concepts, even if worded differently
2. Look for conceptual similarities beyond surface-level wording differences
3. Remove all duplicate entries and near-duplicates
4. Generalize terms to be broadly applicable across loan agreements
5. When consolidating terms with different credit risk assessments, always preserve "True" over "False"
6. IMPORTANT: Do preserve genuinely different financial concepts. For example:
   - "EBITDA Covenant" and "Leverage Ratio Covenant" are different types of financial covenants
   - "Cross-Default Provision" and "Payment Default Provision" represent different default triggers
   - "Senior Debt" and "Subordinated Debt" are distinct debt classes

EXAMPLES OF TERMS THAT SHOULD BE CONSOLIDATED:
- "Prepayment Penalty" and "Prepayment Fee" (same concept, different wording)
- "Cross-Default Provision" and "Cross-Default Clause" (same concept, different wording)
- "Events of Default Definition" and "Default Events" (same concept, different wording)
- "Security Interest" and "Collateral Security Rights" (conceptually overlapping)
- "Financial Covenant Compliance" and "Financial Ratio Requirements" (conceptually similar)

Remember: Be aggressive in deduplication of similar concepts, but preserve truly distinct financial and legal terms.

Here is the list of extracted terms:
{json.dumps(terms, indent=2)}

Provide your consolidated list in this structured JSON format:
{{
  "unique_terms": [
    {{
      "term": "string - the conceptual term type (generalized)",
      "credit_risk_relevant": true/false
    }},
    ...
  ]
}}
"""
        
        # Create content structure for the API request
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        # Configure generation parameters
        generate_config = types.GenerateContentConfig(
            temperature=0.0,  # Set to 0 for deterministic output
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )
        
        # Call the API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_config,
        )
        
        # Parse JSON from response
        try:
            result = json.loads(response.text)
            if "unique_terms" in result:
                deduplicated_terms = result["unique_terms"]
                print(f"  Document-level deduplication: {len(terms)} → {len(deduplicated_terms)} terms")
                return deduplicated_terms
            else:
                print("Warning: Response missing 'unique_terms' key:", result)
                return terms  # Return original terms if failed
        except json.JSONDecodeError:
            print(f"Error decoding JSON from document-level deduplication: {response.text}")
            return terms  # Return original terms if failed
            
    except Exception as e:
        print(f"Error in deduplicate_terms_at_document_level: {str(e)}")
        # Sleep to avoid rate limiting
        time.sleep(2)
        return terms  # Return original terms if failed

def deduplicate_terms_at_corpus_level(existing_corpus: List[Dict[str, Any]], new_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Third round of LLM processing: De-duplicate terms at the corpus level.
    Uses the standard Gemini model for consistency with other functions.
    
    Args:
        existing_corpus: The current corpus of unique terms.
        new_terms: New terms to evaluate against the existing corpus.
        
    Returns:
        Updated corpus with new unique terms added.
    """
    if not new_terms:
        return existing_corpus
    
    if not existing_corpus:
        return new_terms
    
    try:
        # Create prompt with instructions
        prompt = f"""You are a financial expert specializing in loan agreement analysis. Your task is to AGGRESSIVELY update an existing corpus of unique loan agreement terms with new terms, avoiding any duplication or conceptual overlap.

Be extremely thorough in identifying conceptually similar terms between the existing corpus and new terms. Your goal is to maintain a truly unique corpus of conceptual terms without redundancy while preserving genuinely distinct concepts.

INSTRUCTIONS:
1. Review the existing corpus of terms thoroughly
2. For each new term, rigorously evaluate it against ALL existing corpus terms:
   - If it represents a concept already in the corpus (even partially), DO NOT add it
   - If it represents a genuinely new concept, add it to the corpus
3. Apply a high standard for what constitutes a "new concept"
4. If a term exists in both lists with different credit risk assessments, always preserve the version marked as "true" for credit_risk_relevant
5. Ensure all terms remain generalized and broadly applicable
6. IMPORTANT: Do preserve truly different financial concepts. For example:
   - Different types of covenants (e.g., "Debt-to-EBITDA Covenant" vs "Interest Coverage Covenant")
   - Different security mechanisms (e.g., "First Lien Security" vs "Second Lien Security")
   - Different default triggers (e.g., "Payment Default" vs "Covenant Default")

EXAMPLES OF CONCEPTUAL OVERLAP TO WATCH FOR:
- "Debt Service Coverage Ratio" and "DSCR Requirements" (same concept, different wording)
- "Default Interest Rate" and "Interest Rate During Default" (same concept, different wording)
- "Material Adverse Change Clause" and "MAC Provision" (same concept, different wording)
- "Security Documentation" and "Collateral Documentation" (conceptually overlapping)
- "Financial Statement Delivery Requirements" and "Financial Reporting Obligations" (conceptually similar)

Remember: Be aggressive in deduplication of similar concepts, but preserve genuinely distinct financial and legal terms.

Here is the existing corpus of terms:
{json.dumps(existing_corpus, indent=2)}

Here are the new terms to evaluate:
{json.dumps(new_terms, indent=2)}

Provide your updated corpus in this structured JSON format:
{{
  "unique_terms": [
    {{
      "term": "string - the conceptual term type (generalized)",
      "credit_risk_relevant": true/false
    }},
    ...
  ]
}}
"""
        
        # Create content structure for the API request
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        # Configure generation parameters
        generate_config = types.GenerateContentConfig(
            temperature=0.0,  # Set to 0 for deterministic output
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )
        
        # Try with primary method first
        try:
            # Call the API with the standard model
            response = client.models.generate_content(
                model=MODEL_NAME,  # Use the standard model defined in settings
                contents=contents,
                config=generate_config,
            )
            
            # Parse JSON from response
            result = json.loads(response.text)
            if "unique_terms" in result:
                updated_corpus = result["unique_terms"]
                print(f"  Corpus-level deduplication: {len(existing_corpus)} + {len(new_terms)} → {len(updated_corpus)} terms")
                return updated_corpus
            else:
                print("Warning: Response missing 'unique_terms' key in corpus deduplication")
                raise ValueError("Invalid response format")
                
        except Exception as e:
            print(f"First attempt at corpus-level deduplication failed: {str(e)}. Trying fallback method...")
            
            # Try an alternative approach - breaking the corpus into smaller batches
            if len(existing_corpus) > 50 and len(new_terms) > 10:
                print("  Using batch processing for large corpus deduplication")
                return deduplicate_terms_at_corpus_level_batch(existing_corpus, new_terms)
            else:
                # Fall back to manual combination if both attempts fail
                return manual_combine_terms(existing_corpus, new_terms)
            
    except Exception as e:
        print(f"Error in deduplicate_terms_at_corpus_level: {str(e)}")
        # Sleep to avoid rate limiting
        time.sleep(2)
        # Fallback to manual combination if LLM fails
        return manual_combine_terms(existing_corpus, new_terms)

def deduplicate_terms_at_corpus_level_batch(existing_corpus: List[Dict[str, Any]], new_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback method for corpus-level deduplication that processes new terms in smaller batches.
    
    Args:
        existing_corpus: The existing corpus of terms.
        new_terms: New terms to add to the corpus.
        
    Returns:
        Updated corpus with new unique terms added.
    """
    # Create a copy of the existing corpus
    updated_corpus = existing_corpus.copy()
    
    # Process new terms in smaller batches (10 terms at a time)
    batch_size = 10
    for i in range(0, len(new_terms), batch_size):
        batch = new_terms[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(new_terms) + batch_size - 1)//batch_size}")
        
        try:
            # Create prompt with instructions
            prompt = f"""You are a financial expert specializing in loan agreement analysis. Your task is to evaluate a small batch of new terms against an existing corpus and identify ONLY terms that represent truly new concepts.

INSTRUCTIONS:
1. Review the existing corpus conceptually
2. For each new term in the batch, determine if it represents a concept NOT found in the corpus
3. Return ONLY the terms that represent new concepts
4. Be selective - only include genuinely new financial or legal concepts
5. IMPORTANT: Do preserve truly different financial concepts. For example:
   - Different types of covenants (e.g., "Debt-to-EBITDA Covenant" vs "Interest Coverage Covenant")
   - Different security mechanisms (e.g., "First Lien Security" vs "Second Lien Security") 
   - Different default triggers (e.g., "Payment Default" vs "Covenant Default")

Here is the existing corpus (conceptual reference only):
{json.dumps([term.get("term", "") for term in updated_corpus], indent=2)}

Here are the new terms to evaluate:
{json.dumps(batch, indent=2)}

Provide ONLY the truly new terms in this JSON format:
{{
  "new_unique_terms": [
    {{
      "term": "string - the conceptual term type",
      "credit_risk_relevant": true/false
    }},
    ...
  ]
}}
"""
            
            # Create content structure for the API request
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            # Configure generation parameters
            generate_config = types.GenerateContentConfig(
                temperature=0.0,  # Set to 0 for deterministic output
                top_p=0.95,
                top_k=40,
                max_output_tokens=4096,
                response_mime_type="application/json",
            )
            
            # Call the API
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=generate_config,
            )
            
            # Parse JSON from response
            result = json.loads(response.text)
            if "new_unique_terms" in result and isinstance(result["new_unique_terms"], list):
                new_unique_terms = result["new_unique_terms"]
                
                # Manual deduplication as a safety measure
                for term in new_unique_terms:
                    term_text = term.get("term", "").strip().lower()
                    if not term_text:
                        continue
                        
                    # Check if term already exists in updated corpus
                    exists = False
                    for existing_term in updated_corpus:
                        if existing_term.get("term", "").strip().lower() == term_text:
                            exists = True
                            # Update credit risk relevance if needed
                            if term.get("credit_risk_relevant") and not existing_term.get("credit_risk_relevant"):
                                existing_term["credit_risk_relevant"] = True
                            break
                    
                    if not exists:
                        updated_corpus.append(term)
                
                print(f"    Added {len(new_unique_terms)} new terms from batch")
            else:
                print("    No new unique terms found in batch")
                
        except Exception as e:
            print(f"  Error processing batch: {str(e)}")
            # Add all terms from the batch using manual deduplication
            updated_corpus = manual_combine_terms(updated_corpus, batch)
        
        # Sleep briefly to avoid rate limiting
        time.sleep(1)
    
    return updated_corpus

def manual_combine_terms(existing_corpus: List[Dict[str, Any]], new_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Fallback function to manually combine terms if the LLM-based approach fails.
    
    Args:
        existing_corpus: The existing corpus of terms.
        new_terms: New terms to add to the corpus.
        
    Returns:
        Combined list of terms with duplicates removed.
    """
    print("  Using manual deduplication as fallback")
    
    # Create a dictionary of existing terms (lowercase for case-insensitive comparison)
    term_dict = {term.get("term", "").lower(): term for term in existing_corpus}
    
    # Add new terms, avoiding duplicates
    for term_info in new_terms:
        term_text = term_info.get("term", "").strip()
        if not term_text:
            continue
            
        term_lower = term_text.lower()
        
        # If term already exists, only replace if new one is credit risk relevant
        if term_lower in term_dict:
            if term_info.get("credit_risk_relevant") and not term_dict[term_lower].get("credit_risk_relevant"):
                term_dict[term_lower] = term_info
        else:
            term_dict[term_lower] = term_info
    
    # Convert back to list
    return list(term_dict.values())

def process_loan_text(text: str, chunk_size: int, overlap_size: int, loan_id: str = "unknown") -> List[Dict[str, Any]]:
    """
    Process a full loan agreement text by chunking it and extracting terms from each chunk.
    
    Args:
        text: The full text of the loan agreement.
        chunk_size: Maximum number of tokens per chunk.
        overlap_size: Number of tokens to overlap between chunks.
        loan_id: ID of the loan being processed.
        
    Returns:
        List of consolidated unique terms and their credit risk relevance.
    """
    if not text or text.isspace():
        return []
    
    chunks = chunk_text(text, chunk_size, overlap_size)
    
    all_terms = []
    for chunk in chunks:
        terms = extract_terms_from_chunk(chunk)
        all_terms.extend(terms)
    
    # First-level deduplication (case-insensitive)
    term_dict = {}
    for term_info in all_terms:
        term_text = term_info.get("term", "").strip()
        if not term_text:
            continue
            
        term_lower = term_text.lower()
        
        # If term already exists, only replace if new one is credit risk relevant
        if term_lower in term_dict:
            if term_info.get("credit_risk_relevant") and not term_dict[term_lower].get("credit_risk_relevant"):
                term_dict[term_lower] = term_info
        else:
            term_dict[term_lower] = term_info
    
    chunk_deduplicated_terms = list(term_dict.values())
    
    # Second-level deduplication: Use LLM to deduplicate at document level
    document_deduplicated_terms = deduplicate_terms_at_document_level(
        chunk_deduplicated_terms, loan_id
    )
    
    return document_deduplicated_terms

def process_all_loans(df: pd.DataFrame, chunk_size: int, overlap_size: int) -> List[Dict[str, Any]]:
    """
    Process all loans in the DataFrame to extract unique terms.
    
    Args:
        df: DataFrame containing loan data with text column.
        chunk_size: Maximum number of tokens per chunk.
        overlap_size: Number of tokens to overlap between chunks.
        
    Returns:
        List of all unique terms across all loans.
    """
    # Skip rows with empty text
    df_filtered = df[df['text'].notna() & (df['text'] != "")]
    
    # Initialize corpus of terms
    corpus_terms = []
    document_term_counts = []
    
    # Process each loan
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing loans"):
        try:
            # Process loan text to extract terms (includes chunk-level and document-level deduplication)
            loan_terms = process_loan_text(
                row['text'], 
                chunk_size, 
                overlap_size, 
                loan_id=row.get('accession', f"loan_{idx}")
            )
            
            document_term_counts.append(len(loan_terms))
            print(f"Extracted {len(loan_terms)} terms from loan {row.get('accession', f'loan_{idx}')}")
            
            # Third-level deduplication: Use LLM to update corpus with new terms
            corpus_terms = deduplicate_terms_at_corpus_level(corpus_terms, loan_terms)
            print(f"Corpus now contains {len(corpus_terms)} unique terms after processing loan {row.get('accession', f'loan_{idx}')}")
            
            # Sleep briefly to avoid API rate limits
            time.sleep(1)
        except Exception as e:
            print(f"Error processing loan {row.get('accession', f'loan_{idx}')}: {str(e)}")
    
    print(f"Total terms extracted from all documents: {sum(document_term_counts)}")
    print(f"Final corpus contains {len(corpus_terms)} unique terms after all deduplication steps")
    
    return corpus_terms

def standardize_terms(all_terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Standardize and deduplicate terms across all processed loans.
    
    Args:
        all_terms: List of term dictionaries from all loans.
        
    Returns:
        List of standardized, unique terms.
    """
    # Sort by credit risk relevance, then alphabetically by term
    unique_terms = sorted(
        all_terms, 
        key=lambda x: (not x.get("credit_risk_relevant", False), x.get("term", "").lower())
    )
    
    return unique_terms

def save_terms_to_csv(terms: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save terms to a CSV file.
    
    Args:
        terms: List of term dictionaries.
        output_path: Path to save the CSV file.
    """
    df = pd.DataFrame(terms)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(terms)} unique terms to {output_path}") 