#!/usr/bin/env python3
"""
Script to pre-download BERT model for offline use.
Run this once when you have good internet connection.
"""

import os
from transformers import AutoTokenizer, AutoModel

# Model to download (can be changed via env var)
MODEL_NAME = os.getenv("FACTCHECKER_BERT_MODEL", "distilbert-base-multilingual-cased")

def download_model():
    """Download BERT model and save to cache."""
    print(f"Downloading BERT model: {MODEL_NAME}")
    print("This may take a while depending on your internet speed...")
    print("Model will be cached for future use.\n")
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✓ Tokenizer downloaded")
        
        print("Downloading model...")
        model = AutoModel.from_pretrained(MODEL_NAME)
        print("✓ Model downloaded")
        
        print(f"\n✓ Successfully downloaded {MODEL_NAME}")
        print("Model is now cached and ready to use.")
        print(f"Cache location: ~/.cache/huggingface/hub/")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("You can still use the system with bi-encoder fallback.")
        return False
    
    return True

if __name__ == "__main__":
    download_model()

