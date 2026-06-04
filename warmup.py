#!/usr/bin/env python3
"""
Warmup script: pre-loads all models and datasets during deployment.
Run this before starting Streamlit to eliminate cold start delays.
"""
import sys
import os

# Suppress transformers warnings during warmup
import warnings
warnings.filterwarnings('ignore')

print("🔥 Warming up Papermind backend...")

try:
    from backend import load_dataset
    
    print("  → Loading arXiv dataset...")
    df, tfidf_vec, tfidf_matrix, terms, classifier, embed_model, retriever = load_dataset()
    
    print(f"  ✓ Loaded {len(df)} papers")
    print(f"  ✓ TF-IDF vectorizer ready ({len(tfidf_vec.get_feature_names_out())} features)")
    print(f"  ✓ Classifier trained on {len(classifier.named_steps['model'].classes_)} categories")
    print(f"  ✓ Embedding model ready (all-MiniLM-L6-v2)")
    print(f"  ✓ Retriever fitted with {len(df)} embeddings")
    
    print("\n✅ Warmup complete! Streamlit is ready to serve requests instantly.")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Warmup failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)

