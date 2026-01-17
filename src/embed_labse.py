"""
LaBSE embedding generation for cross-lingual semantic similarity.

This module handles loading the LaBSE model and generating embeddings
for response texts. Embeddings are cached for reuse.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import json

from .load_prompts import load_config


# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_EMBEDDINGS_CACHE = DEFAULT_DATA_DIR / "embeddings_cache.json"


# Global model cache
_model_cache = {}


def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration.
    
    Returns:
        Dictionary with model_name, embedding_dim, normalize
    """
    config = load_config("embeddings.yaml")
    return config.get("embedding", {
        "model_name": "sentence-transformers/LaBSE",
        "embedding_dim": 768,
        "normalize": True
    })


def load_labse_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Load the LaBSE model (cached after first load).
    
    Args:
        model_name: Model identifier. Defaults to config value.
        
    Returns:
        Loaded SentenceTransformer model
    """
    global _model_cache
    
    if model_name is None:
        config = get_embedding_config()
        model_name = config["model_name"]
    
    if model_name not in _model_cache:
        print(f"Loading LaBSE model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    
    return _model_cache[model_name]


def embed_text(text: str, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """Generate embedding for a single text.
    
    Args:
        text: Text to embed
        model: Optional pre-loaded model
        
    Returns:
        Numpy array of shape (embedding_dim,)
    """
    if model is None:
        model = load_labse_model()
    
    config = get_embedding_config()
    normalize = config.get("normalize", True)
    
    embedding = model.encode(text, normalize_embeddings=normalize)
    return np.array(embedding)


def embed_texts(texts: List[str], model: Optional[SentenceTransformer] = None, 
                show_progress: bool = True) -> np.ndarray:
    """Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed
        model: Optional pre-loaded model
        show_progress: Show progress bar
        
    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    if model is None:
        model = load_labse_model()
    
    config = get_embedding_config()
    normalize = config.get("normalize", True)
    
    embeddings = model.encode(
        texts, 
        normalize_embeddings=normalize,
        show_progress_bar=show_progress
    )
    return np.array(embeddings)


def embed_responses(
    responses: List[Dict[str, Any]],
    model: Optional[SentenceTransformer] = None,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """Generate embeddings for response records.
    
    Args:
        responses: List of response records with 'response_text' field
        model: Optional pre-loaded model
        show_progress: Show progress bar
        
    Returns:
        Dictionary mapping response key to embedding.
        Key format: "{model_id}_{prompt_id}_{language}_{run_id}"
    """
    if model is None:
        model = load_labse_model()
    
    # Extract texts and keys
    texts = []
    keys = []
    for r in responses:
        key = f"{r['model_id']}_{r['prompt_id']}_{r['language']}_{r['run_id']}"
        keys.append(key)
        texts.append(r["response_text"])
    
    # Generate embeddings
    embeddings = embed_texts(texts, model, show_progress)
    
    # Create result dictionary
    result = {}
    for key, embedding in zip(keys, embeddings):
        result[key] = embedding
    
    return result


def load_embeddings_cache(cache_file: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load cached embeddings from file.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        Dictionary mapping keys to embeddings
    """
    if cache_file is None:
        cache_file = DEFAULT_EMBEDDINGS_CACHE
    
    if not cache_file.exists():
        return {}
    
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    return {k: np.array(v) for k, v in data.items()}


def save_embeddings_cache(
    embeddings: Dict[str, np.ndarray],
    cache_file: Optional[Path] = None
) -> None:
    """Save embeddings to cache file.
    
    Args:
        embeddings: Dictionary mapping keys to embeddings
        cache_file: Path to cache file
    """
    if cache_file is None:
        cache_file = DEFAULT_EMBEDDINGS_CACHE
    
    # Ensure parent directory exists
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    data = {k: v.tolist() for k, v in embeddings.items()}
    
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)


def get_or_compute_embeddings(
    responses: List[Dict[str, Any]],
    cache_file: Optional[Path] = None,
    model: Optional[SentenceTransformer] = None,
    show_progress: bool = True
) -> Dict[str, np.ndarray]:
    """Get embeddings from cache or compute if missing.
    
    Args:
        responses: List of response records
        cache_file: Path to cache file
        model: Optional pre-loaded model
        show_progress: Show progress bar
        
    Returns:
        Dictionary mapping response keys to embeddings
    """
    # Load existing cache
    cache = load_embeddings_cache(cache_file)
    
    # Find responses that need embedding
    to_embed = []
    for r in responses:
        key = f"{r['model_id']}_{r['prompt_id']}_{r['language']}_{r['run_id']}"
        if key not in cache:
            to_embed.append(r)
    
    if to_embed:
        if show_progress:
            print(f"Computing embeddings for {len(to_embed)} responses...")
        
        if model is None:
            model = load_labse_model()
        
        new_embeddings = embed_responses(to_embed, model, show_progress)
        cache.update(new_embeddings)
        
        # Save updated cache
        save_embeddings_cache(cache, cache_file)
    
    # Return only embeddings for requested responses
    result = {}
    for r in responses:
        key = f"{r['model_id']}_{r['prompt_id']}_{r['language']}_{r['run_id']}"
        if key in cache:
            result[key] = cache[key]
    
    return result


def get_embedding_for_response(
    model_id: str,
    prompt_id: int,
    language: str,
    run_id: int,
    embeddings: Dict[str, np.ndarray]
) -> Optional[np.ndarray]:
    """Get embedding for a specific response.
    
    Args:
        model_id: Model ID
        prompt_id: Prompt ID
        language: Language code
        run_id: Run number
        embeddings: Dictionary of embeddings
        
    Returns:
        Embedding array or None if not found
    """
    key = f"{model_id}_{prompt_id}_{language}_{run_id}"
    return embeddings.get(key)


if __name__ == "__main__":
    # Test embedding
    print("Testing LaBSE embedding...")
    
    test_texts = [
        "This is a test sentence in English.",
        "Dies ist ein Testsatz auf Deutsch.",
        "Bu Türkçe bir test cümlesidir."
    ]
    
    model = load_labse_model()
    embeddings = embed_texts(test_texts, model, show_progress=False)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Compute pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings)
    print(f"\nPairwise cosine similarities:")
    print(f"  EN-DE: {sims[0, 1]:.4f}")
    print(f"  EN-TR: {sims[0, 2]:.4f}")
    print(f"  DE-TR: {sims[1, 2]:.4f}")
