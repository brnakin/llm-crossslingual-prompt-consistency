"""
Cosine similarity computation and aggregation for cross-lingual consistency.

This module computes pairwise cosine similarities between embeddings
and aggregates them by language pair and task type.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from .load_prompts import load_prompts, is_open_text_task, get_task_types
from .infer_ollama import load_responses
from .embed_labse import get_or_compute_embeddings, get_embedding_for_response, load_labse_model


# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_METRICS_FILE = DEFAULT_DATA_DIR / "metrics.csv"
DEFAULT_STABILITY_FILE = DEFAULT_DATA_DIR / "stability.csv"


# Language pairs for cross-lingual comparison
LANGUAGE_PAIRS = [("EN", "DE"), ("EN", "TR"), ("DE", "TR")]


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding vector
        emb2: Second embedding vector
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    # Reshape for sklearn
    e1 = emb1.reshape(1, -1)
    e2 = emb2.reshape(1, -1)
    return float(sklearn_cosine(e1, e2)[0, 0])


def compute_cross_lingual_similarity(
    responses: List[Dict[str, Any]],
    embeddings: Dict[str, np.ndarray],
    model_id: str,
    prompt_id: int,
    run_id: int
) -> Dict[str, float]:
    """Compute cross-lingual similarities for a prompt and run.
    
    Args:
        responses: List of response records
        embeddings: Dictionary mapping keys to embeddings
        model_id: Model to filter by
        prompt_id: Prompt to analyze
        run_id: Run number to compare
        
    Returns:
        Dictionary: {"EN-DE": sim, "EN-TR": sim, "DE-TR": sim}
    """
    result = {}
    
    for lang1, lang2 in LANGUAGE_PAIRS:
        emb1 = get_embedding_for_response(model_id, prompt_id, lang1, run_id, embeddings)
        emb2 = get_embedding_for_response(model_id, prompt_id, lang2, run_id, embeddings)
        
        if emb1 is not None and emb2 is not None:
            result[f"{lang1}-{lang2}"] = cosine_similarity(emb1, emb2)
    
    return result


def compute_intra_language_stability(
    responses: List[Dict[str, Any]],
    embeddings: Dict[str, np.ndarray],
    model_id: str,
    prompt_id: int,
    language: str
) -> Optional[float]:
    """Compute intra-language stability (run1 vs run2) for open-text tasks.
    
    Args:
        responses: List of response records
        embeddings: Dictionary mapping keys to embeddings
        model_id: Model to filter by
        prompt_id: Prompt to analyze
        language: Language code
        
    Returns:
        Cosine similarity between run1 and run2, or None if not available
    """
    emb1 = get_embedding_for_response(model_id, prompt_id, language, 1, embeddings)
    emb2 = get_embedding_for_response(model_id, prompt_id, language, 2, embeddings)
    
    if emb1 is not None and emb2 is not None:
        return cosine_similarity(emb1, emb2)
    
    return None


def compute_all_metrics(
    responses_file: Optional[Path] = None,
    metrics_file: Optional[Path] = None,
    stability_file: Optional[Path] = None,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute all metrics for open-text tasks.
    
    Args:
        responses_file: Path to responses.jsonl
        metrics_file: Path to save metrics.csv
        stability_file: Path to save stability.csv
        show_progress: Show progress
        
    Returns:
        Tuple of (metrics_df, stability_df)
    """
    if metrics_file is None:
        metrics_file = DEFAULT_METRICS_FILE
    if stability_file is None:
        stability_file = DEFAULT_STABILITY_FILE
    
    # Load responses
    responses = load_responses(responses_file)
    
    if not responses:
        print("No responses found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter to open-text tasks only
    open_text_responses = [r for r in responses if is_open_text_task(r["task_type"])]
    
    if not open_text_responses:
        print("No open-text responses found!")
        return pd.DataFrame(), pd.DataFrame()
    
    if show_progress:
        print(f"Processing {len(open_text_responses)} open-text responses...")
    
    # Load model and compute embeddings
    model = load_labse_model()
    embeddings = get_or_compute_embeddings(open_text_responses, model=model, show_progress=show_progress)
    
    # Get unique combinations
    models = sorted(set(r["model_id"] for r in open_text_responses))
    prompts_df = load_prompts(prepend_control_line=False)
    open_text_prompts = prompts_df[prompts_df["task_type"].apply(is_open_text_task)]
    prompt_ids = sorted(open_text_prompts["prompt_id"].unique())
    
    # Compute cross-lingual metrics
    metrics_rows = []
    for model_id in models:
        for prompt_id in prompt_ids:
            task_type = open_text_prompts[open_text_prompts["prompt_id"] == prompt_id]["task_type"].iloc[0]
            
            for run_id in [1, 2]:
                sims = compute_cross_lingual_similarity(
                    responses, embeddings, model_id, prompt_id, run_id
                )
                
                for pair, sim in sims.items():
                    metrics_rows.append({
                        "model_id": model_id,
                        "prompt_id": prompt_id,
                        "task_type": task_type,
                        "pair": pair,
                        "run_id": run_id,
                        "cosine_similarity": sim,
                        "flag_low_similarity": False  # Will be updated below
                    })
    
    metrics_df = pd.DataFrame(metrics_rows)
    
    # Flag bottom 10% per model
    if len(metrics_df) > 0:
        for model_id in models:
            mask = metrics_df["model_id"] == model_id
            threshold = metrics_df.loc[mask, "cosine_similarity"].quantile(0.10)
            metrics_df.loc[mask & (metrics_df["cosine_similarity"] <= threshold), "flag_low_similarity"] = True
    
    # Compute intra-language stability
    stability_rows = []
    for model_id in models:
        for prompt_id in prompt_ids:
            task_type = open_text_prompts[open_text_prompts["prompt_id"] == prompt_id]["task_type"].iloc[0]
            
            for language in ["EN", "DE", "TR"]:
                stability = compute_intra_language_stability(
                    responses, embeddings, model_id, prompt_id, language
                )
                
                if stability is not None:
                    stability_rows.append({
                        "model_id": model_id,
                        "prompt_id": prompt_id,
                        "task_type": task_type,
                        "language": language,
                        "stability_type": "open_text_cosine",
                        "run_id_a": 1,
                        "run_id_b": 2,
                        "stability_value": stability
                    })
    
    stability_df = pd.DataFrame(stability_rows)
    
    # Save files
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_file, index=False)
    stability_df.to_csv(stability_file, index=False)
    
    if show_progress:
        print(f"Saved metrics to {metrics_file}")
        print(f"Saved stability to {stability_file}")
    
    return metrics_df, stability_df


def load_metrics(metrics_file: Optional[Path] = None) -> pd.DataFrame:
    """Load metrics from CSV.
    
    Args:
        metrics_file: Path to metrics.csv
        
    Returns:
        DataFrame with metrics
    """
    if metrics_file is None:
        metrics_file = DEFAULT_METRICS_FILE
    
    if not metrics_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(metrics_file)


def load_stability(stability_file: Optional[Path] = None) -> pd.DataFrame:
    """Load stability metrics from CSV.
    
    Args:
        stability_file: Path to stability.csv
        
    Returns:
        DataFrame with stability metrics
    """
    if stability_file is None:
        stability_file = DEFAULT_STABILITY_FILE
    
    if not stability_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(stability_file)


def aggregate_by_language_pair(
    metrics_df: pd.DataFrame,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate metrics by language pair.
    
    Args:
        metrics_df: Metrics DataFrame
        model_id: Optional model filter
        
    Returns:
        DataFrame with mean, std, count by pair
    """
    df = metrics_df.copy()
    if model_id:
        df = df[df["model_id"] == model_id]
    
    return df.groupby("pair").agg({
        "cosine_similarity": ["mean", "std", "count"]
    }).round(4)


def aggregate_by_task_type(
    metrics_df: pd.DataFrame,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate metrics by task type.
    
    Args:
        metrics_df: Metrics DataFrame
        model_id: Optional model filter
        
    Returns:
        DataFrame with mean, std, count by task type
    """
    df = metrics_df.copy()
    if model_id:
        df = df[df["model_id"] == model_id]
    
    return df.groupby("task_type").agg({
        "cosine_similarity": ["mean", "std", "count"]
    }).round(4)


def aggregate_by_pair_and_task(
    metrics_df: pd.DataFrame,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate metrics by language pair and task type.
    
    Args:
        metrics_df: Metrics DataFrame
        model_id: Optional model filter
        
    Returns:
        DataFrame with mean, std by pair and task type
    """
    df = metrics_df.copy()
    if model_id:
        df = df[df["model_id"] == model_id]
    
    return df.groupby(["pair", "task_type"]).agg({
        "cosine_similarity": ["mean", "std", "count"]
    }).round(4)


def get_flagged_examples(
    metrics_df: pd.DataFrame,
    responses: Optional[List[Dict[str, Any]]] = None,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Get flagged low-similarity examples with response texts.
    
    Args:
        metrics_df: Metrics DataFrame
        responses: Optional list of response records
        model_id: Optional model filter
        
    Returns:
        DataFrame with flagged examples
    """
    df = metrics_df[metrics_df["flag_low_similarity"]].copy()
    
    if model_id:
        df = df[df["model_id"] == model_id]
    
    if responses:
        # Add response texts
        resp_dict = {}
        for r in responses:
            key = (r["model_id"], r["prompt_id"], r["language"], r["run_id"])
            resp_dict[key] = r["response_text"]
        
        def get_responses(row):
            texts = {}
            pair = row["pair"]
            lang1, lang2 = pair.split("-")
            for lang in [lang1, lang2]:
                key = (row["model_id"], row["prompt_id"], lang, row["run_id"])
                texts[f"response_{lang}"] = resp_dict.get(key, "")
            return pd.Series(texts)
        
        response_cols = df.apply(get_responses, axis=1)
        df = pd.concat([df, response_cols], axis=1)
    
    return df.sort_values("cosine_similarity")


def compute_stability_summary(
    stability_df: pd.DataFrame,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Compute stability summary by language.
    
    Args:
        stability_df: Stability DataFrame
        model_id: Optional model filter
        
    Returns:
        DataFrame with mean, std by language
    """
    df = stability_df.copy()
    if model_id:
        df = df[df["model_id"] == model_id]
    
    return df.groupby("language").agg({
        "stability_value": ["mean", "std", "count"]
    }).round(4)


if __name__ == "__main__":
    # Test with sample data
    print("Testing similarity computation...")
    
    # Create sample embeddings
    np.random.seed(42)
    emb1 = np.random.randn(768)
    emb2 = np.random.randn(768)
    
    # Normalize
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    sim = cosine_similarity(emb1, emb2)
    print(f"Cosine similarity between random embeddings: {sim:.4f}")
    
    # Test with identical embeddings
    sim_same = cosine_similarity(emb1, emb1)
    print(f"Cosine similarity with itself: {sim_same:.4f}")
