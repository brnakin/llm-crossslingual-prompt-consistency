"""
Task-aware agreement checks for discrete-answer tasks.

This module extracts labels/answers from classification, reasoning, and factual
responses and checks for cross-lingual and intra-language agreement.
"""

import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from .load_prompts import load_prompts, is_discrete_task
from .infer_ollama import load_responses


# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_TASK_METRICS_FILE = DEFAULT_DATA_DIR / "task_metrics.csv"
DEFAULT_STABILITY_FILE = DEFAULT_DATA_DIR / "stability.csv"


# Expected answers for factual prompts (used for validation, not matching)
FACTUAL_EXPECTED = {
    13: ["h2o", "h₂o"],  # Water formula
    14: ["1945"],  # WWII end year
    15: ["ottawa"],  # Canada capital
    16: ["george orwell", "orwell"],  # 1984 author
}


def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Raw text
        
    Returns:
        Normalized lowercase text with extra whitespace removed
    """
    return " ".join(text.lower().strip().split())


def extract_classification_label(
    response_text: str,
    prompt_id: int
) -> Optional[str]:
    """Extract classification label from response.
    
    Args:
        response_text: The model's response
        prompt_id: The prompt ID (5-8 are classification)
        
    Returns:
        Extracted label or None if not found
    """
    text = normalize_text(response_text)
    
    if prompt_id == 5:  # Sentiment: Positive/Negative
        if "positive" in text or "positiv" in text or "olumlu" in text:
            return "positive"
        elif "negative" in text or "negativ" in text or "olumsuz" in text:
            return "negative"
    
    elif prompt_id == 6:  # Agreement: Agree/Disagree
        if "disagree" in text:
            return "disagree"
        elif "agree" in text:
            return "agree"
    
    elif prompt_id == 7:  # Intent: Request/Complaint
        if "request" in text:
            return "request"
        elif "complaint" in text:
            return "complaint"
    
    elif prompt_id == 8:  # Formality: Formal/Informal
        if "informal" in text:
            return "informal"
        elif "formal" in text:
            return "formal"
    
    return None


def extract_reasoning_answer(
    response_text: str,
    prompt_id: int
) -> Optional[str]:
    """Extract reasoning answer from response.
    
    Args:
        response_text: The model's response
        prompt_id: The prompt ID (9-12 are reasoning)
        
    Returns:
        Extracted answer or None if not found
    """
    text = normalize_text(response_text)
    
    if prompt_id == 9:  # Logic: A, B, or C (who is tallest)
        match = re.search(r'\b([abc])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    elif prompt_id == 10:  # Math: speed in km/h
        # Look for number followed by optional km/h
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:km/h|km\/h)?', text)
        if match:
            return match.group(1)
    
    elif prompt_id == 11:  # Time: minutes
        # Look for number (should be 90)
        match = re.search(r'(\d+)', text)
        if match:
            return match.group(1)
    
    elif prompt_id == 12:  # Decision: A, B, or C (weather clothing)
        match = re.search(r'\b([abc])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def extract_factual_answer(
    response_text: str,
    prompt_id: int
) -> Optional[str]:
    """Extract factual answer from response.
    
    Args:
        response_text: The model's response
        prompt_id: The prompt ID (13-16 are factual)
        
    Returns:
        Extracted answer or None if not found
    """
    text = normalize_text(response_text)
    
    if prompt_id == 13:  # Water formula
        if "h2o" in text or "h₂o" in text:
            return "H2O"
    
    elif prompt_id == 14:  # WWII end year
        match = re.search(r'\b(19\d{2})\b', text)
        if match:
            return match.group(1)
    
    elif prompt_id == 15:  # Canada capital
        # Look for Ottawa in various forms
        if "ottawa" in text:
            return "Ottawa"
    
    elif prompt_id == 16:  # 1984 author
        if "orwell" in text:
            return "George Orwell"
    
    return None


def extract_answer(
    response_text: str,
    task_type: str,
    prompt_id: int
) -> Optional[str]:
    """Extract answer based on task type.
    
    Args:
        response_text: The model's response
        task_type: The task type
        prompt_id: The prompt ID
        
    Returns:
        Extracted answer or None
    """
    if task_type == "classification":
        return extract_classification_label(response_text, prompt_id)
    elif task_type == "reasoning":
        return extract_reasoning_answer(response_text, prompt_id)
    elif task_type == "factual":
        return extract_factual_answer(response_text, prompt_id)
    return None


def check_cross_lingual_match(
    answers: Dict[str, Optional[str]]
) -> Tuple[str, Dict[str, Optional[str]]]:
    """Check if answers match across languages.
    
    Args:
        answers: Dictionary {language: answer}
        
    Returns:
        Tuple of (result, extracted_keys)
        result: "match", "mismatch", or "uncertain"
    """
    valid_answers = {k: v for k, v in answers.items() if v is not None}
    
    if len(valid_answers) < 2:
        return "uncertain", answers
    
    unique_values = set(valid_answers.values())
    
    if len(unique_values) == 1:
        return "match", answers
    else:
        return "mismatch", answers


def compute_task_metrics(
    responses_file: Optional[Path] = None,
    task_metrics_file: Optional[Path] = None,
    stability_file: Optional[Path] = None,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute task-aware metrics for discrete-answer tasks.
    
    Args:
        responses_file: Path to responses.jsonl
        task_metrics_file: Path to save task_metrics.csv
        stability_file: Path to stability.csv (will be appended)
        show_progress: Show progress
        
    Returns:
        Tuple of (task_metrics_df, stability_df)
    """
    if task_metrics_file is None:
        task_metrics_file = DEFAULT_TASK_METRICS_FILE
    if stability_file is None:
        stability_file = DEFAULT_STABILITY_FILE
    
    # Load responses
    responses = load_responses(responses_file)
    
    if not responses:
        print("No responses found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter to discrete-answer tasks only
    discrete_responses = [r for r in responses if is_discrete_task(r["task_type"])]
    
    if not discrete_responses:
        print("No discrete-answer responses found!")
        return pd.DataFrame(), pd.DataFrame()
    
    if show_progress:
        print(f"Processing {len(discrete_responses)} discrete-answer responses...")
    
    # Organize responses by model, prompt, run, language
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in discrete_responses:
        key = (r["model_id"], r["prompt_id"], r["task_type"])
        organized[key][r["run_id"]][r["language"]] = r["response_text"]
    
    # Compute cross-lingual metrics
    task_metrics_rows = []
    
    for (model_id, prompt_id, task_type), runs_data in organized.items():
        for run_id, lang_responses in runs_data.items():
            # Extract answers for each language
            answers = {}
            for lang, response_text in lang_responses.items():
                answers[lang] = extract_answer(response_text, task_type, prompt_id)
            
            # Check cross-lingual agreement
            result, extracted = check_cross_lingual_match(answers)
            
            task_metrics_rows.append({
                "model_id": model_id,
                "prompt_id": prompt_id,
                "task_type": task_type,
                "check_type": task_type,
                "run_id": run_id,
                "result": result,
                "key_en": extracted.get("EN"),
                "key_de": extracted.get("DE"),
                "key_tr": extracted.get("TR")
            })
    
    task_metrics_df = pd.DataFrame(task_metrics_rows)
    
    # Compute intra-language stability (run1 vs run2)
    stability_rows = []
    
    for (model_id, prompt_id, task_type), runs_data in organized.items():
        if 1 in runs_data and 2 in runs_data:
            for language in ["EN", "DE", "TR"]:
                if language in runs_data[1] and language in runs_data[2]:
                    answer1 = extract_answer(runs_data[1][language], task_type, prompt_id)
                    answer2 = extract_answer(runs_data[2][language], task_type, prompt_id)
                    
                    if answer1 is not None and answer2 is not None:
                        stability_value = 1 if answer1 == answer2 else 0
                    else:
                        stability_value = None
                    
                    if stability_value is not None:
                        stability_rows.append({
                            "model_id": model_id,
                            "prompt_id": prompt_id,
                            "task_type": task_type,
                            "language": language,
                            "stability_type": "discrete_match",
                            "run_id_a": 1,
                            "run_id_b": 2,
                            "stability_value": stability_value
                        })
    
    stability_df = pd.DataFrame(stability_rows)
    
    # Save task metrics
    task_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    task_metrics_df.to_csv(task_metrics_file, index=False)
    
    # Append stability to existing file or create new
    if stability_file.exists():
        existing_stability = pd.read_csv(stability_file)
        # Remove any existing discrete_match entries to avoid duplicates
        existing_stability = existing_stability[existing_stability["stability_type"] != "discrete_match"]
        combined_stability = pd.concat([existing_stability, stability_df], ignore_index=True)
        combined_stability.to_csv(stability_file, index=False)
    else:
        stability_df.to_csv(stability_file, index=False)
    
    if show_progress:
        print(f"Saved task metrics to {task_metrics_file}")
        print(f"Updated stability in {stability_file}")
    
    return task_metrics_df, stability_df


def load_task_metrics(task_metrics_file: Optional[Path] = None) -> pd.DataFrame:
    """Load task metrics from CSV.
    
    Args:
        task_metrics_file: Path to task_metrics.csv
        
    Returns:
        DataFrame with task metrics
    """
    if task_metrics_file is None:
        task_metrics_file = DEFAULT_TASK_METRICS_FILE
    
    if not task_metrics_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(task_metrics_file)


def aggregate_task_metrics_by_task_type(
    task_metrics_df: pd.DataFrame,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate task metrics by task type.
    
    Args:
        task_metrics_df: Task metrics DataFrame
        model_id: Optional model filter
        
    Returns:
        DataFrame with match rates by task type
    """
    df = task_metrics_df.copy()
    if model_id:
        df = df[df["model_id"] == model_id]
    
    def compute_rates(group):
        total = len(group)
        matches = (group["result"] == "match").sum()
        mismatches = (group["result"] == "mismatch").sum()
        uncertain = (group["result"] == "uncertain").sum()
        
        return pd.Series({
            "total": total,
            "matches": matches,
            "mismatches": mismatches,
            "uncertain": uncertain,
            "match_rate": matches / total if total > 0 else 0,
            "mismatch_rate": mismatches / total if total > 0 else 0
        })
    
    return df.groupby("task_type").apply(compute_rates).round(4)


def aggregate_task_metrics_by_prompt(
    task_metrics_df: pd.DataFrame,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate task metrics by prompt.
    
    Args:
        task_metrics_df: Task metrics DataFrame
        model_id: Optional model filter
        
    Returns:
        DataFrame with match rates by prompt
    """
    df = task_metrics_df.copy()
    if model_id:
        df = df[df["model_id"] == model_id]
    
    def compute_rates(group):
        total = len(group)
        matches = (group["result"] == "match").sum()
        
        return pd.Series({
            "task_type": group["task_type"].iloc[0],
            "total": total,
            "matches": matches,
            "match_rate": matches / total if total > 0 else 0
        })
    
    return df.groupby("prompt_id").apply(compute_rates)


def get_mismatched_examples(
    task_metrics_df: pd.DataFrame,
    responses: Optional[List[Dict[str, Any]]] = None,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Get examples where answers don't match across languages.
    
    Args:
        task_metrics_df: Task metrics DataFrame
        responses: Optional list of response records
        model_id: Optional model filter
        
    Returns:
        DataFrame with mismatched examples
    """
    df = task_metrics_df[task_metrics_df["result"] == "mismatch"].copy()
    
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
            for lang in ["EN", "DE", "TR"]:
                key = (row["model_id"], row["prompt_id"], lang, row["run_id"])
                texts[f"response_{lang}"] = resp_dict.get(key, "")
            return pd.Series(texts)
        
        response_cols = df.apply(get_responses, axis=1)
        df = pd.concat([df, response_cols], axis=1)
    
    return df


def compute_discrete_stability_summary(
    stability_df: pd.DataFrame,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """Compute stability summary for discrete tasks.
    
    Args:
        stability_df: Stability DataFrame (filtered to discrete_match)
        model_id: Optional model filter
        
    Returns:
        DataFrame with stability rates by language
    """
    df = stability_df[stability_df["stability_type"] == "discrete_match"].copy()
    
    if model_id:
        df = df[df["model_id"] == model_id]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    return df.groupby("language").agg({
        "stability_value": ["mean", "std", "sum", "count"]
    }).round(4)


if __name__ == "__main__":
    # Test extraction functions
    print("Testing answer extraction...")
    
    # Classification tests
    test_cases = [
        ("Negative", "classification", 5),
        ("Disagree", "classification", 6),
        ("Request", "classification", 7),
        ("Informal", "classification", 8),
        ("A", "reasoning", 9),
        ("60 km/h", "reasoning", 10),
        ("90", "reasoning", 11),
        ("B", "reasoning", 12),
        ("H2O", "factual", 13),
        ("1945", "factual", 14),
        ("Ottawa", "factual", 15),
        ("George Orwell", "factual", 16),
    ]
    
    for response, task_type, prompt_id in test_cases:
        answer = extract_answer(response, task_type, prompt_id)
        print(f"  {task_type} #{prompt_id}: '{response}' -> {answer}")
