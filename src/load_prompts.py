"""
Load and filter prompts from the primary dataset.

This module provides utilities for loading prompts from prompts_primary.csv
and filtering them by task type, language, or prompt ID.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml


# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_PROMPTS_FILE = DEFAULT_DATA_DIR / "prompts_primary.csv"
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "configs"


def load_config(config_name: str = "models.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        config_name: Name of the config file (models.yaml or embeddings.yaml)
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = DEFAULT_CONFIG_DIR / config_name
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_control_line(language: str) -> str:
    """Get the response control line for a given language.
    
    Args:
        language: Language code (EN, DE, or TR)
        
    Returns:
        Control line string to prepend to prompts
    """
    config = load_config("models.yaml")
    control_lines = config.get("control_lines", {})
    return control_lines.get(language, "")


def load_prompts(
    prompts_file: Optional[Path] = None,
    task_type: Optional[str] = None,
    language: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None,
    prepend_control_line: bool = True
) -> pd.DataFrame:
    """Load prompts from CSV with optional filtering.
    
    Args:
        prompts_file: Path to the prompts CSV file. Defaults to data/prompts_primary.csv
        task_type: Filter by task type (summarization, classification, reasoning, factual, creative)
        language: Filter by language (EN, DE, TR)
        prompt_ids: Filter by specific prompt IDs
        prepend_control_line: Whether to prepend the language-specific control line
        
    Returns:
        DataFrame with columns: prompt_id, task_type, language, text
    """
    if prompts_file is None:
        prompts_file = DEFAULT_PROMPTS_FILE
    
    df = pd.read_csv(prompts_file)
    
    # Apply filters
    if task_type is not None:
        df = df[df["task_type"] == task_type]
    
    if language is not None:
        df = df[df["language"] == language]
    
    if prompt_ids is not None:
        df = df[df["prompt_id"].isin(prompt_ids)]
    
    # Prepend control lines if requested
    if prepend_control_line:
        df = df.copy()
        df["text"] = df.apply(
            lambda row: f"{get_control_line(row['language'])}\n\n{row['text']}", 
            axis=1
        )
    
    return df.reset_index(drop=True)


def load_prompts_as_list(
    prompts_file: Optional[Path] = None,
    task_type: Optional[str] = None,
    language: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None,
    prepend_control_line: bool = True
) -> List[Dict[str, Any]]:
    """Load prompts as a list of dictionaries.
    
    Args:
        prompts_file: Path to the prompts CSV file
        task_type: Filter by task type
        language: Filter by language
        prompt_ids: Filter by specific prompt IDs
        prepend_control_line: Whether to prepend the language-specific control line
        
    Returns:
        List of dictionaries with keys: prompt_id, task_type, language, text
    """
    df = load_prompts(
        prompts_file=prompts_file,
        task_type=task_type,
        language=language,
        prompt_ids=prompt_ids,
        prepend_control_line=prepend_control_line
    )
    return df.to_dict(orient="records")


def get_task_types() -> List[str]:
    """Get list of all task types in the dataset.
    
    Returns:
        List of task type strings
    """
    df = load_prompts(prepend_control_line=False)
    return sorted(df["task_type"].unique().tolist())


def get_languages() -> List[str]:
    """Get list of all languages in the dataset.
    
    Returns:
        List of language codes
    """
    df = load_prompts(prepend_control_line=False)
    return sorted(df["language"].unique().tolist())


def get_prompt_ids_by_task_type(task_type: str) -> List[int]:
    """Get prompt IDs for a specific task type.
    
    Args:
        task_type: The task type to filter by
        
    Returns:
        List of prompt IDs
    """
    df = load_prompts(task_type=task_type, prepend_control_line=False)
    return sorted(df["prompt_id"].unique().tolist())


def is_open_text_task(task_type: str) -> bool:
    """Check if a task type is open-text (uses LaBSE similarity).
    
    Args:
        task_type: The task type to check
        
    Returns:
        True if open-text (summarization, creative), False otherwise
    """
    return task_type in ["summarization", "creative"]


def is_discrete_task(task_type: str) -> bool:
    """Check if a task type is discrete-answer (uses exact match).
    
    Args:
        task_type: The task type to check
        
    Returns:
        True if discrete (classification, reasoning, factual), False otherwise
    """
    return task_type in ["classification", "reasoning", "factual"]


if __name__ == "__main__":
    # Test loading
    print("Loading all prompts...")
    df = load_prompts()
    print(f"Total prompts: {len(df)}")
    print(f"Task types: {get_task_types()}")
    print(f"Languages: {get_languages()}")
    
    print("\nSample prompt (EN, summarization):")
    sample = load_prompts(task_type="summarization", language="EN")
    if len(sample) > 0:
        print(sample.iloc[0]["text"][:200] + "...")
