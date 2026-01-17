"""
Local inference client for Ollama.

This module handles all LLM inference via the Ollama local server,
with support for multiple runs, caching, and non-compliance detection.
"""

import json
import ollama
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Generator
from tqdm import tqdm
import re

from .load_prompts import load_prompts_as_list, load_config, is_discrete_task


# Default paths
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_RESPONSES_FILE = DEFAULT_DATA_DIR / "responses.jsonl"


def get_inference_params() -> Dict[str, Any]:
    """Get inference parameters from config.
    
    Returns:
        Dictionary with temperature, num_predict, runs_per_prompt
    """
    config = load_config("models.yaml")
    return config.get("inference", {
        "temperature": 0.3,
        "num_predict": 256,
        "runs_per_prompt": 2
    })


def get_models() -> List[Dict[str, str]]:
    """Get list of models from config.
    
    Returns:
        List of model configurations with id, name, provider
    """
    config = load_config("models.yaml")
    return config.get("models", [])


def detect_non_compliance(response_text: str, task_type: str, prompt_id: int) -> bool:
    """Detect if a response violates expected output FORMAT.
    
    This function only checks FORMAT compliance, not factual correctness.
    Factual accuracy is measured separately via LaBSE semantic similarity.
    
    Rules:
    - Classification: Must contain expected label (multilingual)
    - Reasoning: Must contain A/B/C or numbers as appropriate
    - Factual: NO CHECK (accuracy ≠ format compliance)
    - Open-text: NO CHECK (summarization, creative)
    
    Args:
        response_text: The model's response text
        task_type: The task type of the prompt
        prompt_id: The prompt ID
        
    Returns:
        True if non-compliant (format violation), False otherwise
    """
    text = response_text.strip()
    text_lower = text.lower()
    
    # Only check discrete-answer tasks for format compliance
    if not is_discrete_task(task_type):
        return False
    
    # Classification checks - must have the label (multilingual support)
    if task_type == "classification":
        if prompt_id == 5:  # Sentiment: Positive/Negative
            valid = ["positive", "negative", "positiv", "negativ", "olumlu", "olumsuz"]
        elif prompt_id == 6:  # Agreement: Agree/Disagree
            valid = ["agree", "disagree"]
        elif prompt_id == 7:  # Intent: Request/Complaint
            valid = ["request", "complaint"]
        elif prompt_id == 8:  # Formality: Formal/Informal
            valid = ["formal", "informal"]
        else:
            return False
        
        # Check if response contains a valid label
        if not any(v in text_lower for v in valid):
            return True
    
    # Reasoning checks - must have expected format
    elif task_type == "reasoning":
        if prompt_id in [9, 12]:  # A/B/C answers
            if not re.search(r'\b[ABC]\b', text, re.IGNORECASE):
                return True
        elif prompt_id in [10, 11]:  # Numeric answers
            if not re.search(r'\d+', text):
                return True
    
    # Factual: NO FORMAT CHECK
    # Factual correctness is not a format issue - LaBSE measures semantic consistency
    # Wrong answers (e.g., "Toronto" instead of "Ottawa") are still valid FORMAT
    
    return False


def generate_response(
    model_id: str,
    prompt_text: str,
    temperature: float = 0.3,
    num_predict: int = 256
) -> str:
    """Generate a single response from Ollama.
    
    Args:
        model_id: Ollama model ID (e.g., 'gemma3:1b')
        prompt_text: The full prompt text
        temperature: Sampling temperature
        num_predict: Maximum tokens to generate
        
    Returns:
        Generated response text
    """
    response = ollama.generate(
        model=model_id,
        prompt=prompt_text,
        options={
            "temperature": temperature,
            "num_predict": num_predict
        }
    )
    return response["response"]


def create_response_record(
    prompt: Dict[str, Any],
    model_id: str,
    run_id: int,
    response_text: str,
    temperature: float,
    num_predict: int
) -> Dict[str, Any]:
    """Create a response record for JSONL storage.
    
    Args:
        prompt: Prompt dictionary with prompt_id, task_type, language, text
        model_id: The model ID used
        run_id: Run number (1 or 2)
        response_text: The generated response
        temperature: Temperature used
        num_predict: Max tokens setting
        
    Returns:
        Complete response record dictionary
    """
    non_compliant = detect_non_compliance(
        response_text, 
        prompt["task_type"], 
        prompt["prompt_id"]
    )
    
    return {
        "prompt_id": prompt["prompt_id"],
        "task_type": prompt["task_type"],
        "language": prompt["language"],
        "model_id": model_id,
        "run_id": run_id,
        "temperature": temperature,
        "max_new_tokens": num_predict,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_text": prompt["text"],
        "response_text": response_text,
        "non_compliant": non_compliant
    }


def load_responses(responses_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load existing responses from JSONL file.
    
    Args:
        responses_file: Path to responses.jsonl
        
    Returns:
        List of response records
    """
    if responses_file is None:
        responses_file = DEFAULT_RESPONSES_FILE
    
    responses = []
    if responses_file.exists():
        with open(responses_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    responses.append(json.loads(line))
    return responses


def save_response(
    response: Dict[str, Any],
    responses_file: Optional[Path] = None
) -> None:
    """Append a single response to the JSONL file.
    
    Args:
        response: Response record to save
        responses_file: Path to responses.jsonl
    """
    if responses_file is None:
        responses_file = DEFAULT_RESPONSES_FILE
    
    # Ensure parent directory exists
    responses_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(responses_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(response, ensure_ascii=False) + "\n")


def get_existing_keys(responses_file: Optional[Path] = None) -> set:
    """Get set of existing (model_id, prompt_id, language, run_id) keys.
    
    Args:
        responses_file: Path to responses.jsonl
        
    Returns:
        Set of tuples representing existing response keys
    """
    responses = load_responses(responses_file)
    return {
        (r["model_id"], r["prompt_id"], r["language"], r["run_id"])
        for r in responses
    }


def run_inference(
    model_ids: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    prompt_ids: Optional[List[int]] = None,
    runs: Optional[List[int]] = None,
    responses_file: Optional[Path] = None,
    skip_existing: bool = True,
    show_progress: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """Run inference for specified combinations.
    
    Args:
        model_ids: List of model IDs to use. Defaults to all configured models.
        task_types: Filter prompts by task type
        languages: Filter prompts by language
        prompt_ids: Filter prompts by ID
        runs: List of run IDs (e.g., [1, 2]). Defaults to config value.
        responses_file: Path to save responses
        skip_existing: Skip already-completed combinations
        show_progress: Show progress bar
        
    Yields:
        Response records as they are generated
    """
    # Load configuration
    params = get_inference_params()
    temperature = params["temperature"]
    num_predict = params["num_predict"]
    runs_per_prompt = params["runs_per_prompt"]
    
    if model_ids is None:
        model_ids = [m["id"] for m in get_models()]
    
    if runs is None:
        runs = list(range(1, runs_per_prompt + 1))
    
    # Load prompts
    prompts = load_prompts_as_list(
        task_type=task_types[0] if task_types and len(task_types) == 1 else None,
        language=languages[0] if languages and len(languages) == 1 else None,
        prompt_ids=prompt_ids
    )
    
    # Apply additional filters if multiple values
    if task_types and len(task_types) > 1:
        prompts = [p for p in prompts if p["task_type"] in task_types]
    if languages and len(languages) > 1:
        prompts = [p for p in prompts if p["language"] in languages]
    
    # Get existing keys for skipping
    existing_keys = get_existing_keys(responses_file) if skip_existing else set()
    
    # Calculate total work
    total_calls = len(model_ids) * len(prompts) * len(runs)
    
    # Build work items
    work_items = []
    for model_id in model_ids:
        for prompt in prompts:
            for run_id in runs:
                key = (model_id, prompt["prompt_id"], prompt["language"], run_id)
                if key not in existing_keys:
                    work_items.append((model_id, prompt, run_id))
    
    if show_progress:
        print(f"Total inference calls needed: {len(work_items)} (skipped {total_calls - len(work_items)} existing)")
    
    # Run inference
    iterator = tqdm(work_items, desc="Generating responses") if show_progress else work_items
    
    for model_id, prompt, run_id in iterator:
        try:
            response_text = generate_response(
                model_id=model_id,
                prompt_text=prompt["text"],
                temperature=temperature,
                num_predict=num_predict
            )
            
            record = create_response_record(
                prompt=prompt,
                model_id=model_id,
                run_id=run_id,
                response_text=response_text,
                temperature=temperature,
                num_predict=num_predict
            )
            
            # Save to file
            save_response(record, responses_file)
            
            yield record
            
        except Exception as e:
            print(f"Error generating response for {model_id}, prompt {prompt['prompt_id']}, "
                  f"lang {prompt['language']}, run {run_id}: {e}")


def run_full_inference(
    responses_file: Optional[Path] = None,
    skip_existing: bool = True
) -> int:
    """Run inference for all models, prompts, languages, and runs.
    
    Args:
        responses_file: Path to save responses
        skip_existing: Skip already-completed combinations
        
    Returns:
        Number of responses generated
    """
    count = 0
    for _ in run_inference(
        responses_file=responses_file,
        skip_existing=skip_existing,
        show_progress=True
    ):
        count += 1
    return count


def get_responses_by_prompt(
    prompt_id: int,
    model_id: Optional[str] = None,
    responses_file: Optional[Path] = None
) -> Dict[str, Dict[int, str]]:
    """Get responses for a specific prompt grouped by language and run.
    
    Args:
        prompt_id: The prompt ID
        model_id: Optional model filter
        responses_file: Path to responses.jsonl
        
    Returns:
        Dictionary: {language: {run_id: response_text}}
    """
    responses = load_responses(responses_file)
    
    result = {}
    for r in responses:
        if r["prompt_id"] == prompt_id:
            if model_id is None or r["model_id"] == model_id:
                lang = r["language"]
                run = r["run_id"]
                if lang not in result:
                    result[lang] = {}
                result[lang][run] = r["response_text"]
    
    return result


if __name__ == "__main__":
    # Test loading config
    print("Models configured:")
    for model in get_models():
        print(f"  - {model['id']}: {model['name']}")
    
    print("\nInference parameters:")
    params = get_inference_params()
    for k, v in params.items():
        print(f"  {k}: {v}")
