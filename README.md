# Cross-Lingual Prompt Consistency Evaluation

A framework for evaluating how consistently Large Language Models (LLMs) respond to semantically equivalent prompts across different languages (English, German, Turkish).

## Overview

This project measures cross-lingual prompt consistency using:
- **LaBSE embeddings** for semantic similarity on open-text tasks (summarization, creative)
- **Exact/normalized match** for discrete-answer tasks (classification, reasoning, factual)
- **Ollama** for local inference with open-source models

### Key Features

- Evaluate multiple open-source LLMs (gemma3:1b, llama3.2:1b)
- Compare responses across EN-DE, EN-TR, and DE-TR language pairs
- Measure both cross-lingual consistency and intra-language stability
- Generate heatmaps, distribution plots, and summary tables
- Flag and review low-consistency examples for qualitative analysis

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd llm-crossslingual-prompt-consistency

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull gemma3:1b
ollama pull llama3.2:1b
```

## Project Structure

```
├── configs/
│   ├── models.yaml           # Model IDs and inference parameters
│   └── embeddings.yaml       # LaBSE configuration
├── data/
│   ├── prompts_primary.csv   # 20 prompts × 3 languages
│   ├── responses.jsonl       # Generated LLM responses (cached)
│   ├── metrics.csv           # Cross-lingual similarity scores
│   ├── stability.csv         # Intra-language stability metrics
│   └── task_metrics.csv      # Discrete-answer agreement results
├── src/
│   ├── load_prompts.py       # Prompt loading and filtering
│   ├── infer_ollama.py       # Ollama inference client
│   ├── embed_labse.py        # LaBSE embedding generation
│   ├── similarity.py         # Cosine similarity computation
│   ├── task_checks.py        # Discrete-answer extraction
│   └── plots.py              # Visualization generation
├── notebooks/
│   ├── 01_collect_responses.ipynb    # Run inference
│   ├── 02_metrics_and_plots.ipynb    # Compute metrics and visualize
│   └── 03_qualitative_review.ipynb   # Review flagged examples
└── outputs/
    ├── plots/                # Generated visualizations
    └── reports/              # Summary tables and evidence sets
```

## Usage

### Quick Start

1. **Start Ollama server:**
   ```bash
   ollama serve
   ```

2. **Run the notebooks in order:**
   - `01_collect_responses.ipynb` - Generate LLM responses
   - `02_metrics_and_plots.ipynb` - Compute metrics and create visualizations
   - `03_qualitative_review.ipynb` - Review flagged low-consistency examples

### Using the Source Modules

```python
from src.load_prompts import load_prompts, get_task_types
from src.infer_ollama import run_full_inference, load_responses
from src.similarity import compute_all_metrics
from src.task_checks import compute_task_metrics
from src.plots import generate_all_plots

# Load prompts
prompts = load_prompts(task_type='summarization', language='EN')

# Run inference
run_full_inference()

# Compute metrics
metrics_df, stability_df = compute_all_metrics()
task_metrics_df, _ = compute_task_metrics()

# Generate visualizations
generate_all_plots()
```

## Prompt Dataset

The dataset contains 20 prompts across 5 task types:

| Task Type | Prompts | Evaluation Method |
|-----------|---------|-------------------|
| Summarization | 1-4 | LaBSE cosine similarity |
| Classification | 5-8 | Exact label match |
| Reasoning | 9-12 | Exact answer match |
| Factual | 13-16 | Exact answer match |
| Creative | 17-20 | LaBSE cosine similarity |

Each prompt is translated into English (EN), German (DE), and Turkish (TR).

## Metrics

### Cross-Lingual Consistency
- **Open-text tasks**: Cosine similarity between LaBSE embeddings of responses in different languages (same run_id)
- **Discrete-answer tasks**: Match rate of extracted answers across languages

### Intra-Language Stability
Measures model randomness by comparing run1 vs run2 within each language.

### Flagging
- Bottom 10% similarity scores are flagged for qualitative review
- Mismatched discrete-answer cases are flagged separately

## Configuration

### Model Parameters (`configs/models.yaml`)
```yaml
inference:
  temperature: 0.3
  num_predict: 256
  runs_per_prompt: 2
```

### Embedding Configuration (`configs/embeddings.yaml`)
```yaml
embedding:
  model_name: sentence-transformers/LaBSE
  normalize: true
```

## Output Files

| File | Description |
|------|-------------|
| `data/responses.jsonl` | All LLM responses with metadata |
| `data/metrics.csv` | Cross-lingual cosine similarities |
| `data/stability.csv` | Intra-language stability scores |
| `data/task_metrics.csv` | Discrete-answer match results |
| `outputs/plots/*.png` | Heatmaps and distribution plots |
| `outputs/reports/*.csv` | Summary tables |

## License

MIT License

## Authors

- Baran Akin
- Mehdi Farzin

Supervisor: Shan Faiz
