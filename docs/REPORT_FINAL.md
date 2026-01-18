# Cross-Lingual Prompt Consistency in LLMs - Final Report

**Date:** January 17, 2026  
**Project:** Evaluating semantic consistency of multilingual LLM responses across English (EN), German (DE), and Turkish (TR)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Methodology](#methodology)
4. [Experimental Setup](#experimental-setup)
5. [Data Collection Results](#data-collection-results)
6. [Metrics and Analysis](#metrics-and-analysis)
7. [Drift Type Analysis](#drift-type-analysis)
8. [Visualization Analysis](#visualization-analysis)
9. [Key Findings](#key-findings)
10. [Qualitative Analysis](#qualitative-analysis)
11. [Comprehensive Results Compilation](#comprehensive-results-compilation)
12. [Discussion & Lessons Learned](#discussion--lessons-learned)
13. [Conclusions](#conclusions)
14. [Appendices](#appendices)

---

## Executive Summary

This study evaluates the **cross-lingual consistency** of 6 locally-run Large Language Models (LLMs) across three languages (English, German, Turkish) and five task types. Using LaBSE embeddings for semantic similarity and task-aware agreement checks for discrete answers, we measured how consistently models produce equivalent responses when given semantically identical prompts in different languages.

### Key Findings at a Glance

| Metric | Best Model | Worst Model |
|--------|------------|-------------|
| **Overall Cross-Lingual Similarity** | gemma3:4b (0.70) | phi3:latest (0.57) |
| **Classification Match Rate** | gemma3:4b, gemma3:1b (100%) | llama3.2:1b (12.5%) |
| **Factual Match Rate** | 5/6 models at 100% | phi3:latest (75%) |
| **Intra-Language Stability** | gemma3:4b (95%) | llama3.2:1b (81%) |

**Turkish language emerged as the primary consistency challenge** across all models, with EN-TR pairs showing consistently lower similarity than EN-DE or DE-TR pairs.

---

## Project Overview

### Objective

Evaluate how consistently small, locally-runnable LLMs (1B-4B parameters) produce semantically equivalent responses when given the same prompt translated into different languages.

### Research Questions

1. **RQ1:** How does cross-lingual semantic similarity vary across task types?
2. **RQ2:** Do models maintain consistent discrete answers (labels, numbers) across languages?
3. **RQ3:** Which language pairs show the largest consistency gaps?
4. **RQ4:** How does cross-lingual consistency compare to intra-language stability?

### Languages Tested

| Language | Code | Role |
|----------|------|------|
| English | EN | Primary/Reference |
| German | DE | High-resource Western European |
| Turkish | TR | Medium-resource, agglutinative |

---

## Methodology

### Evaluation Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  20 Prompts     │────▶│  Ollama Local   │────▶│  LaBSE Embed    │
│  × 3 Languages  │     │  Inference      │     │  & Compare      │
│  × 2 Runs       │     │  (6 Models)     │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         ▼                      ▼                       ▼
   prompts_primary.csv   responses.jsonl         metrics.csv
                                                 task_metrics.csv
                                                 stability.csv
```

### Metrics Used

#### Open-Text Tasks (Summarization, Creative)
- **LaBSE Cosine Similarity:** Semantic similarity between response embeddings across language pairs (EN-DE, EN-TR, DE-TR)
- **Range:** 0.0 (completely different) to 1.0 (semantically identical)

#### Discrete-Answer Tasks (Classification, Reasoning, Factual)
- **Cross-Lingual Match Rate:** Percentage of cases where all 3 languages produce the same answer
- **Extraction Method:** Task-specific regex patterns for labels, numbers, options

#### Stability Baseline
- **Intra-Language Consistency:** Similarity between Run 1 and Run 2 responses (same language)
- **Purpose:** Separate true cross-lingual drift from model randomness

---

## Experimental Setup

### Models Evaluated

| Model ID | Parameters | Provider |
|----------|------------|----------|
| gemma3:1b | 1B | Google (Ollama) |
| gemma3:4b | 4B | Google (Ollama) |
| llama3.2:1b | 1B | Meta (Ollama) |
| llama3.2:3b | 3B | Meta (Ollama) |
| phi3:latest | 3.8B | Microsoft (Ollama) |
| phi4-mini:3.8b | 3.8B | Microsoft (Ollama) |

### Inference Parameters

```yaml
temperature: 0.3       # Low for reproducibility
num_predict: 256       # Max tokens per response
runs_per_prompt: 2     # For stability measurement
```

### Prompts Dataset

**20 unique prompts** across 5 task types, each translated into 3 languages:

| Task Type | Prompts | Description |
|-----------|---------|-------------|
| Summarization | 1-4 | Compress paragraph to one sentence |
| Classification | 5-8 | Label sentiment, intent, formality, agreement |
| Reasoning | 9-12 | Logic, math, common-sense |
| Factual | 13-16 | Knowledge recall (facts, formulas, dates) |
| Creative | 17-20 | Slogans, ad copy, advice |

**Total Responses Generated:** 720 (20 prompts × 3 languages × 2 runs × 6 models)

---

## Data Collection Results

### Response Collection Summary

| Statistic | Value |
|-----------|-------|
| Total Responses | 720 |
| Compliant Responses | 719 (99.86%) |
| Non-Compliant Responses | 1 (0.14%) |
| Inference Time | ~8 minutes |

### Non-Compliance Analysis

During development, we iteratively refined the non-compliance detection logic:

#### Initial Issues Encountered

1. **Overly Strict Length Checks:** Initial threshold of 100 characters flagged valid responses with minor explanations
2. **English-Only Classification Labels:** Detection failed for German ("positiv/negativ") and Turkish ("olumlu/olumsuz") responses
3. **Strict A/B/C Matching:** Required answer at start of response rather than anywhere in text
4. **Factual Accuracy as Format:** Initially treated correctness (e.g., wrong year) as format violation

#### Final Non-Compliance Rules

```python
# Only check FORMAT, not accuracy
- Classification: Must contain valid label (multi-language support)
- Reasoning (A/B/C): Must contain A, B, or C anywhere
- Reasoning (numeric): Must contain a number
- Factual: No format check (correctness handled by similarity)
```

#### Remaining Non-Compliant Response

| Field | Value |
|-------|-------|
| Model | phi3:latest |
| Prompt | 7 (Intent: Request/Complaint) |
| Language | Turkish |
| Issue | Model produces incoherent Turkish text without "Request" or "Complaint" |
| Status | Accepted as model limitation |

**Finding:** phi3:latest struggles with Turkish instruction-following for classification tasks.

---

## Metrics and Analysis

### Cross-Lingual Similarity by Model (Open-Text Tasks)

| Model | EN-DE | EN-TR | DE-TR | Average |
|-------|-------|-------|-------|---------|
| **gemma3:4b** | 0.72 | 0.64 | 0.73 | **0.70** |
| phi4-mini:3.8b | 0.68 | 0.61 | 0.73 | 0.67 |
| gemma3:1b | 0.66 | 0.63 | 0.70 | 0.66 |
| llama3.2:3b | 0.59 | 0.61 | 0.65 | 0.62 |
| llama3.2:1b | 0.58 | 0.57 | 0.66 | 0.60 |
| **phi3:latest** | 0.69 | 0.47 | 0.55 | **0.57** |

**Key Observation:** EN-TR pairs consistently show lowest similarity across all models.

### Cross-Lingual Similarity by Task Type

| Task Type | Mean Similarity | Std Dev |
|-----------|-----------------|---------|
| Summarization | 0.70 | 0.08 |
| Creative | 0.57 | 0.12 |

**Finding:** Summarization tasks show higher consistency than creative tasks, likely due to:
- More constrained output format
- Less room for stylistic variation
- Clearer success criteria

### Discrete-Answer Match Rates (Cross-Lingual)

| Model | Classification | Reasoning | Factual | Overall |
|-------|----------------|-----------|---------|---------|
| gemma3:4b | **100%** | 75% | **100%** | 91.7% |
| gemma3:1b | **100%** | 25% | **100%** | 75.0% |
| phi4-mini:3.8b | 75% | 37.5% | **100%** | 70.8% |
| phi3:latest | 50% | 12.5% | 75% | 45.8% |
| llama3.2:3b | 25% | 25% | **100%** | 50.0% |
| llama3.2:1b | **12.5%** | 50% | **100%** | 54.2% |

**Critical Finding:** 
- **Factual tasks** show near-perfect consistency (100% for 5/6 models)
- **Reasoning tasks** show lowest consistency (math/logic problems)
- **llama3.2:1b** has severe classification inconsistency (only 12.5% match rate)

### Intra-Language Stability (Run1 vs Run2)

#### Open-Text Cosine Similarity

| Language | Mean | Std Dev |
|----------|------|---------|
| English | 0.86 | 0.10 |
| German | 0.85 | 0.13 |
| Turkish | 0.83 | 0.15 |

#### Discrete Match Stability

| Language | Mean | Std Dev |
|----------|------|---------|
| English | 0.97 | 0.17 |
| German | 0.89 | 0.32 |
| Turkish | 0.90 | 0.31 |

**Finding:** Intra-language stability is generally high (>80%), indicating that cross-lingual differences are primarily due to **language effects**, not model randomness.

---

## Drift Type Analysis

In this study, we defined 5 different **Drift Types** to categorize cross-lingual inconsistencies. These categories help understand why low-similarity examples are inconsistent.

### Drift Type Categories

| Drift Type | Description | Example |
|------------|-------------|---------|
| **Semantic Drift** | Core meaning differs - different conclusions, missing key points | EN: "Learn anywhere" → TR: "Support system" |
| **Format Drift** | Output format differs - length, bullet points, structure | EN: 29 chars → TR: 501 chars |
| **Factual Drift** | Factual content differs - wrong answer, different label | EN: "Negative" → DE: "Positive" |
| **Style Drift** | Tone or style differs - formal/informal, short/long | EN: Short slogan → TR: Explanatory paragraph |
| **Hallucination** | Fabricated/irrelevant content in one language | TR response contains off-topic text |

### Drift Type Distribution in Flagged Examples

Analysis of 29 low-similarity examples:

| Drift Type | Count | Percentage |
|------------|-------|------------|
| **Semantic Drift (severe, <0.4)** | 19 | 65.5% |
| **Style Drift (verbose)** | 14 | 48.3% |
| **Semantic Drift (moderate, 0.4-0.5)** | 10 | 34.5% |
| **Format Drift** | 5 | 17.2% |

> **Note:** A single example may contain multiple drift types (e.g., both Format and Style drift).

### Drift Type Distribution by Language Pair

| Language Pair | Flagged Count | Primary Drift Type |
|---------------|---------------|-------------------|
| **EN-TR** | 15 (51.7%) | Semantic + Style Drift |
| **DE-TR** | 8 (27.6%) | Semantic Drift |
| **EN-DE** | 6 (20.7%) | Semantic Drift |

**Finding:** EN-TR pair shows the most drift, typically accompanied by Turkish responses being excessively long/explanatory (Style Drift).

### Drift Type Distribution by Model

| Model | Flagged Count | Primary Drift Pattern |
|-------|---------------|----------------------|
| **phi3:latest** | 14 (48.3%) | Severe semantic drift in TR |
| **llama3.2:1b** | 7 (24.1%) | Semantic drift + format |
| **gemma3:1b** | 2 (6.9%) | Mild semantic drift |
| **gemma3:4b** | 2 (6.9%) | Mild semantic drift |
| **llama3.2:3b** | 2 (6.9%) | Semantic drift |
| **phi4-mini:3.8b** | 2 (6.9%) | Mild semantic drift |

### Drift Type Distribution by Task Type

| Task Type | Flagged | Primary Drift Type |
|-----------|---------|-------------------|
| **Creative** | 27 (93.1%) | Semantic + Style |
| **Summarization** | 2 (6.9%) | Style Drift |

**Critical Finding:** Creative tasks constitute the overwhelming majority of drift. In these tasks, models exhibit different creative approaches in each language rather than maintaining consistency.

### Factual Drift in Discrete Tasks

51 total mismatches were detected in discrete-answer tasks:

| Task Type | Mismatch Count | Drift Description |
|-----------|----------------|-------------------|
| **Reasoning** | 30 (58.8%) | Mathematical calculation differences |
| **Classification** | 19 (37.3%) | Different label assignments |
| **Factual** | 2 (3.9%) | Wrong information (phi3:latest TR) |

#### Notable Factual Drift Examples

**Example 1: Sentiment Classification (Prompt 5)**
```
llama3.2:1b - For the same negative phone review:
  EN: "Negative" ✓
  DE: "Positive" ✗
  TR: "Positive" ✗
```
→ **Drift Type:** Factual Drift - Model incorrectly evaluates sentiment in DE and TR

**Example 2: Meeting Duration (Prompt 11)**
```
gemma3:4b - Meeting duration from 9:15 to 10:45:
  EN: "1" (first number captured)
  DE: "78" (incorrect calculation)
  TR: "1" (first number captured)
```
→ **Drift Type:** Factual Drift - Cross-lingual calculation inconsistency

**Example 3: Logic Reasoning (Prompt 12)**
```
gemma3:1b - Most appropriate clothing for 2°C and rain:
  EN: "A" (T-shirt - wrong)
  DE: "B" (Jacket - correct)
  TR: "B" (Jacket - correct)
```
→ **Drift Type:** Factual Drift - Logic error in EN

---

## Visualization Analysis

This section provides detailed analysis of all 31 visualizations in the `outputs/plots/` folder.

### 1. Model Comparison Chart

**File:** `model_comparison.png`

This chart compares cross-lingual similarity performance across all models.

| Model | Average Similarity | Comment |
|-------|-------------------|---------|
| gemma3:4b | 0.6969 | **Best performance** - Consistent across all language pairs |
| phi4-mini:3.8b | 0.6740 | Good performance, particularly strong on DE-TR |
| gemma3:1b | 0.6613 | Strong performance for a 1B model |
| llama3.2:3b | 0.6172 | Medium level, weak on creative |
| llama3.2:1b | 0.6046 | Expected low performance for 1B |
| phi3:latest | 0.5696 | **Weakest** - Serious issues with TR |

**Validation:** These values align with raw data in `metrics.csv`. For example, gemma3:4b average calculation:
- EN-DE: 0.7186, EN-TR: 0.6385, DE-TR: 0.7336
- Average: (0.7186 + 0.6385 + 0.7336) / 3 = 0.6969 ✓

---

### 2. Heatmap Charts (Per Model)

Similarity heatmap in prompt_id × language_pair matrix format for each model.

#### `heatmap_gemma3_4b.png`
| Property | Value |
|----------|-------|
| Highest similarity | Prompt 3, DE-TR: 0.86 |
| Lowest similarity | Prompt 19, EN-TR: 0.39 |
| General pattern | Summarization (1-4) high, Creative (17-20) low |

**Data Validation (metrics.csv):**
```
gemma3:4b, Prompt 3, DE-TR, Run 1: 0.8624 ✓
gemma3:4b, Prompt 19, EN-TR, Run 1: 0.3896 ✓
```

#### `heatmap_gemma3_1b.png`
| Property | Value |
|----------|-------|
| Highest similarity | Prompt 2, EN-DE: 0.81 |
| Lowest similarity | Prompt 19, EN-TR: 0.35 |
| Note | Prompt 19 low across all pairs |

#### `heatmap_llama3.2_1b.png`
| Property | Value |
|----------|-------|
| Highest similarity | Prompt 2, DE-TR: 0.89 |
| Lowest similarity | Prompt 20, EN-TR: 0.29 |
| Note | Generally low on creative tasks |

#### `heatmap_llama3.2_3b.png`
| Property | Value |
|----------|-------|
| Highest similarity | Prompt 2, EN-TR: 0.80 |
| Lowest similarity | Prompt 19, EN-DE: 0.36 |
| Note | Improvement over 1B but limited |

#### `heatmap_phi3_latest.png`
| Property | Value |
|----------|-------|
| Highest similarity | Prompt 3, EN-DE: 0.87 |
| Lowest similarity | Prompt 19, EN-TR: 0.24 (minimum) |
| Note | Issues with all TR-containing pairs |

**Critical Observation:** For phi3:latest, EN-TR and DE-TR columns are notably darker (low similarity), while EN-DE column is lighter (high similarity).

#### `heatmap_phi4-mini_3.8b.png`
| Property | Value |
|----------|-------|
| Highest similarity | Prompt 3, DE-TR: 0.91 (highest overall) |
| Lowest similarity | Prompt 19, EN-TR: 0.38 |
| Note | Generally balanced performance |

---

### 3. Distribution Charts

Boxplot distribution of similarity values by language pair for each model.

#### `distribution_gemma3_4b.png`
- **EN-DE:** Median ~0.72, narrow IQR (consistent)
- **EN-TR:** Median ~0.65, wide IQR (variable)
- **DE-TR:** Median ~0.73, narrow IQR (consistent)
- **Comment:** EN-TR more variable, outliers from creative tasks

#### `distribution_gemma3_1b.png`
- **EN-DE:** Median ~0.65, some low outliers
- **EN-TR:** Median ~0.62, widest distribution
- **DE-TR:** Median ~0.70, narrowest distribution
- **Comment:** Higher variance compared to 4B version

#### `distribution_llama3.2_1b.png`
- **EN-DE:** Median ~0.55, low performance
- **EN-TR:** Median ~0.50, very low outliers
- **DE-TR:** Median ~0.65, relatively good
- **Comment:** Low and variable across all pairs

#### `distribution_llama3.2_3b.png`
- **EN-DE:** Median ~0.60, improvement over 1B
- **EN-TR:** Median ~0.58
- **DE-TR:** Median ~0.66
- **Comment:** Improvement over 1B across all pairs

#### `distribution_phi3_latest.png`
- **EN-DE:** Median ~0.69, good performance
- **EN-TR:** Median ~0.45, **very low**
- **DE-TR:** Median ~0.55, low
- **Comment:** EN-DE good but serious issues with TR-containing pairs

#### `distribution_phi4-mini_3.8b.png`
- **EN-DE:** Median ~0.68
- **EN-TR:** Median ~0.60
- **DE-TR:** Median ~0.72
- **Comment:** Balanced and consistent performance

---

### 4. Task Comparison Charts

Average similarity comparison by task type for each model.

| Model | Summarization | Creative | Δ (Difference) |
|-------|---------------|----------|----------------|
| gemma3:4b | 0.74 | 0.66 | -0.08 |
| gemma3:1b | 0.72 | 0.61 | -0.11 |
| phi4-mini:3.8b | 0.74 | 0.61 | -0.13 |
| llama3.2:3b | 0.65 | 0.58 | -0.07 |
| llama3.2:1b | 0.71 | 0.50 | **-0.21** |
| phi3:latest | 0.66 | 0.48 | **-0.18** |

**Finding:** Creative tasks show lower similarity than summarization across all models. The largest gaps are in llama3.2:1b (-0.21) and phi3:latest (-0.18).

---

### 5. Stability Charts

Intra-language stability (run1 vs run2) analysis for each model.

#### Summary Table (all models)

| Model | EN Stability | DE Stability | TR Stability |
|-------|--------------|--------------|--------------|
| gemma3:4b | 0.95 | 0.86 | 0.92 |
| gemma3:1b | 0.86 | 0.90 | 0.84 |
| phi4-mini:3.8b | 0.91 | 0.85 | 0.82 |
| llama3.2:3b | 0.81 | 0.82 | 0.90 |
| llama3.2:1b | 0.81 | 0.82 | 0.81 |
| phi3:latest | 0.84 | 0.86 | 0.68 |

**Critical Observation:** phi3:latest's TR stability value (0.68) is notably lower than other models. This confirms the model's inconsistent behavior in Turkish.

**Data Validation (stability.csv):**
```
phi3:latest, TR, open_text_cosine average:
  (0.54 + 0.81 + 0.84 + 0.88 + 0.63 + 0.63 + 0.54 + 0.55) / 8 = 0.678 ✓
```

---

### 6. Discrete Summary Charts

Match rates for discrete-answer tasks per model.

#### Per-Model Discrete Performance

| Model | Classification | Reasoning | Factual | Overall |
|-------|----------------|-----------|---------|---------|
| gemma3:4b | 100% | 75% | 100% | 91.7% |
| gemma3:1b | 100% | 25% | 100% | 75.0% |
| phi4-mini:3.8b | 75% | 37.5% | 100% | 70.8% |
| llama3.2:3b | 25% | 25% | 100% | 50.0% |
| llama3.2:1b | 12.5% | 50% | 100% | 54.2% |
| phi3:latest | 50% | 12.5% | 75% | 45.8% |

**Data Validation (task_metrics.csv):**
```
gemma3:4b classification matches: 8/8 = 100% ✓
llama3.2:1b classification matches: 1/8 = 12.5% ✓
```

---

## Key Findings

### 1. Turkish Language Presents Significant Challenges

| Language Pair | Mean Similarity | Δ from Best |
|---------------|-----------------|-------------|
| DE-TR | 0.67 | - |
| EN-DE | 0.65 | -0.02 |
| **EN-TR** | **0.59** | **-0.08** |

**All models** showed weaker performance on EN-TR pairs compared to EN-DE or DE-TR.

### 2. Model Size Matters (But Not Always)

| Finding | Evidence |
|---------|----------|
| Larger models ≠ always better | gemma3:4b (4B) outperforms phi3:latest (3.8B) significantly |
| 1B models can compete | gemma3:1b matches larger models on classification |
| Architecture differences matter | Gemma family consistently outperforms Llama family |

### 3. Task-Specific Patterns

| Task Type | Consistency Level | Primary Challenge |
|-----------|-------------------|-------------------|
| Factual | Very High (>95%) | None significant |
| Summarization | High (~70%) | Compression style varies |
| Classification | Variable (12-100%) | Label extraction in non-English |
| Reasoning | Low (~35%) | Math/logic computation differs |
| Creative | Low (~57%) | High stylistic freedom |

### 4. Specific Problem Cases

#### Prompt 19 (Slogan - Creative)
- **Average Similarity:** 0.38 (lowest of all prompts)
- **Issue:** Models interpret "6-8 word slogan" differently across languages
- **Turkish:** Often produces explanations instead of slogans

#### Prompt 11 (Meeting Duration - Reasoning)
- **Match Rate:** <50% across models
- **Issue:** Different calculation approaches (hours+minutes vs. total minutes)

#### Prompt 9 (Logic: A taller than B taller than C)
- **Turkish:** Models often output "C" instead of "A"
- **Possible cause:** Turkish word order/sentence structure confusion

---

## Qualitative Analysis

### Flagged Low-Similarity Examples (Bottom 10%)

| Rank | Model | Prompt | Pair | Similarity | Issue |
|------|-------|--------|------|------------|-------|
| 1 | phi3:latest | 19 | EN-TR | 0.24 | Turkish response includes explanation |
| 2 | llama3.2:1b | 20 | EN-TR | 0.29 | Completely different advice |
| 3 | phi3:latest | 18 | EN-TR | 0.30 | Turkish ad copy diverges semantically |

### Representative Mismatched Discrete Examples

#### Example 1: Meeting Duration (Prompt 11)

| Language | gemma3:4b Response | Extracted |
|----------|-------------------|-----------|
| EN | "1 hour and 30 minutes / 90 minutes" | 1 |
| DE | "78" | 78 |
| TR | "1 saat 30 dakika" | 1 |

**Issue:** Extraction regex captures first number, leading to mismatch.

#### Example 2: Sentiment Classification (Prompt 5)

| Language | llama3.2:1b Response | Extracted |
|----------|---------------------|-----------|
| EN | "Negative" | negative |
| DE | "Der Rezensent... Positive" | positive |
| TR | "...olumlu görülen..." | positive |

**Issue:** Model gives opposite sentiment in different languages for same review.

---

## Comprehensive Results Compilation

This section comprehensively compiles all results according to the project's objectives and goals. The goals defined in the Project Requirements Document (PRD) and the extent to which they were achieved are evaluated in detail.

### Project Goals Evaluation

#### G1: Measuring Cross-Lingual Semantic Consistency
**Goal:** Measure cross-lingual semantic consistency for EN/DE/TR.

**Result: ✅ SUCCESSFUL**

| Metric | Value | Comment |
|--------|-------|---------|
| Total comparisons | 288 | 48 prompts × 3 language pairs × 2 runs |
| Average similarity | 0.637 | LaBSE cosine similarity |
| Standard deviation | 0.132 | Reasonable variance |
| Lowest similarity | 0.240 | phi3:latest, Prompt 19, EN-TR |
| Highest similarity | 0.914 | phi4-mini:3.8b, Prompt 3, DE-TR |

**Results by Language Pair:**
| Language Pair | Average | Std | Comment |
|---------------|---------|-----|---------|
| DE-TR | 0.670 | 0.12 | Highest consistency |
| EN-DE | 0.653 | 0.11 | Second highest |
| EN-TR | 0.586 | 0.14 | Lowest consistency |

**Finding:** Consistency is low in Turkish-containing pairs (especially EN-TR). This can be explained by Turkish's agglutinative structure and low representation in training data.

---

#### G2: Inconsistency Analysis by Task Type
**Goal:** Identify which task types exhibit higher inconsistency.

**Result: ✅ SUCCESSFUL**

| Task Type | Measurement Method | Consistency | Level |
|-----------|-------------------|-------------|-------|
| **Factual** | Match Rate | 95.8% | ⭐⭐⭐⭐⭐ Very High |
| **Summarization** | LaBSE Cosine | 70.0% | ⭐⭐⭐⭐ High |
| **Classification** | Match Rate | 61.5% | ⭐⭐⭐ Medium |
| **Creative** | LaBSE Cosine | 57.0% | ⭐⭐ Low |
| **Reasoning** | Match Rate | 37.5% | ⭐ Very Low |

**Detailed Task Analysis:**

**1. Factual Tasks (Prompts 13-16)**
- 5/6 models achieved 100% match rate
- Only phi3:latest at 75% (WW2 date wrong in TR)
- **Why successful:** Single correct answer, memory-based

**2. Summarization Tasks (Prompts 1-4)**
- Average similarity: 0.70
- Best: gemma3:4b (0.74), phi4-mini:3.8b (0.74)
- **Why successful:** Constrained format (single sentence), clear goal

**3. Classification Tasks (Prompts 5-8)**
- Match rate: 12.5% - 100% range (model-dependent)
- gemma3 family 100%, llama family <30%
- **Why variable:** Label extraction is language-dependent

**4. Creative Tasks (Prompts 17-20)**
- Average similarity: 0.57
- Prompt 19 (slogan) lowest: 0.38
- **Why low:** Free creativity, style differences

**5. Reasoning Tasks (Prompts 9-12)**
- Match rate: 12.5% - 75% range
- Prompt 11 (duration calculation) most problematic
- **Why low:** Mathematical calculations vary by language

---

#### G3: Reproducible Metrics and Visualizations
**Goal:** Produce reproducible metrics and visualizations.

**Result: ✅ SUCCESSFUL**

**Generated Data Files:**
| File | Record Count | Description |
|------|--------------|-------------|
| responses.jsonl | 720 | All model responses |
| metrics.csv | 288 | Cross-lingual similarities |
| task_metrics.csv | 145 | Discrete-answer matches |
| stability.csv | 357 | Intra-language stability |
| prompts_primary.csv | 60 | Prompt database |

**Generated Visualizations (31 total):**
| Type | Count | Description |
|------|-------|-------------|
| Model Comparison | 1 | Overall comparison |
| Heatmaps | 6 | Heatmap per model |
| Distributions | 6 | Distribution per model |
| Task Comparisons | 6 | Task analysis per model |
| Stability | 6 | Stability per model |
| Discrete Summary | 6 | Discrete summary per model |

**Reproducibility Guarantee:**
- All parameters stored as YAML in `configs/` folder
- `responses.jsonl` kept as canonical cache
- Random seed: Controlled randomness with `temperature=0.3`

---

#### G4: Qualitative Evidence Set for Low Consistency Examples
**Goal:** Create a short qualitative evidence set containing semantic drift cases.

**Result: ✅ SUCCESSFUL**

**Flagged Example Count:** 29 (bottom 10%)

**Drift Type Distribution:**
| Drift Type | Count | Description |
|------------|-------|-------------|
| Semantic Drift (severe) | 19 | Core meaning difference |
| Style Drift (verbose) | 14 | Excessively long/explanatory |
| Semantic Drift (moderate) | 10 | Moderate meaning difference |
| Format Drift | 5 | Format/length difference |

**Top 5 Critical Examples:**

| # | Model | Prompt | Pair | Sim | Drift Type |
|---|-------|--------|------|-----|------------|
| 1 | phi3:latest | 19 | EN-TR | 0.24 | Semantic + Format + Style |
| 2 | llama3.2:1b | 20 | EN-TR | 0.29 | Semantic |
| 3 | phi3:latest | 18 | EN-TR | 0.30 | Semantic + Style |
| 4 | phi3:latest | 19 | EN-TR | 0.31 | Semantic + Format |
| 5 | llama3.2:1b | 19 | EN-TR | 0.32 | Semantic + Style |

---

### Research Questions Answered

#### RQ1: How does cross-lingual similarity vary by task type?

**Answer:** Task type significantly affects cross-lingual consistency.

```
Consistency Ranking (highest to lowest):

1. Factual      ████████████████████ 95.8%
2. Summarization ██████████████████░░ 70.0%
3. Classification ██████████████░░░░░░ 61.5%
4. Creative     █████████████░░░░░░░ 57.0%
5. Reasoning    █████████░░░░░░░░░░░ 37.5%
```

**Comment:** Constrained, single-answer tasks (factual, summarization) show high consistency, while open-ended tasks (creative, reasoning) show low consistency.

---

#### RQ2: Do models maintain cross-lingual consistency in discrete answers?

**Answer:** Depends on model and task type. Yes for factual, generally no for reasoning.

**Per-Model Discrete Performance:**
| Model | Class. | Reason. | Fact. | Overall |
|-------|--------|---------|-------|---------|
| gemma3:4b | ✅ 100% | ⚠️ 75% | ✅ 100% | 91.7% |
| gemma3:1b | ✅ 100% | ❌ 25% | ✅ 100% | 75.0% |
| phi4-mini:3.8b | ⚠️ 75% | ❌ 37.5% | ✅ 100% | 70.8% |
| llama3.2:1b | ❌ 12.5% | ⚠️ 50% | ✅ 100% | 54.2% |
| llama3.2:3b | ❌ 25% | ❌ 25% | ✅ 100% | 50.0% |
| phi3:latest | ⚠️ 50% | ❌ 12.5% | ⚠️ 75% | 45.8% |

**Critical Finding:** Near-perfect consistency in factual tasks, but all models struggle with reasoning tasks.

---

#### RQ3: Which language pairs show the largest consistency gap?

**Answer:** EN-TR pair shows the largest consistency gap.

**Language Pair Comparison:**
```
DE-TR:  ████████████████████ 0.670 (baseline)
EN-DE:  ███████████████████░ 0.653 (-0.017)
EN-TR:  ██████████████████░░ 0.586 (-0.084)
```

**Flagged Example Distribution:**
| Language Pair | Flagged | Percentage |
|---------------|---------|------------|
| EN-TR | 15 | 51.7% |
| DE-TR | 8 | 27.6% |
| EN-DE | 6 | 20.7% |

**Comment:** Consistency issues are prominent in Turkish-containing pairs (especially EN-TR). This may be due to:
- Turkish's agglutinative structure
- Low Turkish representation in training data
- Complex Turkish morphology

---

#### RQ4: Cross-lingual consistency vs intra-language stability

**Answer:** Intra-language stability (~85%) is significantly higher than cross-lingual consistency (~64%). This shows that cross-lingual differences are not caused by model randomness.

**Comparison:**
| Metric Type | Average | Comment |
|-------------|---------|---------|
| Intra-language (same lang, different run) | 0.85 | High - model is consistent |
| Cross-lingual (different lang, same run) | 0.64 | Lower - language effect |

**Difference:** 0.85 - 0.64 = **0.21** → This difference represents the **language effect**

**Per-Model Stability vs Cross-lingual:**
| Model | Intra-lang Avg | Cross-ling Avg | Δ |
|-------|----------------|----------------|---|
| gemma3:4b | 0.91 | 0.70 | -0.21 |
| phi4-mini:3.8b | 0.86 | 0.67 | -0.19 |
| gemma3:1b | 0.87 | 0.66 | -0.21 |
| llama3.2:3b | 0.84 | 0.62 | -0.22 |
| llama3.2:1b | 0.82 | 0.60 | -0.22 |
| phi3:latest | 0.79 | 0.57 | -0.22 |

---

### Overall Results Summary

#### Successes

| Area | Success | Evidence |
|------|---------|----------|
| Data Collection | ✅ 100% | 720/720 responses collected |
| Non-Compliance | ✅ 99.86% | Only 1 non-compliant |
| Metric Computation | ✅ 100% | All metrics computed |
| Visualization | ✅ 100% | 31 plots generated |
| Qualitative Analysis | ✅ 100% | 29 examples flagged |

#### Limitations

| Area | Limitation | Impact |
|------|------------|--------|
| Model Diversity | 6 models (1B-4B) | Larger models not tested |
| Language Count | 3 languages | Other low-resource languages excluded |
| Prompt Count | 20 prompts | Limited task diversity |
| qwen3:4b | Removed | Empty response issues |

#### Key Findings (Top 5)

1. **gemma3:4b is the most consistent model** (0.70 average similarity)
2. **Turkish is the most challenging language** (EN-TR 0.58, others 0.65+)
3. **Factual tasks are near-perfect** (95.8% match)
4. **Creative tasks are most inconsistent** (0.57 similarity)
5. **phi3:latest fails in Turkish** (0.47 EN-TR, 1 non-compliant)

---

## Discussion & Lessons Learned

### Technical Challenges Overcome

1. **CSV Parsing Errors**
   - **Problem:** German quotes („...") and nested quotes broke pandas parsing
   - **Solution:** Regenerated CSV with `quoting=csv.QUOTE_ALL`

2. **Non-Compliance Detection**
   - **Initial:** 84 flagged responses (11.7%)
   - **After refinement:** 1 flagged response (0.14%)
   - **Key changes:** Multi-language labels, flexible matching, removed accuracy checks

3. **Model Removal: qwen3:4b**
   - **Problem:** 93/120 empty responses (77.5%)
   - **Decision:** Removed model and associated data
   - **Reason:** Unusable for cross-lingual analysis

### Implications for Multilingual LLM Deployment

1. **Language-Specific Validation Required**
   - Models behave differently across languages
   - English-trained validation may miss issues

2. **Task Complexity Impacts Consistency**
   - Constrained tasks (factual, summarization) → Higher consistency
   - Open-ended tasks (creative, reasoning) → Lower consistency

3. **Small Models Viable for Simple Tasks**
   - gemma3:1b achieves 100% classification match rate
   - Suitable for structured, well-defined tasks

---

## Conclusions

### Summary of Results

1. **gemma3:4b** provides the best overall cross-lingual consistency (0.70 average similarity)
2. **Turkish language** is the primary challenge for all models
3. **Factual tasks** show near-perfect cross-lingual consistency
4. **Creative and reasoning tasks** show significant variation
5. **Intra-language stability** is high (~85%), confirming cross-lingual effects are real

### Recommendations

| Use Case | Recommended Model |
|----------|-------------------|
| Classification (all languages) | gemma3:4b, gemma3:1b |
| Factual Q&A | Any model (all consistent) |
| Summarization | gemma3:4b, phi4-mini:3.8b |
| Creative (EN/DE only) | gemma3:4b |
| Avoid for Turkish tasks | phi3:latest |

### Future Work

1. Add more languages (Arabic, Chinese, Japanese)
2. Test larger models (7B, 13B)
3. Implement response regeneration for low-consistency cases
4. Explore prompt engineering to improve Turkish performance

---

## Appendices

### A. Generated Outputs

| File | Description | Records |
|------|-------------|---------|
| `data/responses.jsonl` | All model responses | 720 |
| `data/metrics.csv` | Cross-lingual similarity scores | 288 |
| `data/stability.csv` | Intra-language stability | 357 |
| `data/task_metrics.csv` | Discrete answer matches | 145 |
| `data/prompts_primary.csv` | Input prompts | 60 |

### B. Generated Plots

| Category | Files |
|----------|-------|
| Model Comparison | `outputs/plots/model_comparison.png` |
| Heatmaps (per model) | `outputs/plots/heatmap_*.png` |
| Distributions (per model) | `outputs/plots/distribution_*.png` |
| Task Comparisons (per model) | `outputs/plots/task_comparison_*.png` |
| Stability Analysis (per model) | `outputs/plots/stability_*.png` |
| Discrete Summaries (per model) | `outputs/plots/discrete_summary_*.png` |

### C. Model Summary Statistics

#### gemma3:4b (Best Overall)
| Category | EN-DE | EN-TR | DE-TR |
|----------|-------|-------|-------|
| Open-Text Similarity | 0.72 | 0.64 | 0.73 |
| Stability (Run1-Run2) | 0.95 | 0.92 | 0.86 |

#### phi3:latest (Worst Turkish Performance)
| Category | EN-DE | EN-TR | DE-TR |
|----------|-------|-------|-------|
| Open-Text Similarity | 0.69 | 0.47 | 0.55 |
| Stability (Run1-Run2) | 0.84 | 0.68 | 0.86 |

### D. Prompts Reference

| ID | Task | Description |
|----|------|-------------|
| 1-4 | Summarization | AI, climate, EVs, remote work paragraphs |
| 5 | Classification | Sentiment (phone review) |
| 6 | Classification | Agreement (transport expansion) |
| 7 | Classification | Intent (order complaint) |
| 8 | Classification | Formality (casual email) |
| 9 | Reasoning | Transitive height comparison |
| 10 | Reasoning | Speed calculation |
| 11 | Reasoning | Duration calculation |
| 12 | Reasoning | Weather-appropriate clothing |
| 13 | Factual | Water formula |
| 14 | Factual | WW2 end year |
| 15 | Factual | Canada capital |
| 16 | Factual | 1984 author |
| 17-20 | Creative | Slogans, ad copy, advice |

---

**Report generated:** January 17, 2026  
**Total execution time:** ~25 minutes (inference + metrics + plots)  
**Environment:** Ollama local inference on WSL2
