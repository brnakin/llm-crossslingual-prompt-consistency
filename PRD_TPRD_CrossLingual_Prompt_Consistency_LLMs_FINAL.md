# Evaluating Cross-Lingual Prompt Consistency in Large Language Models
**Version:** 2.0 (Post-Implementation)  
**Date:** 17 January 2026  
**Owner:** Baran Akin, Mehdi Farzin  
**Supervisor:** Shan Faiz  
**Decision Log:** Local-first execution (Ollama) to avoid request limits; Open-source LLMs only (no commercial baseline); **Final Models:** `gemma3:1b`, `gemma3:4b`, `llama3.2:1b`, `llama3.2:3b`, `phi3:latest`, `phi4-mini:3.8b`; LaBSE embeddings for open-text tasks only; Discrete-answer tasks evaluated via exact/normalized match; `temperature = 0.3`; `max_new_tokens = 256`; Two runs per prompt-language; Cross-lingual comparisons are paired by the same `run_id`; Intra-language stability baseline is measured as run1 vs run2 within each language.

---

## IMPLEMENTATION STATUS SUMMARY

> **Bu bölüm, projenin gerçek uygulaması ile orijinal PRD arasındaki farklılıkları belgelemektedir.**

### ✅ Başarıyla Tamamlanan Hedefler

| Hedef | Durum | Notlar |
|-------|-------|--------|
| G1: Cross-lingual tutarlılık ölçümü | ✅ Tamamlandı | 288 karşılaştırma yapıldı |
| G2: Görev türü analizi | ✅ Tamamlandı | 5 görev türü analiz edildi |
| G3: Tekrarlanabilir metrikler | ✅ Tamamlandı | 31 görselleştirme üretildi |
| G4: Niteliksel kanıt seti | ✅ Tamamlandı | 29 örnek flaglendi |

### 📝 Orijinal Plandan Sapmalar

#### 1. Model Listesi Değişiklikleri

**Orijinal Plan:**
- `gemma3:1b`
- `llama3.2:1b`

**Gerçek Uygulama:**
- `gemma3:1b` ✅
- `gemma3:4b` ✅ (eklendi)
- `llama3.2:1b` ✅
- `llama3.2:3b` ✅ (eklendi)
- `phi3:latest` ✅ (eklendi)
- `phi4-mini:3.8b` ✅ (eklendi)
- ~~`qwen3:4b`~~ ❌ (kaldırıldı - boş yanıt sorunu)

**Sebep:** Daha kapsamlı model karşılaştırması için model çeşitliliği artırıldı. qwen3:4b, 93/120 boş yanıt ürettiği için kaldırıldı.

#### 2. Non-Compliance Detection Değişiklikleri

**Orijinal Plan:**
- Basit format kontrolü

**Gerçek Uygulama (iteratif geliştirme):**
1. **İlk versiyon:** Çok katı (84 non-compliant)
2. **Düzeltme 1:** Uzunluk eşiği 100→200
3. **Düzeltme 2:** Çok dilli etiket desteği (DE: positiv/negativ, TR: olumlu/olumsuz)
4. **Düzeltme 3:** A/B/C eşleşme esnekliği
5. **Final versiyon:** Sadece format kontrolü, doğruluk kontrolü kaldırıldı

**Sonuç:** 84 → 1 non-compliant (phi3:latest, TR, Prompt 7)

#### 3. Retry Mekanizması Eklendi

**Orijinal Plan:**
- Retry mekanizması opsiyonel

**Gerçek Uygulama:**
- Non-compliant yanıtlar için stricter prompt wrapper ile retry
- Retry sonrası hala non-compliant olanlar qualitative review için saklandı

#### 4. Ek Görselleştirmeler

**Orijinal Plan:**
- Heatmap + Distribution plot

**Gerçek Uygulama (31 dosya):**
- Model comparison (1)
- Heatmaps per model (6)
- Distributions per model (6)
- Task comparisons per model (6)
- Stability plots per model (6)
- Discrete summaries per model (6)

#### 5. Drift Type Kategorileri Eklendi

**Orijinal Plan:**
- Basit "low similarity" flagging

**Gerçek Uygulama:**
- 5 drift type tanımlandı:
  1. Semantic Drift
  2. Format Drift
  3. Factual Drift
  4. Style Drift
  5. Hallucination

### 📊 Uygulama Metrikleri

| Metrik | Plan | Gerçek | Durum |
|--------|------|--------|-------|
| Toplam yanıt | 120 × M | 720 | ✅ (M=6) |
| Coverage | %100 | %100 | ✅ |
| Non-compliant rate | <%5 | %0.14 | ✅ |
| Görselleştirme | 2+ | 31 | ✅ |
| Flagged örnek | 6+ | 29 | ✅ |

### 🔧 Teknik Zorluklar ve Çözümleri

| Zorluk | Çözüm |
|--------|-------|
| CSV parsing hatası (German quotes) | `quoting=csv.QUOTE_ALL` ile yeniden üretim |
| qwen3:4b boş yanıtları | Model ve verisi kaldırıldı |
| Katı non-compliance tespiti | İteratif refinement, çok dilli destek |
| Türkçe etiket tanıma | olumlu/olumsuz, positiv/negativ eklendi |

---

## 1) Repository Skeleton (proposed)

```text
crosslingual-consistency/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ configs/
│  ├─ models.yaml               # model ids + decoding params
│  └─ embeddings.yaml           # LaBSE config
├─ data/
│  ├─ prompts_primary.csv       # 20 prompts x 3 languages (EN/DE/TR)
│  ├─ responses.jsonl           # generated outputs + metadata (cached)
│  ├─ metrics.csv               # cross-lingual semantic similarity for open-text tasks
│  ├─ stability.csv            # intra-language stability (run1 vs run2) for open-text tasks
│  └─ task_metrics.csv          # task-aware agreement checks (where applicable)
├─ src/
│  ├─ load_prompts.py
│  ├─ infer_ollama.py           # local inference client
│  ├─ embed_labse.py            # LaBSE embedding
│  ├─ similarity.py             # cosine + aggregation
│  ├─ task_checks.py            # classification/reasoning/factual agreement
│  └─ plots.py                  # heatmap + distribution plots
├─ notebooks/
│  ├─ 01_collect_responses.ipynb
│  ├─ 02_metrics_and_plots.ipynb
│  └─ 03_qualitative_review.ipynb
└─ outputs/
   ├─ plots/
   └─ reports/
```

---

## 2) Product Requirements Document (PRD)

### 2.1 Overview
**Problem:** LLMs often deliver strong English outputs, but can produce inconsistent answers in other languages for semantically equivalent prompts. This creates reliability and fairness risks for multilingual users.  
**Solution:** A controlled and reproducible evaluation framework that measures cross-lingual prompt consistency for English, German, and Turkish using (1) LaBSE-based semantic similarity and (2) task-aware agreement checks for discrete-answer tasks.

### 2.2 Goals / Non-Goals
**Goals**
- G1: Quantify cross-lingual semantic consistency across EN/DE/TR for a curated prompt set.
- G2: Identify which task types exhibit higher inconsistency (summarization, classification, reasoning, factual recall, creative).
- G3: Produce reproducible metrics and visualizations suitable for a proposal and an implementation-grade experiment.
- G4: Provide a short qualitative evidence set for semantic drift cases (low-consistency samples).

**Non-Goals**
- N1: Fine-tuning, RLHF, or modifying model weights.
- N2: Large-scale benchmarking across many languages beyond EN/DE/TR.
- N4: Building a production system or user-facing application beyond the evaluation pipeline.

### 2.3 Personas & Use Cases
- **Research Student:** Needs a defensible methodology and reproducible results for a short academic project.
- **ML Engineer:** Wants a lightweight way to compare multilingual stability across open LLMs.
- **Instructor/Reviewer:** Needs transparent assumptions, datasets, and evaluation steps.

Use cases:
- Compare two open-source instruction-tuned models for multilingual consistency.
- Detect which task types produce higher semantic drift for Turkish vs German.
- Generate an interpretable report (plots + flagged examples) from one run.

### 2.4 Scope (what is evaluated)
- Languages: EN, DE, TR
- Models: open-source instruction-tuned LLMs (local-first)
- Prompt dataset: 20 items, 5 task types
- Runs: two runs per prompt-language condition
- Metrics:
  - Primary (open-text tasks: summarization, creative): LaBSE embeddings + cosine similarity (paired EN-DE, EN-TR, DE-TR by the same `run_id`)
  - Primary (discrete-answer tasks: classification, reasoning, factual): exact/normalized match via task-aware checks
  - Secondary (optional): LaBSE cosine for discrete-answer tasks as an exploratory signal only (reported separately)
- Outputs:
  - response archive (JSONL)
  - metrics tables (CSV)
  - plots (heatmap + distributions)
  - small qualitative set (flagged lowest similarity examples)

### 2.5 Success Metrics & KPIs (student project targets)
- **Coverage:** 20 prompts x 3 languages x 2 runs per selected model completed without missing cells.
- **Reproducibility:** Prompts, parameters, and results are logged and re-runnable.
- **Interpretability:** At least:
  - 1 heatmap per model
  - 1 distribution plot per model
  - 1 table summarizing mean/std by language pair and task type
- **Qualitative evidence:** At least 6 flagged cases (lowest similarity) with short annotation of drift type.

### 2.6 Key Decisions

- **LLM access:** local-first via Ollama to avoid API request limits and ensure repeatability.
- **Embedding encoder:** LaBSE only (used as the semantic metric for open-text tasks).
- **Discrete-answer evaluation:** exact/normalized match (task-aware checks) is the primary metric for classification, reasoning, and factual prompts.
- **Decoding:** `temperature = 0.3`, `max_new_tokens = 256`.
- **Runs:** 2 runs per prompt-language.
- **Run pairing (cross-lingual):** compare EN/DE/TR outputs generated with the same `run_id` (paired comparisons: run1-to-run1, run2-to-run2).
- **Stability baseline (intra-language):** compare run1 vs run2 within EN, within DE, and within TR to quantify model randomness independent of translation.
- **Public datasets:** optional extension only (not required for completion).

### 2.7 Risks & Mitigations
- **Translation drift creates fake inconsistency:** Manual translation plus back-translation checks; revise items with meaning shift.
- **Cosine similarity reflects semantics, not correctness (and can be unstable for very short outputs):** Use LaBSE cosine as the primary metric only for open-text tasks. For discrete-answer tasks (classification/reasoning/factual), use exact/normalized match as the primary metric and treat any embedding-based score as exploratory.
- **Length differences distort embeddings:** Limit output length via `max_new_tokens = 256` and concise prompt instructions.
- **Compute constraints for local models:** Keep dataset small; cache outputs in JSONL; prioritize one model if needed.

### 2.8 Acceptance Criteria (PRD)
- A: Primary dataset is available as CSV with prompt_id, task_type, language, and full prompt text.
- B: Response archive contains at least 120 outputs per model (20 prompts x 3 languages x 2 runs).
- B2: `stability.csv` contains intra-language stability results (run1 vs run2) for EN, DE, and TR, for open-text and discrete-answer tasks.
- C: For open-text tasks (summarization, creative), `metrics.csv` contains paired cross-lingual cosine similarities for all prompt ids and language pairs (EN-DE, EN-TR, DE-TR), for both runs.
- D: For discrete-answer tasks (classification, reasoning, factual), `task_metrics.csv` contains paired cross-lingual match/mismatch results and is included in the summary reporting.
- E: Plots are generated and saved (heatmap + distribution, plus a stability context view).
- F: A flagged qualitative set of low-consistency examples is produced for review.

---

## 3) Technical PRD (TPRD)

### 3.1 Architecture (single-pass pipeline)
```
prompts_primary.csv
  -> inference (Ollama local, fixed decoding params)
  -> responses.jsonl (canonical cache)
  -> task routing by task_type
     - open-text (summarization, creative): LaBSE embeddings -> cosine similarity (paired EN-DE, EN-TR, DE-TR by the same run_id)
     - discrete-answer (classification, reasoning, factual): task-aware extraction -> exact/normalized match
  -> intra-language stability baseline
     - open-text: LaBSE cosine for run1 vs run2 within EN/DE/TR
     - discrete-answer: match rate for run1 vs run2 within EN/DE/TR
  -> aggregation (by pair, by task type, and vs stability baseline)
  -> plots (heatmap + distributions)
  -> outputs/ (metrics tables + plots + flagged examples)
```

### 3.2 Data Schemas

**Prompt dataset schema (CSV)**
- `prompt_id` (int)
- `task_type` (summarization | classification | reasoning | factual | creative)
- `language` (EN | DE | TR)
- `text` (string, full instruction plus input)

**Response dataset schema (JSONL)**
- Note: include `non_compliant` boolean for output-format violations.
```json
{
  "prompt_id": 1,
  "task_type": "summarization",
  "language": "EN",
  "model_id": "llama3.2:1b",
  "run_id": 1,
  "temperature": 0.3,
  "max_new_tokens": 256,
  "timestamp_utc": "2026-01-17T12:00:00Z",
  "prompt_text": "...",
  "response_text": "...",
  "non_compliant": false
}
```

**Metrics schema (CSV: `metrics.csv`, open-text only)**
- `model_id`
- `prompt_id`
- `task_type` (summarization | creative)
- `pair` (EN-DE | EN-TR | DE-TR)
- `run_id` (paired cross-lingual comparison: run1-to-run1, run2-to-run2)
- `cosine_similarity`
- `flag_low_similarity` (bool, bottom 10% within model)

**Stability schema (CSV: `stability.csv`)**
- `model_id`
- `prompt_id`
- `task_type`
- `language` (EN | DE | TR)
- `stability_type` (open_text_cosine | discrete_match)
- `run_id_a` (1)
- `run_id_b` (2)
- `stability_value` (cosine similarity for open-text; 1/0 match for discrete-answer tasks)

**Task-aware metrics schema (CSV)**
- `model_id`
- `prompt_id`
- `task_type`
- `check_type` (classification | reasoning | factual)
- `result` (match | mismatch | uncertain)
- `key_en` / `key_de` / `key_tr` (extracted where applicable)

### 3.3 Model Inference Requirements (Ollama)
**Response Language Policy**
- For EN prompts: respond in English.
- For DE prompts: respond in German.
- For TR prompts: respond in Turkish.
- For tasks that require fixed tokens (e.g., classification labels, A/B/C multiple-choice, numeric-only answers): the output format requirement takes precedence; extra commentary is not allowed.

**Prompt Wrapper (applied programmatically per request)**
Prepend the following control line to every prompt:
- EN: `Respond in English. Follow the output format exactly. Do not add explanations.`
- DE: `Antworte auf Deutsch. Halte dich exakt an das Ausgabeformat. Keine Erklärungen.`
- TR: `Türkçe yanıt ver. Çıktı formatına tam uy. Açıklama ekleme.`

**Ollama parameter mapping**
- `max_new_tokens` corresponds to `num_predict` in Ollama.
- Keep other sampling parameters at defaults unless explicitly set; log the effective parameters used.

**Models (final implementation, Ollama tags)**
- `gemma3:1b` - Google, 1B parameters
- `gemma3:4b` - Google, 4B parameters
- `llama3.2:1b` - Meta, 1B parameters
- `llama3.2:3b` - Meta, 3B parameters
- `phi3:latest` - Microsoft, 3.8B parameters
- `phi4-mini:3.8b` - Microsoft, 3.8B parameters
- ~~`qwen3:4b`~~ - Removed (empty response issues)

**Fixed decoding parameters**
- `temperature = 0.3`
- `max_new_tokens = 256`
- Two runs per prompt-language, stored as separate `run_id` records.

**Request volume**
For M models, total inference calls:
- Calls = 20 prompts x 3 languages x 2 runs x M
- Calls = 120 x M

**Actual implementation (M=6):**
- Total calls = 120 x 6 = **720 responses**
- Inference time: ~8 minutes (local Ollama)

Optional public dataset extension adds:
- Calls_additional = N x 3 x 2 x M = 6N x M
Where N is the number of added public items (EN/DE/TR triplets).

**Note:** qwen3:4b was initially included (M=7) but removed after 93/120 empty responses.

### 3.4 Embedding and Similarity (LaBSE-only, open-text tasks)
**Applicability**
- Use LaBSE cosine as the semantic metric only for open-text outputs (summarization, creative).
- For discrete-answer tasks (classification, reasoning, factual), treat LaBSE cosine as exploratory only and do not mix it into the main aggregate.

**Encoder**
- `sentence-transformers/LaBSE`

**Cross-lingual computation (paired by run_id)**
- Embed each open-text response as a dense vector.
- Compute cosine similarity for each prompt and each language pair using the same `run_id` across languages:
  - run1: EN_run1 vs DE_run1, EN_run1 vs TR_run1, DE_run1 vs TR_run1
  - run2: EN_run2 vs DE_run2, EN_run2 vs TR_run2, DE_run2 vs TR_run2

**Intra-language stability baseline**
- For open-text tasks, compute cosine similarity of run1 vs run2 within each language (EN, DE, TR) and store it in `stability.csv`.

**Aggregation**
- Mean and standard deviation by language pair (cross-lingual).
- Mean and standard deviation by task type (open-text only).
- Compare cross-lingual consistency against the intra-language stability baseline to separate language effects from sampling randomness.

**Flagging**
- Mark the bottom 10% similarity values per model as `flag_low_similarity = True` for qualitative inspection.

### 3.5 Task-aware checks (primary for discrete-answer tasks)
These checks are the primary evaluation signal for classification, reasoning, and factual prompts, because outputs are intentionally short and format-constrained (e.g., `A/B/C`, years, formulas), where embedding cosine can be unstable.

**Classification prompts**
- Extract predicted label (positive/negative, agreement/disagreement, request/complaint, formal/informal).
- Cross-lingual: check whether EN/DE/TR labels match for the same `run_id`.
- Intra-language stability: check whether run1 label equals run2 label within each language; store 1/0 in `stability.csv` with `stability_type = discrete_match`.

**Reasoning prompts**
- Extract numeric or deterministic answer (speed, duration, ordering).
- Cross-lingual: check whether answers match across languages for the same `run_id`.
- Intra-language stability: run1 vs run2 within each language (1/0).

**Factual prompts**
- Extract key entity (inventor, year, capital, author).
- Cross-lingual: check whether extracted entities match across languages for the same `run_id`.
- Intra-language stability: run1 vs run2 within each language (1/0).

**Outputs**
- Write cross-lingual match/mismatch results to `task_metrics.csv`.
- Write intra-language stability results (1/0 for discrete tasks) to `stability.csv`.

### 3.6 Visualizations (required)
- **Heatmap:** rows prompt_id, columns language pairs, values cosine similarity.
- **Distribution plot:** boxplot or density per language pair.
- **Summary table:** mean/std by pair and by task type (open-text), plus discrete match rates (classification/reasoning/factual).
- **Stability table/plot:** run1 vs run2 stability by language (EN/DE/TR) to contextualize cross-lingual results.

### 3.7 Reproducibility and Logging
- Log: model_id, decoding params, prompt_id, language, run_id, timestamp.
- Capture environment versions: Ollama version, model tag + digest, Python version, and `sentence-transformers` version.
- `responses.jsonl` is the canonical cache and should be treated as immutable once generated.
- Store config files used for the run (`models.yaml`, `embeddings.yaml`) alongside outputs.

**Output compliance handling**
- Validate each response against the expected output format.
- If a response violates constraints (e.g., adds explanations when “output only” is required):
  - Record `non_compliant = true` in the response metadata.
  - Optionally re-run the same prompt once with the same parameters and a stricter wrapper line.
  - If still non-compliant, keep the original and exclude from exact-match task checks (but keep for qualitative review).


### 3.8 One-shot Execution Checklist (no sprints)
1. Finalize prompts and translations. Run back-translation checks and revise if needed.
2. Generate responses for all models and runs. Save to `responses.jsonl`.
3. Compute intra-language stability baselines (run1 vs run2) and save to `stability.csv`.
4. Compute LaBSE embeddings and paired cross-lingual cosine similarities for open-text tasks. Save to `metrics.csv`.
5. Compute task-aware cross-lingual checks for discrete-answer tasks. Save to `task_metrics.csv`.
6. Generate plots and summary tables (including stability context). Save to `outputs/plots/` and `outputs/reports/`.
7. Create a flagged qualitative set from the lowest similarity cases.

### 3.9 Acceptance (TPRD)
- `/data/prompts_primary.csv` exists with 20 items x 3 languages.
- `responses.jsonl` contains 120 x M records, with correct metadata and no missing languages.
- `stability.csv` includes intra-language stability results (run1 vs run2) for EN/DE/TR.
- `metrics.csv` includes paired cross-lingual cosine scores for open-text tasks (summarization/creative) and all language pairs.
- `task_metrics.csv` includes task-aware checks for applicable prompts.
- Plots are generated and saved.
- Flagged examples list exists and can be reviewed.

---

## 10) Timeline (Milestones)

**M1 — Dataset readiness**
- Finalize the 20-item primary prompt set (EN source).
- Manual translation to DE and TR.
- Back-translation validation and revision of any drifting items.
- Output: `data/prompts_primary.csv` (complete EN/DE/TR triplets).

**M2 — Inference run completed (local-first)**
- Configure Ollama models and fixed decoding parameters (`temperature=0.3`, `max_new_tokens=256`).
- Execute two runs per prompt-language per model.
- Output: `data/responses.jsonl` (canonical cache, complete coverage).

**M3 — Stability baseline + semantic similarity metrics**
- Compute intra-language stability (run1 vs run2) within EN, DE, TR:
  - open-text: LaBSE cosine
  - discrete-answer: match rate (1/0)
- Compute LaBSE embeddings for open-text responses.
- Calculate paired cross-lingual cosine similarity for EN–DE, EN–TR, and DE–TR (same run_id across languages).
- Aggregate mean/std by language pair and task type (open-text), and report discrete match rates separately.
- Outputs: `data/stability.csv`, `data/metrics.csv`.

**M4 — Task-aware cross-lingual checks completed (discrete-answer tasks)**
- Run lightweight agreement checks for discrete-answer prompts (paired by run_id):
  - classification label agreement
  - reasoning numeric/deterministic agreement
  - factual key-entity agreement
- Output: `data/task_metrics.csv`.

**M5 — Visualization and review package**
- Generate plots: heatmap + distribution plot per model.
- Produce a flagged set of lowest-similarity cases for manual inspection.
- Output: `outputs/plots/*` + `outputs/reports/*`.

---

## Appendix A) Primary Prompt and Input Set (20 items, EN/DE/TR)

### Summarization (1 to 4)

**1.**  
- EN: Summarize the following paragraph in one sentence: “Artificial intelligence is changing many industries by automating repetitive work and helping people make faster decisions. Companies use AI to detect patterns in large datasets, but the results depend on data quality and careful evaluation. While productivity can increase, some tasks may be replaced and employees may need reskilling. Clear policies are also needed to reduce privacy risks and unfair outcomes.”  
- DE: Fasse den folgenden Absatz in einem Satz zusammen: „Künstliche Intelligenz verändert viele Branchen, indem sie repetitive Arbeit automatisiert und Menschen hilft, schneller Entscheidungen zu treffen. Unternehmen nutzen KI, um Muster in großen Datensätzen zu erkennen, aber die Ergebnisse hängen von Datenqualität und sorgfältiger Bewertung ab. Obwohl die Produktivität steigen kann, können einige Tätigkeiten ersetzt werden und Beschäftigte benötigen möglicherweise Umschulungen. Außerdem sind klare Richtlinien nötig, um Datenschutzrisiken und unfairen Ergebnissen vorzubeugen.“  
- TR: Aşağıdaki paragrafı tek cümlede özetle: “Yapay zeka, tekrarlayan işleri otomatikleştirerek ve daha hızlı karar vermeye yardımcı olarak birçok sektörü dönüştürüyor. Şirketler büyük veri kümelerinde örüntüleri yakalamak için yapay zekayı kullanıyor, ancak sonuçlar veri kalitesine ve dikkatli değerlendirmeye bağlıdır. Verimlilik artabilse de bazı görevler ortadan kalkabilir ve çalışanların yeniden beceri kazanması gerekebilir. Ayrıca gizlilik risklerini ve adaletsiz sonuçları azaltmak için net politikalar gerekir.”

**2.**  
- EN: Summarize the following text in one sentence: “Climate change is contributing to rising sea levels and more frequent extreme weather events in many regions. Coastal cities face higher flood risk, and storms can damage infrastructure and disrupt supply chains. Adaptation measures such as flood barriers and heat plans can reduce harm, but they require long-term funding. Reducing emissions remains essential to limit future impacts.”  
- DE: Fasse diesen Text in einem Satz zusammen: „Der Klimawandel trägt in vielen Regionen zu steigenden Meeresspiegeln und häufigeren Extremwetterereignissen bei. Küstenstädte sind stärker von Überschwemmungen bedroht, und Stürme können Infrastruktur beschädigen und Lieferketten stören. Anpassungsmaßnahmen wie Hochwasserschutz und Hitzeschutzpläne können Schäden verringern, erfordern jedoch langfristige Finanzierung. Gleichzeitig bleibt die Reduktion von Emissionen entscheidend, um zukünftige Folgen zu begrenzen.“  
- TR: Bu metni tek cümlede özetle: “İklim değişikliği, birçok bölgede deniz seviyelerinin yükselmesine ve aşırı hava olaylarının daha sık görülmesine katkıda bulunuyor. Kıyı şehirlerinde sel riski artıyor ve fırtınalar altyapıya zarar vererek tedarik zincirlerini aksatabiliyor. Setler ve sıcak hava planları gibi uyum önlemleri zararı azaltabilir, ancak uzun vadeli finansman gerektirir. Gelecekteki etkileri sınırlamak için emisyonların azaltılması hâlâ kritik önemdedir.”

**3.**  
- EN: Summarize the following paragraph in one sentence: “Electric vehicles can reduce local air pollution because they have no tailpipe emissions, but their total climate benefit depends on how electricity is generated. Battery production requires mining and energy-intensive processing, which can create environmental impacts. Recycling and longer battery lifetimes can improve sustainability over time. Policy and infrastructure choices strongly influence whether the overall system becomes cleaner.”  
- DE: Fasse den folgenden Absatz in einem Satz zusammen: „Elektrofahrzeuge können die lokale Luftverschmutzung reduzieren, da sie keine Auspuffemissionen haben, aber ihr gesamter Klimavorteil hängt davon ab, wie Strom erzeugt wird. Die Batterieproduktion erfordert Rohstoffabbau und energieintensive Verarbeitung, was Umweltbelastungen verursachen kann. Recycling und längere Batterielebensdauern können die Nachhaltigkeit mit der Zeit verbessern. Politische Entscheidungen und Infrastruktur beeinflussen stark, ob das Gesamtsystem sauberer wird.“  
- TR: Aşağıdaki paragrafı tek cümlede özetle: “Elektrikli araçlar egzoz emisyonu üretmedikleri için yerel hava kirliliğini azaltabilir, ancak toplam iklim faydası elektriğin nasıl üretildiğine bağlıdır. Pil üretimi madencilik ve enerji yoğun işlemler gerektirir ve bu da çevresel etkiler yaratabilir. Geri dönüşüm ve pillerin daha uzun ömürlü olması zamanla sürdürülebilirliği artırabilir. Politika ve altyapı tercihleri, sistemin genel olarak daha temiz hale gelip gelmeyeceğini güçlü biçimde belirler.”

**4.**  
- EN: Summarize the following paragraph in one sentence: “Remote work can increase flexibility and reduce commuting time, which may improve well-being for some employees. However, it can also reduce informal collaboration and make onboarding harder for new team members. Communication often becomes more scheduled and documentation-heavy, which changes how teams coordinate. Clear boundaries and hybrid policies can help prevent burnout and maintain productivity.”  
- DE: Fasse den folgenden Absatz in einem Satz zusammen: „Remote-Arbeit kann die Flexibilität erhöhen und Pendelzeiten reduzieren, was das Wohlbefinden mancher Beschäftigter verbessern kann. Gleichzeitig kann sie informelle Zusammenarbeit verringern und das Onboarding neuer Teammitglieder erschweren. Kommunikation wird häufig stärker geplant und dokumentationslastiger, wodurch sich die Zusammenarbeit verändert. Klare Grenzen und Hybrid-Regelungen können helfen, Burnout zu vermeiden und die Produktivität zu sichern.“  
- TR: Aşağıdaki paragrafı tek cümlede özetle: “Uzaktan çalışma esnekliği artırabilir ve işe gidiş geliş süresini azaltarak bazı çalışanların iyi oluşunu destekleyebilir. Ancak gayriresmi iş birliğini azaltabilir ve yeni ekip üyelerinin işe uyumunu zorlaştırabilir. İletişim daha planlı ve dokümantasyon ağırlıklı hale gelir, bu da ekiplerin koordinasyon biçimini değiştirir. Net sınırlar ve hibrit politikalar, tükenmişliği önleyip verimliliği korumaya yardımcı olabilir.”

### Classification (5 to 8)

**5. Sentiment (output must be exactly: Positive or Negative)**  
- EN: Label the sentiment as Positive or Negative: “After a week of use, the phone overheats during video calls, the battery drops from 100% to 20% by afternoon, and the screen shows random flickering. I regret buying it.”  
- DE: Ordne die Stimmung als Positiv oder Negativ ein: „Nach einer Woche Nutzung wird das Handy bei Videoanrufen sehr heiß, der Akku fällt bis zum Nachmittag von 100% auf 20%, und der Bildschirm flackert zufällig. Ich bereue den Kauf.“  
- TR: Duyguyu Olumlu veya Olumsuz olarak etiketle: “Bir haftalık kullanımın ardından telefon görüntülü aramalarda aşırı ısınıyor, pil öğleden sonra 100%’den 20%’ye düşüyor ve ekran rastgele titriyor. Aldığıma pişmanım.”

**6. Agreement (output must be exactly: Agree or Disagree)**  
- EN: Decide if the response indicates agreement or disagreement. Output only Agree or Disagree: Statement: “Public transport should be expanded to reduce traffic.” Response: “I see your point, but I don’t think it would solve congestion in this city.”  
- DE: Entscheide, ob die Antwort Zustimmung oder Ablehnung ausdrückt. Gib nur Agree oder Disagree aus: Aussage: „Der öffentliche Nahverkehr sollte ausgebaut werden, um den Verkehr zu reduzieren.“ Antwort: „Ich verstehe deinen Punkt, aber ich glaube nicht, dass das hier die Staus wirklich löst.“  
- TR: Yanıtın katılım mı karşı çıkma mı olduğunu belirle. Sadece Agree veya Disagree yaz: İfade: “Trafiği azaltmak için toplu taşıma genişletilmeli.” Yanıt: “Söylediğini anlıyorum ama bunun bu şehirde sıkışıklığı çözeceğini düşünmüyorum.”

**7. Intent (output must be exactly: Request or Complaint)**  
- EN: Classify the primary intent as Request or Complaint. Output only Request or Complaint: “My order arrived with a cracked screen protector and missing accessories. This is unacceptable. Please issue a refund or send a replacement.”  
- DE: Ordne die Hauptabsicht als Request oder Complaint ein. Gib nur Request oder Complaint aus: „Meine Bestellung kam mit einem gesprungenen Displayschutz und fehlendem Zubehör an. Das ist inakzeptabel. Bitte erstatten Sie den Betrag oder senden Sie einen Ersatz.“  
- TR: Ana niyeti Request veya Complaint olarak sınıflandır. Sadece Request veya Complaint yaz: “Siparişim ekran koruyucusu çatlak ve aksesuarları eksik şekilde geldi. Bu kabul edilemez. Lütfen para iadesi yapın veya yenisini gönderin.”

**8. Formality (output must be exactly: Formal or Informal)**  
- EN: Label the message as Formal or Informal. Output only Formal or Informal: “Hi Alex, could you send me the updated document when you have a moment? Thanks!”  
- DE: Markiere die Nachricht als Formal oder Informal. Gib nur Formal oder Informal aus: „Hi Alex, könntest du mir das aktualisierte Dokument schicken, wenn du kurz Zeit hast? Danke!“  
- TR: Mesajı Formal veya Informal olarak etiketle. Sadece Formal veya Informal yaz: “Selam Alex, müsait olduğunda güncellenmiş dokümanı bana gönderebilir misin? Teşekkürler!”

### Reasoning (9 to 12)

**9. Logic (output must be exactly: A, B, or C)**  
- EN: If A is taller than B and B is taller than C, who is the tallest? Output only A, B, or C.  
- DE: Wenn A größer als B und B größer als C ist, wer ist am größten? Gib nur A, B oder C aus.  
- TR: A, B’den ve B de C’den uzunsa, en uzun kimdir? Sadece A, B veya C yaz.

**10. Math (output must be a single number with unit: km/h)**  
- EN: A car travels 90 kilometers in 1.5 hours. What is its average speed? Output only the final value in km/h.  
- DE: Ein Auto fährt 90 Kilometer in 1,5 Stunden. Wie hoch ist die Durchschnittsgeschwindigkeit? Gib nur den Endwert in km/h aus.  
- TR: Bir araba 1,5 saatte 90 kilometre gidiyor. Ortalama hızı nedir? Sadece km/h cinsinden nihai değeri yaz.

**11. Time (output must be a single value in minutes)**  
- EN: A meeting started at 9:15 and ended at 10:45. How many minutes did it last? Output only the number of minutes.  
- DE: Ein Meeting begann um 9:15 und endete um 10:45. Wie viele Minuten dauerte es? Gib nur die Minutenanzahl aus.  
- TR: Toplantı 9:15’te başlayıp 10:45’te bitti. Kaç dakika sürdü? Sadece dakika sayısını yaz.

**12. Decision (multiple-choice; output must be exactly: A, B, or C)**  
- EN: It is 2°C and raining. Which option is most appropriate to wear? A) T-shirt and sandals B) Warm waterproof jacket and closed shoes C) Shorts and sunglasses. Output only A, B, or C.  
- DE: Es sind 2°C und es regnet. Welche Option ist am geeignetsten? A) T-Shirt und Sandalen B) Warme wasserdichte Jacke und geschlossene Schuhe C) Shorts und Sonnenbrille. Gib nur A, B oder C aus.  
- TR: Hava 2°C ve yağmur yağıyor. Hangisini giymek en uygundur? A) Tişört ve sandalet B) Sıcak su geçirmez mont ve kapalı ayakkabı C) Şort ve güneş gözlüğü. Sadece A, B veya C yaz.

### Factual Recall (13 to 16)

**13. (output must be exactly the formula)**  
- EN: What is the chemical formula for water? Output only the formula.  
- DE: Wie lautet die chemische Formel für Wasser? Gib nur die Formel aus.  
- TR: Suyun kimyasal formülü nedir? Sadece formülü yaz.

**14. (output must be exactly the year)**  
- EN: In which year did World War II end? Output only the year.  
- DE: In welchem Jahr endete der Zweite Weltkrieg? Gib nur das Jahr aus.  
- TR: İkinci Dünya Savaşı hangi yılda sona erdi? Sadece yılı yaz.

**15. (output must be exactly the city name)**  
- EN: What is the capital city of Canada? Output only the city name.  
- DE: Was ist die Hauptstadt von Kanada? Gib nur den Städtenamen aus.  
- TR: Kanada’nın başkenti neresidir? Sadece şehir adını yaz.

**16. (output must be exactly the author’s name)**  
- EN: Who wrote the novel “1984”? Output only the author’s name.  
- DE: Wer schrieb den Roman „1984“? Gib nur den Namen des Autors aus.  
- TR: “1984” romanını kim yazdı? Sadece yazarın adını yaz.

### Creative Generation (17 to 20)

**17. Tagline (2 sentences; no bullet points)**  
- EN: Write exactly two sentences promoting teamwork in a workplace. Keep the tone professional. No bullet points.  
- DE: Schreibe genau zwei Sätze, die Teamarbeit am Arbeitsplatz bewerben. Professioneller Ton. Keine Aufzählungspunkte.  
- TR: İş yerinde takım çalışmasını teşvik eden tam olarak iki cümle yaz. Tonu profesyonel olsun. Madde işareti kullanma.

**18. Short ad copy (2 sentences; include one eco benefit + one taste note)**  
- EN: Write exactly two sentences of ad copy for an eco-friendly coffee brand. Include one environmental benefit and one taste description. No bullet points.  
- DE: Schreibe genau zwei Werbesätze für eine umweltfreundliche Kaffeemarke. Nenne einen Umweltvorteil und eine Geschmacksbeschreibung. Keine Aufzählungspunkte.  
- TR: Çevre dostu bir kahve markası için tam olarak iki cümle reklam metni yaz. Bir çevresel fayda ve bir tat açıklaması ekle. Madde işareti kullanma.

**19. Slogan (6 to 8 words; output only the slogan)**  
- EN: Write a slogan for an online education platform. It must be 6 to 8 words. Output only the slogan.  
- DE: Schreibe einen Slogan für eine Online-Bildungsplattform. Er muss aus 6 bis 8 Wörtern bestehen. Gib nur den Slogan aus.  
- TR: Bir çevrimiçi eğitim platformu için slogan yaz. 6 ile 8 kelime arasında olmalı. Sadece sloganı yaz.

**20. One-line advice (one sentence; include a time-management action)**  
- EN: Give one sentence of advice to a student preparing for exams. The sentence must include a concrete time-management action (e.g., plan, schedule, time-block). Output only the sentence.  
- DE: Gib einen Satz Ratschlag für einen Studenten, der sich auf Prüfungen vorbereitet. Der Satz muss eine konkrete Zeitmanagement-Aktion enthalten (z.B. planen, Zeitblöcke erstellen). Gib nur den Satz aus.  
- TR: Sınavlara hazırlanan bir öğrenciye tek cümlelik tavsiye ver. Cümle somut bir zaman yönetimi eylemi içermeli (ör. planla, programla, zaman blokları oluştur). Sadece cümleyi yaz.

---

## 11) Final Results Summary (Post-Implementation)

### Key Findings

| Finding | Value | Significance |
|---------|-------|--------------|
| Best overall model | **gemma3:4b** | 0.70 average cross-lingual similarity |
| Worst overall model | **phi3:latest** | 0.57 average (severe TR issues) |
| Most consistent task | **Factual** | 95.8% cross-lingual match rate |
| Least consistent task | **Reasoning** | 37.5% cross-lingual match rate |
| Hardest language pair | **EN-TR** | 0.586 average similarity |
| Easiest language pair | **DE-TR** | 0.670 average similarity |

### Model Rankings (Cross-Lingual Consistency)

| Rank | Model | Avg Similarity | Discrete Match |
|------|-------|----------------|----------------|
| 1 | gemma3:4b | 0.6969 | 91.7% |
| 2 | phi4-mini:3.8b | 0.6740 | 70.8% |
| 3 | gemma3:1b | 0.6613 | 75.0% |
| 4 | llama3.2:3b | 0.6172 | 50.0% |
| 5 | llama3.2:1b | 0.6046 | 54.2% |
| 6 | phi3:latest | 0.5696 | 45.8% |

### Recommendations

| Use Case | Recommended Model | Avoid |
|----------|-------------------|-------|
| Classification (all languages) | gemma3:4b, gemma3:1b | llama3.2:1b |
| Factual Q&A | Any model | - |
| Summarization | gemma3:4b, phi4-mini:3.8b | phi3:latest |
| Creative tasks | gemma3:4b (EN/DE only) | phi3:latest (TR) |
| Turkish language support | gemma3:4b | phi3:latest |

### Deliverables Produced

| Deliverable | Location | Status |
|-------------|----------|--------|
| Response archive | `data/responses.jsonl` | ✅ 720 records |
| Cross-lingual metrics | `data/metrics.csv` | ✅ 288 records |
| Stability metrics | `data/stability.csv` | ✅ 357 records |
| Discrete metrics | `data/task_metrics.csv` | ✅ 145 records |
| Visualizations | `outputs/plots/` | ✅ 31 files |
| Model summaries | `outputs/reports/` | ✅ 7 files |
| Final report | `REPORT_FINAL.md` | ✅ ~700 lines |

---

**End of Document**

**Document History:**
- v1.0 (17 Jan 2026): Initial PRD/TPRD
- v2.0 (17 Jan 2026): Post-implementation update with actual results and deviations
