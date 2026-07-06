
# Redrob Hackathon – AI Candidate Ranking System.

AI-powered candidate ranking system for Redrob Hackathon 2026.

## Overview

This project is a hybrid AI-based candidate ranking system designed to process **100,000 candidate profiles** and return the **Top 100 most relevant candidates** for a **Senior AI Engineer** role.

The system combines:
* **Semantic Search** using FAISS (HNSW)
* **Keyword Matching** using BM25
* **23 Redrob Behavioral Signals**
* **Skill Ontology-Based Reasoning**
* **Percentile Feature Normalization**
* **Hybrid Score Fusion** with JD-Adaptive Weights

---

## Problem Statement

| Parameter | Details |
| :--- | :--- |
| **Input** | `candidates.jsonl` (100,000 candidate profiles)<br>One Job Description (JD) for Senior AI Engineer role |
| **Output** | `output/CodeFusion.csv` — Top 100 ranked candidates with human-readable reasoning |

---

## Required Files for Execution

Due to GitHub file size limitations (100 MB per file), the preprocessed output folder could not be pushed directly to this repository. Please download it separately.

### Download Preprocessed Cache
👉 **[Download output/ folder from Google Drive](https://drive.google.com/drive/folders/1kY6fZWpESaeaR7TBeM1WTAJOBPJo2viH)**
* **Size:** ~555 MB
* **Contents:** Contains all necessary preprocessed cache files for immediate ranking execution.

### Setup Instructions

1. Ensure `candidates.jsonl` (provided by organizers) is in the project root directory.
2. Download the `output/` folder from the link above and place it in the project root. Your project structure must look like this:

```text
redrob-ai-ranker/
├── rank.py
├── candidates.jsonl         ← Provided by organizers
├── output/                  ← Downloaded from Drive link
│   ├── CodeFusion.csv
│   ├── candidate_embeddings.npy
│   ├── faiss_hnsw.index
│   └── ... (other cache files)
├── requirements.txt
└── README.md

```

3. Open a terminal in the project root directory and run:

```bash
python rank.py

```

4. The ranking pipeline will complete in approximately **10 seconds** using the preprocessed cache. The final output is saved automatically as `output/CodeFusion.csv`.

> ### 💡 Note for Evaluators
> 
> 
> The preprocessed cache enables immediate, lightweight CPU-only ranking without requiring local GPU access. The `candidates.jsonl` file is identical to the one provided by the hackathon organizers. If you wish to verify the complete preprocessing pipeline from scratch, simply delete the `output/` folder and re-run `python rank.py` — this will regenerate all cache files (~30 minutes on GPU).

---

## System Architecture

### Offline Preprocessing (One-Time, GPU)

```text
                    candidates.jsonl (100K)
                               │
                               ▼
                 BGE-M3 Embeddings (GPU, FP16)
                               │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  FAISS HNSW Index     BM25 Index     Feature Matrix (9 features)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
              Skill Ontology + Percentile Normalizer
                           │
                           ▼
               Cached Artifacts (~1.2 GB)

```

### Online Ranking Pipeline (CPU Only, ~10 seconds)

```text
                 Job Description
                        │
                        ▼
             1. JD Embedding (BGE-M3, CPU)
                        │
                        ▼
           2. JD-Adaptive Weight Generation
                        │
                        ▼
        3. FAISS HNSW Retrieval (Top 1000)
                        │
                        ▼
      4. BM25 Re-ranking (on FAISS shortlist)
                        │
                        ▼
       5. Feature Scoring (23 Redrob Signals)
                        │
                        ▼
          6. Skill Ontology Concept Matching
                        │
                        ▼
          7. Hybrid Score Fusion + Tie-Breaking
                        │
                        ▼
     Top 100 Ranked Candidates with Reasoning

```

### Note on Terminology

In this documentation:
- **Offline Preprocessing** refers to one-time computation done before ranking (embedding generation, index building). This is performed once with GPU acceleration.
- **Online Ranking** refers to the real-time ranking pipeline that runs at query time using pre-computed artifacts. This runs on CPU only.

Both stages operate entirely without internet connectivity. No external API calls, cloud services, or network access are required. The term "online" describes the runtime/live nature of the ranking phase, not internet connectivity.


---

## Final Scoring Formula

```text
Final Score = (α × Semantic + β × BM25 + γ × Features)
              × Skill Boost
              × Experience Alignment Factor
              - Honeypot Penalty

Where:
  α, β, γ = JD-Adaptive Weights (auto-tuned per JD)
  Skill Boost = 0.8x - 1.4x (based on ontology matching)
  Experience Penalty = 0.5x - 1.0x (years vs target)

```

### Tie-Breaking Rule

1. **Final Score** (Descending, rounded to 4 decimal places)
2. **Candidate ID** (Ascending, alphabetical)

---

## Feature Engineering

| Feature | Description |
| --- | --- |
| `career_authenticity` | Percentage of career spent in genuine technical engineering roles |
| `retrieval_specialization` | Deep expertise depth in IR (Information Retrieval) and Search systems |
| `ai_experience` | AI/ML keyword density mapped throughout professional work experience |
| `career_progression` | Seniority growth and structural title trajectory over time |
| `skill_quality` | Skill depth weighted by duration, documented proficiency, and endorsements |
| `description_consistency` | Mathematical alignment between claimed skills and contextual job descriptions |
| `education` | Academic baseline relevance with structured tier bonuses for elite institutions |
| `behavioral` | Composite normalized score metric of all 23 Redrob behavioral signals (0-100) |
| `experience_years` | Total chronological years of professional workspace experience |

---

## 23 Redrob Behavioral Signals

| Category | Integrated Signals & Weights |
| --- | --- |
| **1. Engagement Signals (3)** | `recruiter_response_rate` (×35), `interview_completion_rate` (×30), `offer_acceptance_rate` (×25) |
| **2. Pipeline Conversion (3)** | `application_to_interview_rate` (×20), `interview_to_offer_rate` (×25), `offer_to_joining_rate` (×15) |
| **3. Visibility Signals (5)** | `profile_completeness_score` (×0.3), `saved_by_recruiters_30d` (×0.8), `profile_views_received_30d` (×0.04), `profile_views_7d` (×0.12), `search_appearances_30d` (×0.15 or -5 penalty if zero) |
| **4. Responsiveness Signals (3)** | `open_to_work_flag` (+10 if true), `avg_response_time_hours` (tiered), `inmail_response_rate` (×15) |
| **5. Network Strength (3)** | `connection_acceptance_rate` (×8), `skill_endorsements_count` (×0.1), `recommendations_count` (×2.0) |
| **6. Activity Signals (2)** | `profile_update_frequency` (tiered), `job_search_activity_score` (×0.1) |
| **7. Job Fit Signals (2)** | `location_match_score` (×8), `salary_expectation_alignment` (×12) |
| **8. Administrative (2)** | `notice_period_days` (tiered), `visa_sponsorship_needed` (-8 penalty or +3 bonus) |

> **Total: 23/23 Redrob behavioral signals fully integrated into the ranking engine**

---

## Skill Ontology

| Concept | Skills Mapped |
| --- | --- |
| **Fine Tuning** | LoRA, QLoRA, PEFT, fine-tuning, finetuning |
| **Vector Databases** | FAISS, Pinecone, Milvus, Qdrant, Weaviate, ChromaDB |
| **Retrieval** | BM25, hybrid search, semantic search, dense retrieval |
| **Ranking** | Learning-to-Rank, NDCG, MRR, MAP |
| **Embeddings** | Sentence Transformer, BGE, E5, OpenAI Embedding |
| **LLM** | GPT, Claude, LLaMA, Mistral |
| **NLP** | Tokenization, NER, Sentiment Analysis |
| **ML** | Transformers, Deep Learning, CNN, RNN |
| **Cloud** | AWS, Azure, GCP |
| **DevOps** | Docker, Kubernetes, Terraform, CI/CD |

---

## Sandbox Demo (Google Colab)

🚀 **[Open Colab Sandbox](https://colab.research.google.com/drive/1921S3M6IlsciRI9vShxmOV8JplaUsuc2?usp=sharing)**

### How Judges Use the Sandbox

1. Open the Colab link.
2. Click **Runtime > Run All**.
3. Upload `rank.py` when prompted (available in your GitHub repository root).
4. Upload a small validation sample of `candidates.jsonl` when prompted (≤100 entries).
5. Wait ~1-2 minutes for the pipeline to finish execution.
6. `CodeFusionsandboxdemo.csv` downloads automatically to your system containing the sorted results.

### Sandbox vs Production

| Component | Sandbox (Colab) | Production (`rank.py`) |
| --- | --- | --- |
| **Model** | `all-MiniLM-L6-v2` (Lightweight) | `BAAI/bge-m3` (State-of-the-Art) |
| **Embedding Dimension** | 384 | 1024 |
| **Candidate Scope** | Max ≤100 samples | Full 100,000 dataset profiles |
| **Execution Time** | <2 minutes | ~10 seconds (Cached) |
| **Hardware Device** | Standard CPU | GPU (Preprocess) + CPU (Online Ranking) |
| **Output Target** | `CodeFusionsandboxdemo.csv` | `output/CodeFusion.csv` |

---

## Installation

### Prerequisites

* Python 3.8 or higher
* 16 GB RAM (Recommended baseline)
* NVIDIA GPU with 6GB+ VRAM (Required *only* if rebuilding raw preprocessing cache)

### Setup

```bash
# Clone the repository
git clone [https://github.com/Revathi2006/redrob-ai-ranker](https://github.com/Revathi2006/redrob-ai-ranker)
cd redrob-ai-ranker

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install PyTorch with CUDA support (Optimized for GPU Preprocessing)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install core system dependencies
pip install sentence-transformers faiss-cpu rank-bm25 scikit-learn pandas numpy tqdm pyyaml

```

---

## Usage

Ensure your downloaded `output/` cache folder and `candidates.jsonl` file are inside the project directory, then run:

```bash
python rank.py

```

* **First run (without cache folder):** ~30 minutes using GPU-accelerated embedding generation.
* **Subsequent runs (with cache folder):** ~10 seconds on regular CPU.

---

## Output Format

* **File Destination:** `output/CodeFusion.csv`
* **Column Schema:** `candidate_id, rank, score, reasoning`

### Example Format

```csv
candidate_id,rank,score,reasoning
CAND_0018499,1,0.8305,"Senior ML Engineer at Zomato with 7 years experience. strong qlora, milvus, embeddings background relevant to JD requirements. consistent technical career in AI/ML with deep retrieval systems expertise..."
CAND_0039754,2,0.8092,"Senior Applied Scientist at Meta with 16 years experience. strong llms, python, qdrant background relevant to JD requirements. strong senior-level experience exceeds minimum requirements..."

```

---

## Performance Benchmarks

**Hardware Environment Specifications:** Ryzen 7 7735HS | RTX 4050 (6GB VRAM) | 16GB DDR5 RAM | NVMe SSD | 100K Corpus.

| Stage | Execution Time | Hardware Device |
| --- | --- | --- |
| **Embedding Generation** | 20-25 min | GPU (FP16 Optimization) |
| **FAISS Index Build** | 2-3 min | CPU |
| **BM25 Index Build** | 1-2 min | CPU |
| **Feature Computation** | 3-5 min | CPU |
| **Total One-Time Preprocessing** | **~30 min** | **Mixed Architecture** |
| JD Embedding Generation | ~2s | CPU |
| FAISS Shortlist Retrieval | ~1s | CPU |
| BM25 Shortlist Re-ranking | ~6s | CPU |
| Hybrid Score Fusion Matrix | ~1s | CPU |
| **Total Query Ranking Pipeline** | **~10s** | **CPU Only** |

---

## Project Structure

```text
redrob-ai-ranker/
├── rank.py                       # Main ranking pipeline application
├── candidates.jsonl              # Raw input dataset (100K profiles)
├── requirements.txt              # Project environment locks
├── README.md                     # Engineering documentation
├── submission_metadata.yaml      # Submission verification manifest
└── output/                       # Generated Artifact Repository
    ├── CodeFusion.csv            # Final Top 100 Output Report
    ├── candidate_embeddings.npy  # Precomputed dense vector arrays
    ├── faiss_hnsw.index          # Serialized HNSW index graph
    ├── feature_matrix.npy        # Extracted mathematical feature arrays
    └── ...                       # Secondary lookup metadata cache files

```

---

## Validation

Verify that output files fit structural expectations by launching the native script check:

```bash
python validate_submission.py ./output/CodeFusion.csv

```

**Expected Terminal Output:**

```text
Submission is valid.

```

### Validation Compliance Checklist

* Exactly 100 target candidates returned.
* Ranks match sequentially starting from index 1.
* Identical edge scores break ties using alphanumeric ascending order on `candidate_id`.
* Includes all mandatory schema columns (`candidate_id`, `rank`, `score`, `reasoning`).
* All final internal scores consistently rounded to 4 decimal points.

---

## Reproducibility

To guarantee programmatic consistency across any evaluator's environment:

* **Fixed Structural Seed:** `random_state=42` locked throughout the `QuantileTransformer` processing layers.
* **Deterministic Search Graphs:** Fixed operational `efConstruction` and `efSearch` parameters for FAISS HNSW.
* **File Validation Checks:** Real-time SHA-256 validation mapping on input structures blocks mismatched cache data.
* **Explicit Score Tie-Breaking:** Scores rounded exactly once before triggering downstream sort configurations to assure predictable matching loops.

---

## Team

**Team Name:** Code Fusion

* **Revathi S** — revathis19112006@gmail.com
* **Rattishkumar SS** — rattishkumars@gmail.com
* **Yasvanth RD** — yasvanth178@gmail.com *(Team Lead)*

---

## License

Created exclusively for the **Redrob Hackathon Evaluation**. All rights reserved.

```

```
