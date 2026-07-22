# Anima — Local Cognitive AI System

Anima is a fully local, modular AI assistant built around a biologically-inspired cognitive architecture. It runs entirely on your own hardware using quantized open-source models and provides conversational reasoning, autonomous math solving, code generation, OCR-based document learning, and a persistent relational memory graph — all without sending data to any external server.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Module Reference](#module-reference)
- [Models & Libraries Used](#models--libraries-used)
- [Setup & Requirements](#setup--requirements)
- [Running Anima](#running-anima)
- [Key Features](#key-features)
- [Memory System](#memory-system)
- [Cognitive Routing](#cognitive-routing)
- [Skill System](#skill-system)
- [REM Sleep & Autonomous Research](#rem-sleep--autonomous-research)
- [Citations](#citations)

---

## Architecture Overview

Anima maps its internal processing to biological brain lobes. Each lobe is a separate Python module responsible for a distinct cognitive function. The system boots a base language model (Gemma-3-4B, quantized to 4-bit) and dynamically swaps in specialist models (Math Brain, Code Brain, Vision) as needed via a Just-In-Time (JIT) loading strategy to manage VRAM.

```
User Input
    │
    ▼
[ Intent Router ] ──► Deterministic Matrix → BART Classifier → Arbitration Engine
    │
    ├──► CHAT_BRAIN   (Gemma-3-4B, base reasoning + memory recall)
    ├──► MATH_BRAIN   (Qwen2.5-Math-1.5B, JIT loaded)
    ├──► CODE_BRAIN   (Qwen2.5-Coder-3B, JIT loaded)
    └──► BASE_BRAIN   (Vision/OCR path via Nougat)
```

---

## Module Reference

| File | Biological Analogue | Role |
|---|---|---|
| `CAI.py` | Central Nervous System | Main entry point, intent routing, response orchestration |
| `CAIUI.py` | Interface | Streamlit-based chat UI with boot screen |
| `CAIStyles.py` | — | CSS and HTML templates for the UI |
| `CAI_memorygraph.py` | — | Interactive Pyvis memory map renderer |
| `Temporal_memory.py` | Temporal Lobe | Memory storage, retrieval, reinforcement, cognitive scoring |
| `Temporal_relational.py` | Temporal Lobe (Associative) | NetworkX-based relational graph, multi-hop traversal |
| `Hippocampus.py` | Hippocampus | Analogical memory reconstruction from fragments |
| `Hippocampus_dreams.py` | Hippocampus (REM) | Autonomous web research and dream-based consolidation |
| `Frontal_learning.py` | Frontal Lobe | PDF and image ingestion via Nougat OCR |
| `Occipital_vision.py` | Occipital Lobe | Vision pipeline, OCR engine base class, PDF-to-image conversion |
| `Parietal_math.py` | Parietal Lobe (Math) | Dedicated math solver using Qwen2.5-Math + SymPy verification |
| `Parietal_skills.py` | Parietal Lobe (Skills) | Skill registry, multi-agent code generation, isolated sandbox testing |
| `Prefrontal_planner.py` | Prefrontal Cortex (Planning) | Step-by-step task planning with tool-aware decomposition |
| `Prefrontal_curiosity.py` | Prefrontal Cortex (Curiosity) | Knowledge gap detection and dream-queue logging |

---

## Models & Libraries Used

### Language Models

| Model | Role | Source |
|---|---|---|
| `gemma-3-4b-it` (QAT Int4) | Base conversational brain | Google DeepMind |
| `facebook/bart-large-mnli` | Zero-shot intent classification | Meta AI [1] |
| `qwen2.5-math-1.5b-instruct` | Dedicated math reasoning | Alibaba Cloud [2] |
| `qwen2.5-coder-3b-instruct` | Code generation and skill building | Alibaba Cloud [3] |
| `facebook/nougat-small` | Academic OCR (PDF/image) | Meta AI [4] |
| `Babelscape/rebel-large` | Relation triple extraction | Babelscape [5] |

### Embedding & Retrieval

| Model / Library | Role | Source |
|---|---|---|
| `all-MiniLM-L6-v2` | Sentence embeddings for ChromaDB | Sentence Transformers [6] |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking | Sentence Transformers [7] |
| `ChromaDB` | Persistent vector database | Chroma [8] |
| `NetworkX` | Relational graph storage and traversal | Hagberg et al. [9] |

### Core Python Dependencies

```
transformers>=4.40
peft
bitsandbytes
torch>=2.0
chromadb
sentence-transformers
spacy (en_core_web_sm)
sympy
streamlit
pyvis
networkx
pdf2image
Pillow
requests
beautifulsoup4
duckduckgo-search (ddgs)
```

---

## Setup & Requirements

### Hardware

- **GPU**: NVIDIA GPU with at least **8 GB VRAM** recommended (12 GB+ for Code/Math JIT loading alongside base model).
- **RAM**: 16 GB minimum; 32 GB recommended.
- **Storage**: ~20 GB for all model weights.

### Model Paths

The following paths are hardcoded in the source and must be updated to match your local installation:

```python
# CAI.py
MODEL_PATH = "/home/johnray/Personal/gemma-3-transformers-gemma-3-4b-it-qat-int4-unquantized-v1"

# Parietal_math.py
MATH_MODEL_PATH = "/home/johnray/Personal/qwen2.5-math-transformers-1.5b-instruct-v1"

# Parietal_skills.py
CODER_MODEL_PATH = "/home/johnray/Personal/qwen2.5-coder-transformers-3b-instruct-v1"
```

### Install Dependencies

```bash
pip install transformers peft bitsandbytes torch chromadb sentence-transformers \
    spacy sympy streamlit pyvis networkx pdf2image pillow requests \
    beautifulsoup4 duckduckgo-search

python -m spacy download en_core_web_sm
```

---

## Running Anima

```bash
streamlit run CAIUI.py
```

Anima displays a boot screen while loading model weights, then launches the chat interface. The BART intent classifier loads into CPU RAM during boot; Gemma loads onto GPU.

---

## Key Features

### Conversational Reasoning
Anima maintains a sliding window of the last 10 messages as working memory. For complex tasks it enters a **System 2 Deliberation Loop** (`CognitiveScratchpad`) that iterates up to 5 reasoning cycles before producing a final answer.

### Math Solving
When a math query is detected (via regex heuristics or BART classification), the system JIT-loads `Qwen2.5-Math-1.5B`. Each problem is solved by three independent personas (Pure Mathematician, Math Professor, Applied Engineer) whose answers are cross-verified using SymPy symbolic evaluation. The Math Brain is flushed from VRAM immediately after use.

### Code & Tool Generation
The Code Brain (`Parietal_skills.py`) uses `Qwen2.5-Coder-3B` with a **Tree-of-Thoughts** strategy: three engineering personas (Software Engineer, Systems Engineer, Computer Engineer) independently draft solutions, each tested in an **isolated virtual environment sandbox** with a 15-second timeout and a 2 GB memory cap. The fastest passing candidate is saved as a reusable `.py` skill.

### Document Learning
Dropping a PDF or image into the sidebar triggers the **Frontal Lobe**. Nougat OCR (`facebook/nougat-small`) is JIT-loaded, extracts text and math from each page, stores the content in ChromaDB, and then flushes VRAM immediately.

### Vision & Reflex Actions
Images are processed through the **Occipital Vision Lobe**, which extracts an actionable state JSON. If a sensory anomaly is detected (e.g., an out-of-range sensor reading), Anima automatically fires a **Reflex Arc** — directly executing a matched tool from the skill registry without waiting for explicit user instruction.

---

## Memory System

Anima uses a three-layer memory architecture stored in `./ai_memory/`:

**Layer 1 — Episodic Memory**: Raw timestamped conversation events saved to ChromaDB. These represent what happened and are not verified by default.

**Layer 2 — Semantic Memory**: Factual claims distilled from each conversation using the LLM's schema extraction. These carry importance scores (1–10), confidence weights (0–1), and entity tags extracted by spaCy.

**Layer 3 — Relational Graph**: Logical triples `(subject, relation, object)` extracted by the REBEL transformer and stored as a directed NetworkX graph (`temporal_relational_graph.json`). Multi-hop traversal expands the query context at retrieval time.

### Memory Retrieval Pipeline

1. spaCy extracts seed entities from the user query.
2. Graph expansion via Dijkstra / BFS activates associated concepts.
3. ChromaDB performs semantic vector search against the expanded query.
4. A **Cross-Encoder reranker** (`ms-marco-MiniLM-L-6-v2`) re-scores the top candidates.
5. A **Hierarchy of Truth** applies score adjustments: verified semantic facts receive +5.0 logits; unverified episodic memories receive −5.0 to prevent echo-chamber drift.
6. The Hippocampus module performs **analogical synthesis** — connecting retrieved fragments to the current query through inference rather than verbatim recall.

### Memory Reinforcement

Every retrieved memory has its `confidence` and `usage_count` updated after each interaction. Correct predictions increase confidence by 0.1 (capped at 1.0); incorrect ones decrease it by 0.2 (floored at 0.1). The sleep cycle applies **Hebbian strengthening** (frequently used memories grow stronger) and **Ebbinghaus decay** (unused memories older than 30 days fade).

### Memory Visualization

Type `/visualize` in the chat to launch an interactive Pyvis network graph of all memory nodes and their relational edges in your default browser.

---

## Cognitive Routing

The intent router uses a **confidence-based arbitration engine** with four bidding rounds:

| Round | Mechanism | Weight |
|---|---|---|
| 0 | Smart Circuit Breaker (verified high-confidence memory) | +5.0 override |
| 1 | World Model simulation (Gemma evaluates 5 execution paths) | ×1.2 |
| 1.5 | Regex Heuristic (mandatory math detection) | +5.0 mandate |
| 2 | BART semantic classification | ×0.8 |
| 3 | Master's Correction feedback loop | +1.5 |

The winning brain must clear a 0.4 confidence threshold; otherwise the system defaults to `CHAT_BRAIN`.

A **Deterministic Matrix** of regex patterns runs before any probabilistic routing — certain phrase patterns (e.g., "build a script", "find the client") are hard-routed to their respective brains with no model involvement.

---

## Skill System

Anima can generate, save, and reuse Python tools autonomously. Skills live in `./skills/` and are loaded into `SKILL_REGISTRY` at boot. Each skill must expose a `run_skill(user_input: str) -> str` function.

The **Skill Governance** system tracks `usage_count`, `last_used`, and `fail_count` per skill. Tools unused for 30 days or with 3+ sandbox failures are automatically archived to `./skills/archive/`.

Running `check your tools` in chat triggers a full audit: every skill is re-tested in the sandbox and broken scripts are automatically repaired by the Qwen Coder before being overwritten.

---

## REM Sleep & Autonomous Research

Typing `[topic] /sleep` queues a topic for deep research. During a REM cycle (`Hippocampus_dreams.py`), Anima:

1. **Predicts** — generates 3 hypotheses about the topic using graph context.
2. **Experiments** — formulates a targeted DuckDuckGo search query to test those hypotheses.
3. **Evaluates** — scores **epistemic surprise** (0.0–1.0): how wrong were the hypotheses?
4. **Consolidates** — saves research to ChromaDB with importance scaled by surprise score, then extracts relational triples into the graph.

Logical contradictions detected in the graph are automatically queued as `CONTRADICTS` edges and scheduled for resolution in the next REM cycle.

---

## Citations

[1] Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*. ACL 2020. https://arxiv.org/abs/1910.13461

[2] Yang, A., et al. (2024). *Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement*. Alibaba Cloud. https://arxiv.org/abs/2409.12122

[3] Hui, B., et al. (2024). *Qwen2.5-Coder Technical Report*. Alibaba Cloud. https://arxiv.org/abs/2409.12186

[4] Blecher, L., Cucurull, G., Scialom, T., & Stojnic, R. (2023). *Nougat: Neural Optical Understanding for Academic Documents*. arXiv:2308.13418. https://arxiv.org/abs/2308.13418

[5] Cabot, P. L. H., & Navigli, R. (2021). *REBEL: Relation Extraction By End-to-end Language generation*. EMNLP 2021 Findings. https://aclanthology.org/2021.findings-emnlp.204

[6] Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. https://arxiv.org/abs/1908.10084

[7] Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv:1901.04085. https://arxiv.org/abs/1901.04085

[8] Chroma. (2023). *ChromaDB: The AI-native open-source embedding database*. https://www.trychroma.com

[9] Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). *Exploring network structure, dynamics, and function using NetworkX*. Proceedings of the 7th Python in Science Conference (SciPy 2008), 11–15.
