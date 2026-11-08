# Recreating the HR Helpdesk Ragas Validation

This guide documents how to reproduce the validation tools in this folder:
the Ragas evaluator, its Streamlit dashboard, the curated test dataset, and
the optional synthetic-dataset workflow.

The evaluator uses the HR Helpdesk agent's real intent routing, ChromaDB
retrieval, and answer-generation functions. It then scores the generated
answers and retrieved contexts with Ragas. The dashboard can also generate
draft questions and reference answers from the policy chunks, let a user
review and save them, and evaluate only selected cases.

## 1. Folder contents

```text
ragas_validation/
├── dashboard.py
├── validate_ragas.py
├── test_cases.json
├── README.md
├── RECREATE_GUIDE.md
├── results/
│   └── <timestamp>/
│       ├── observations.json
│       ├── scores.csv
│       └── summary.json
└── synthetic_datasets/
    └── synthetic_<timestamp>.json
```

`results/` and `synthetic_datasets/` are created by the tools when needed. The
timestamped examples under those folders are generated data, not Python source
modules. `__pycache__/` may also appear when Python runs; it is disposable
bytecode and is not part of the application.

### Source files and responsibilities

| File | Purpose |
|---|---|
| `validate_ragas.py` | Loads and validates test cases; generates synthetic cases with Ragas; runs the HR agent pipeline; calls Ragas metrics; writes evaluation reports. It can run from the command line or be imported by Streamlit. |
| `dashboard.py` | Streamlit UI for saved reports, workflow inspection, full curated evaluations, synthetic-case generation/review/editing/saving, selecting cases, and evaluating selected cases. |
| `test_cases.json` | Curated evaluation dataset. Each item includes an expected category, question, and reference answer. Synthetic cases never overwrite this file. |
| `README.md` | Short usage instructions. |
| `RECREATE_GUIDE.md` | This detailed implementation and setup guide. |

### Other files used by the evaluator

These are dependencies of the validation folder, stored in the surrounding
HR Helpdesk project:

| File/folder | How validation uses it |
|---|---|
| `../HR_agent.py` | Supplies `classify_question`, `search_chunks`, `generate_answer`, ChromaDB collection access, the Groq-compatible client, and the embedding model. |
| `../hr_policy_chunks.jsonl` | Input chunks used for synthetic generation. Each JSONL row must have `text` plus `metadata.category` and `metadata.sub_category`. |
| `../../chroma_db/` | Persistent Chroma database queried by the agent. Its location is configured in `HR_agent.py` relative to the HR Helpdesk project. |
| `../../../.env` | Workspace-root environment file. `validate_ragas.py` loads it with `python-dotenv`. |

The validation runner imports the agent's functions. It does **not** launch the
agent's Streamlit chat interface.

## 2. Python modules and main interfaces

### `validate_ragas.py`

The module is organized around these functions and types:

- `AgentSentenceTransformerEmbeddings`: adapts the already-loaded
  `sentence-transformers/all-MiniLM-L6-v2` model to Ragas' embedding interface.
  It sets a Ragas `RunConfig` and supports synchronous and asynchronous
  document/query embedding.
- `load_test_cases(path)` and `validate_test_cases(cases)`: validate the
  required schema, non-empty strings, and unique case IDs.
- `generate_synthetic_cases(size)`: reads the project's JSONL chunks, passes
  their text and category metadata to Ragas `TestsetGenerator`, and uses a
  single-hop-specific query synthesizer. It returns question/reference-answer
  cases whose category labels are mapped from their source chunks. Generation
  accepts 1–50 cases.
- `save_synthetic_dataset(name, cases)`: validates and writes a separate JSON
  file under `synthetic_datasets/`. It refuses path-like names and will not
  overwrite an existing dataset file.
- `evaluate_agent(cases, observations_path, progress_callback=None)`: for each
  case, calls `HR_agent.classify_question`, `HR_agent.search_chunks`, and
  `HR_agent.generate_answer`; records routing, retrieved chunk IDs and texts,
  reference answer, and generated answer. Completed observations are written
  incrementally.
- `run_validation(test_cases=None, dataset_name="curated",
  progress_callback=None)`: orchestrates the agent run, Ragas scoring, and
  timestamped report writing. If `test_cases` is omitted, it loads the curated
  `test_cases.json`; if supplied, it validates and runs only those cases.
- `main()`: command-line entry point for the curated evaluation.

The progress callback is optional. The dashboard uses it to show the current
case, routing, retrieval, generation, and Ragas-scoring steps.

### `dashboard.py`

The main UI is divided into dashboard tabs:

- **Dashboard**: aggregate metrics, mean metric scores by category, routing
  results, and per-question scores.
- **Question details**: question, reference answer, expected/selected route,
  retrieved chunks, generated answer, and its individual Ragas scores.
- **Run comparison**: metric trends from completed timestamped runs.
- **Workflow**: report dataset/model information and saved agent observations.
- **Synthetic test data**: generate a draft, review/edit its text, select cases
  (including select-all/deselect-all), save a separate dataset, and evaluate
  selected cases.

The dashboard calls the same `run_validation` function as the CLI. This avoids
maintaining a separate evaluation path.

## 3. Dataset schema

Both curated and saved synthetic datasets are JSON arrays. Every case must have
exactly these five non-empty string fields:

```json
{
  "id": "leave-earned-days",
  "category": "Leave Policy",
  "sub_category": "Leave Categories & Accrual",
  "question": "How many days of Earned Leave are allotted annually?",
  "ground_truth": "Employees receive 20 days of Earned Leave annually."
}
```

- `id`: unique within a dataset.
- `category` and `sub_category`: expected routing label, taken from an approved
  policy category.
- `question`: question sent to the agent.
- `ground_truth`: reference answer supplied to Ragas. It is not sent to the
  agent when it generates its response.

The dashboard's editor adds a temporary `evaluate` checkbox to each row. That
checkbox is a UI selection, not part of the saved dataset schema.

The curated examples in `test_cases.json` are maintained separately. Synthetic
generation makes a draft in memory first; the user must review it and save it
before it is persisted. Synthetic files are named
`synthetic_<timestamp>.json` under `synthetic_datasets/`.
Loading an existing synthetic dataset marks it as saved, so it can be evaluated
without saving a duplicate. If a loaded dataset is edited, saving the changes
creates a new timestamped dataset instead of overwriting the original.

## 4. Evaluation and generated-data flow

### Curated or selected-case evaluation

1. Load and validate cases.
2. Read `GROQ_API_KEY`; create the OpenAI-compatible Groq client using
   `https://api.groq.com/openai/v1` and `GROQ_MODEL`.
3. Import `HR_agent.py`, get its Chroma collection and SentenceTransformer
   model.
4. For each question, get the agent's intent and approved metadata filters.
5. Search Chroma with the agent's query and filters, using the agent's default
   top-K value.
6. Generate an answer from the retrieved excerpts, or use the agent's
   no-context/out-of-scope answer path.
7. Give Ragas the question, generated answer, retrieved context, and
   `ground_truth`.
8. Save individual observations, per-case scores, and aggregate scores.

### Synthetic data generation

1. Read `hr_policy_chunks.jsonl` and validate its text/category metadata.
2. Build LangChain `Document` inputs with the original category labels.
3. Ask Ragas `TestsetGenerator.generate_with_chunks` for the requested number
   of single-hop questions and reference answers using the configured Groq
   model.
4. Map each sample's Ragas reference context back to its source chunk and carry
   over that chunk's category/subcategory. Reject samples without a usable
   source context or with ambiguous labels.
5. Display generated cases in the dashboard editor. Users can edit the question
   and reference answer, select which cases to evaluate, and save the reviewed
   cases in a separate JSON file.

Synthetic answers are model-generated drafts, not automatically verified
policy facts. Review them against the policy document/chunks before using them
as evaluation references.

## 5. Ragas metrics and report fields

The evaluator uses Ragas 0.4.3's initialized metric classes:

- **Faithfulness**: whether answer claims are supported by retrieved context.
- **Answer relevancy**: whether the generated answer addresses the question.
  This metric uses the same MiniLM embedding model as the agent.
- **Context precision**: how relevant retrieved contexts are to the question
  and reference.
- **Context recall**: whether retrieved contexts contain information needed to
  support the reference answer.

Routing accuracy is calculated by this project by checking whether the
expected category/subcategory appears in the agent's selected filters. It is
not a Ragas metric.

Every evaluation creates a new `results/<timestamp>/` folder:

- `observations.json`: inputs, reference answers, generated answers, retrieved
  text, chunk IDs, selected filters, and routing correctness.
- `scores.csv`: per-question routing result and all four Ragas metric scores.
- `summary.json`: model, dataset name, number of cases, routing accuracy,
  overall metric means, and per-category means.

If agent execution or scoring fails, any observations completed before the
failure remain in the run folder.

## 6. Environment and dependencies

Use Python **3.11** and preferably a dedicated virtual environment. The
following versions were present in the working environment when this guide was
written:

```text
ragas==0.4.3
datasets==5.1.0
langchain==0.3.30
langchain-community==0.3.27
langchain-core==0.3.86
langchain-openai==0.3.35
openai==1.109.1
instructor==1.3.2
sentence-transformers==6.0.1
chromadb==1.5.9
streamlit==1.63.0
pandas==3.0.5
python-dotenv==1.2.3
```

The Ragas 0.4.3 evaluator uses metric classes from its version-specific
`ragas.metrics._...` modules. Keep Ragas pinned to 0.4.3 unless updating the
imports and testing the entire evaluation flow against a newer Ragas API.
LangChain should use its 0.3-compatible stack for this Ragas version; mixing in
LangChain 1.x packages can create incompatible core requirements.

For a fresh environment, create and activate it from the workspace root:

```powershell
py -3.11 -m venv .venv-ragas
.\.venv-ragas\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project's HR Helpdesk runtime dependencies as well as the
Ragas-compatible versions listed above. For an environment matching the
versions tested here:

```powershell
python -m pip install `
  "ragas==0.4.3" `
  "datasets==5.1.0" `
  "langchain==0.3.30" `
  "langchain-community==0.3.27" `
  "langchain-core==0.3.86" `
  "langchain-openai==0.3.35" `
  "openai==1.109.1" `
  "instructor==1.3.2" `
  "sentence-transformers==6.0.1" `
  "chromadb==1.5.9" `
  "streamlit==1.63.0" `
  "pandas==3.0.5" `
  "python-dotenv==1.2.3" `
  "rich==13.9.4" `
  "tenacity==8.5.0"
```

The existing HR agent may require additional project-specific packages
depending on its current imports. Check `HR_agent.py` and the embedding
pipeline's dependency list, then verify the environment with:

```powershell
python -m pip check
```

Use an isolated environment rather than changing a different application
environment to resolve unrelated package conflicts.

## 7. Required files and secrets

Before running, confirm that these project inputs exist:

```text
Wknd Batch/
├── chroma_db/
└── HRHelpdesk_RAGAgent/
    ├── HR_agent.py
    ├── hr_policy_chunks.jsonl
    └── ragas_validation/
        ├── dashboard.py
        ├── validate_ragas.py
        └── test_cases.json
```

Put the Groq key in the workspace-root `.env` file:

```dotenv
GROQ_API_KEY=replace_with_your_key
GROQ_MODEL=qwen/qwen3.8-27b
```

The code loads `.env` from the `RAG_Programs` workspace root. An environment
variable already set in the shell takes precedence. Do not commit `.env` or
write API keys into Python, JSON, or Markdown files.

The Chroma collection must already have been built by the project's embedding
pipeline. Synthetic generation reads the JSONL policy chunks; live evaluation
also queries the populated Chroma collection and may load model weights on its
first run.

## 8. Running the tools

Run the dashboard from the workspace root:

```powershell
.\.venv-ragas\Scripts\python.exe -m streamlit run ".\Wknd Batch\HRHelpdesk_RAGAgent\ragas_validation\dashboard.py"
```

In the dashboard:

1. Open **Synthetic test data**.
2. Choose a generation size from 1 to 50 and click **Generate synthetic
   draft**.
3. Review/edit each question and reference answer. The source category labels
   are fixed from matched policy chunks.
4. Check only the questions to evaluate, or use **Select all questions** /
   **Deselect all questions**.
5. Click **Save reviewed dataset**. The curated test set is not overwritten.
6. Click **Evaluate selected questions**. The dashboard runs only the checked
   cases and selects the resulting report.

To run the curated dataset without opening Streamlit:

```powershell
.\.venv-ragas\Scripts\python.exe ".\Wknd Batch\HRHelpdesk_RAGAgent\ragas_validation\validate_ragas.py"
```

Both generation and evaluation make Groq API calls. Evaluation makes calls for
agent routing, answer generation, and Ragas judging. These operations can take
several minutes and may incur API usage. Local embeddings use the Sentence
Transformers model and do not call Groq.

## 9. Recreate the code in another project

1. Create a validation folder adjacent to the existing agent and its chunk
   JSONL file.
2. Copy/adapt `validate_ragas.py`, `dashboard.py`, and `test_cases.json` from
   this folder.
3. Update `WORKSPACE_ROOT`, project paths, environment loading, Groq base URL,
   model name, and policy chunk metadata mapping for the target project.
4. Adapt `evaluate_agent` to call the target agent's actual routing, retrieval,
   and answer functions. Supply the evaluator with the target agent's real
   retrieved contexts, not fabricated context.
5. Keep the case schema consistent: `id`, `category`, `sub_category`,
   `question`, and `ground_truth`.
6. Keep metadata on synthetic input documents so generated reference contexts
   can be mapped to the correct expected category.
7. Create the report output directories and test report writes.
8. Run syntax/import checks and a mocked generation/evaluation test first.
   Then run a small live evaluation with an approved API key.
9. Inspect generated questions and reference answers before treating them as
   ground truth.

Do not treat a successful Ragas score as proof that the HR policy itself is
correct. The scores are relative to the supplied references, retrieved
contexts, test coverage, and evaluator model.