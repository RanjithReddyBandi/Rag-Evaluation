# Rag-Evaluation

This project demonstrates an HR helpdesk retrieval-augmented generation (RAG)
pipeline and evaluates it with Ragas. It contains:

- A ChromaDB ingestion, search, and dashboard example.
- A Ragas evaluation CLI for curated HR policy test cases.
- A Streamlit dashboard for running evaluations and reviewing results.
- Synthetic test-data generation based on policy chunks.

## Prerequisites

- Windows
- Python 3.11 or later
- Internet access for downloading packages and embedding models
- A Groq API key for answer generation and Ragas evaluation

## 1. Open the project folder

Run all commands from this directory:

```powershell
Set-Location "C:\Ranjith\FDE\IshaFDETraining\learnai\Python\RAG_Programs\Rag-Evaluation"
```

An existing virtual environment is included as `.venv-ragas`. You can activate
it or create a new one:

```powershell
# Existing environment
.\.venv-ragas\Scripts\Activate.ps1

# Or create and activate a new environment
py -m venv .venv-ragas
.\.venv-ragas\Scripts\Activate.ps1
```

If PowerShell blocks script activation, run it once in the current PowerShell
session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. Install dependencies

Install both dependency sets:

```powershell
python -m pip install --upgrade pip
python -m pip install -r ".\ChunksChroma\requirements.txt"
python -m pip install -r ".\ragas_validation\requirements.txt"
```

## 3. Configure Groq

Create `.env` in the `Rag-Evaluation` folder if it does not already exist, and
add:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Optional settings supported by the evaluation:

```dotenv
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=qwen/qwen3.8-27b
CHROMA_DB_DIR=ChunksChroma\chroma_db
CHROMA_COLLECTION_NAME=HR_Helpdesk_Policy
```

Do not commit API keys to Git. If a key has already been committed or shared,
revoke it and create a replacement.

## 4. Prepare the ChromaDB collection

The ingestion script loads the default HR policy JSON file and stores it in
the persistent `ChunksChroma\chroma_db` database.

```powershell
Set-Location ".\ChunksChroma"
python ".\ingest_to_chroma.py"
```

The first run can take time because Chroma downloads the
`sentence-transformers/all-MiniLM-L6-v2` embedding model. Run this step again
after changing the policy chunk data.

Optional custom paths:

```powershell
python ".\ingest_to_chroma.py" `
  --input ".\data\hr_helpdesk_policy_chunks.json" `
  --db-dir ".\chroma_db" `
  --collection "HR_Helpdesk_Policy"
```

Return to the project root:

```powershell
Set-Location ".."
```

## 5. Search the indexed policy

Run a command-line search:

```powershell
python ".\ChunksChroma\search_chroma.py" `
  --query "How many sick leave days do I get?" `
  --top-k 3
```

Filter by policy category:

```powershell
python ".\ChunksChroma\search_chroma.py" `
  --query "How many weeks of maternity leave are available?" `
  --category maternity_policy `
  --top-k 3
```

Common categories include:

- `employee_leave_policy`
- `maternity_policy`
- `remote_work_policy`
- `insurance_benefits_policy`

## 6. Run the Ragas CLI evaluation

The curated evaluation uses `ragas_validation\test_cases.json`.

```powershell
python -m ragas_validation.validate_ragas
```

Each completed run creates timestamped files under
`ragas_validation\results\<timestamp>`:

- `observations.json`: questions, answers, routing, and retrieved contexts
- `scores.csv`: scores for each question, when generated
- `summary.json`: aggregate and category metrics

## 7. Run the Streamlit dashboards

### ChromaDB dashboard

```powershell
python -m streamlit run ".\ChunksChroma\chroma_dashboard.py"
```

### Ragas dashboard

```powershell
python -m streamlit run ".\ragas_validation\dashboard.py"
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in a
browser. The Ragas dashboard can:

- View completed validation runs.
- Compare aggregate metric trends.
- Inspect generated answers and retrieved contexts.
- Run the curated evaluation.
- Generate, review, save, and evaluate synthetic test data.

## Project structure

```text
Rag-Evaluation/
|-- .env
|-- .venv-ragas/
|-- HR Policy Document.pdf
|-- hr_policy_chunks.jsonl
|-- ChunksChroma/
|   |-- ingest_to_chroma.py
|   |-- search_chroma.py
|   |-- chroma_dashboard.py
|   |-- data/
|   `-- chroma_db/
`-- ragas_validation/
    |-- validate_ragas.py
    |-- dashboard.py
    |-- test_cases.json
    |-- requirements.txt
    |-- results/
    `-- README.md
```

## Evaluation workflow

1. Ingest policy chunks into ChromaDB.
2. Route an evaluation question to a policy category.
3. Retrieve relevant chunks from ChromaDB.
4. Generate an answer with Groq.
5. Score the answer and retrieval with Ragas.
6. Review reports in the CLI output or Streamlit dashboard.

The evaluated metrics include answer relevancy, faithfulness, context
precision, and context recall, subject to the metrics and model compatibility
provided by the installed Ragas version.

## Troubleshooting

### `GROQ_API_KEY is not configured`

Make sure `.env` is in the project root and contains a valid Groq API key, or
set it for the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key"
```

### ChromaDB collection not found

Run the ingestion script from the project root:

```powershell
python ".\ChunksChroma\ingest_to_chroma.py"
```

### Import or package errors

Confirm the virtual environment is active and install both requirements files
again:

```powershell
python -m pip install -r ".\ChunksChroma\requirements.txt"
python -m pip install -r ".\ragas_validation\requirements.txt"
```
