# Ragas validation

This end-to-end evaluation runs ten curated HR-policy questions through the
agent's intent classification, ChromaDB retrieval, and answer generation. It
then uses Ragas to score faithfulness, answer relevancy, context precision, and
context recall. The test set covers all five top-level policy categories, with
at least two questions per category.

For the full file inventory, module responsibilities, data schemas, dependency
versions, and recreation steps, see [RECREATE_GUIDE.md](./RECREATE_GUIDE.md).

## Streamlit dashboard

From the workspace root, launch the dashboard with the project environment:

```powershell
.\.rag_programs\Scripts\python.exe -m streamlit run ".\RAG_Programs\ragas_validation\dashboard.py"
```

The dashboard shows aggregate and per-question metrics, test inputs, generated
answers, retrieved chunks, routing results, and comparisons between completed
runs. Use **Run full validation** to run the same evaluation pipeline in the
dashboard. Live evaluation calls Groq for both agent answers and Ragas judging
and may take several minutes.

The **Synthetic test data** tab can generate a separate draft dataset from the
HR policy chunks. Review or edit its questions and reference answers, save the
reviewed dataset, then select the cases to evaluate. Synthetic datasets are
stored under `synthetic_datasets/`; the curated `test_cases.json` is not changed.
Loaded saved datasets are ready to evaluate without saving another copy.
Changes to a saved dataset must be saved as a new timestamped dataset, leaving
the original intact.
Synthetic generation also uses Groq and may take several minutes.

The script loads `GROQ_API_KEY` from the workspace-root `.env` file
(`RAG_Programs\.env`). An already-set environment variable takes precedence.
Run it from the workspace root with the project virtual environment:

```powershell
.\.rag_programs\Scripts\python.exe ".\.\RAG_Programs\ragas_validation\dashboard.py"
```

The evaluation makes Groq calls both for agent answers and Ragas judging. It
reuses the agent's `sentence-transformers/all-MiniLM-L6-v2` model for answer
relevancy embeddings. The script does not start the Streamlit UI.

Each run creates a timestamped folder under `results/` containing:

- `observations.json` — generated answers, retrieved contexts, routing decisions,
  and retrieved chunk IDs.
- `scores.csv` — per-question Ragas scores and routing results.
- `summary.json` — aggregate and per-category mean scores.

Edit `test_cases.json` to adjust the questions and reference answers. Scores are
reported as a baseline; no pass/fail thresholds are applied.