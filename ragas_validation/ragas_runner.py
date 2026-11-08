"""Run the HR agent through Ragas and write validation reports."""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from sentence_transformers import SentenceTransformer

from .dataset import EvaluationCase
from .groq_agent import AgentResponse
from .logging_config import configure_logging
from .settings import EvaluationSettings


LOGGER = configure_logging()
METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
METRIC_COLUMN_ALIASES = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy",),
    "context_precision": (
        "context_precision",
        "llm_context_precision_with_reference",
    ),
    "context_recall": ("context_recall",),
}
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class AgentProtocol(Protocol):
    """Define the smallest agent interface required by the runner.

    Methods:
        answer_question: Return the routing, contexts, and answer for a question.
    """

    def answer_question(self, question: str, top_k: int = 3) -> AgentResponse:
        """Process one employee question.

        Parameters:
            question: Non-empty employee policy question.
            top_k: Maximum retrieved chunks.

        Returns:
            An AgentResponse containing routing, contexts, and answer.

        Raises:
            ValueError: If the question is empty.
            RuntimeError: If agent processing fails.
        """
        ...


class MiniLMMRagasEmbeddings(BaseRagasEmbeddings):
    """Adapt SentenceTransformer embeddings to the Ragas embedding interface.

    Purpose:
        Reuse the same local MiniLM model for answer relevancy embeddings.

    Parameters:
        model_name: SentenceTransformers model identifier.

    Exceptions raised:
        OSError: If the embedding model cannot be loaded.
        RuntimeError: If a query or document cannot be encoded.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        super().__init__()
        LOGGER.debug(
            "Function entry: MiniLMMRagasEmbeddings.__init__",
            extra={"model_name": model_name},
        )
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as error:
            LOGGER.exception("Failed to load embedding model '%s'", model_name)
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'."
            ) from error

    def embed_query(self, text: str) -> list[float]:
        """Embed one query string.

        Parameters:
            text: Non-empty query text.

        Returns:
            One embedding vector.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If encoding fails.
        """
        if not isinstance(text, str) or not text.strip():
            LOGGER.error("Embedding query cannot be empty")
            raise ValueError("Embedding query cannot be empty.")
        try:
            vector = self._model.encode(text.strip(), normalize_embeddings=True)
            return [float(value) for value in vector]
        except Exception as error:
            LOGGER.exception("Failed to embed query text")
            raise RuntimeError("Could not embed query text.") from error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings.

        Parameters:
            texts: Non-empty strings to embed.

        Returns:
            One embedding vector per input document.

        Raises:
            ValueError: If texts is empty or contains invalid values.
            RuntimeError: If encoding fails.
        """
        if not texts:
            LOGGER.error("Embedding document list cannot be empty")
            raise ValueError("Embedding document list cannot be empty.")
        normalized_texts: list[str] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                LOGGER.error("Every embedding document must be non-empty text")
                raise ValueError("Every embedding document must be non-empty text.")
            normalized_texts.append(text.strip())

        try:
            vectors = self._model.encode(normalized_texts, normalize_embeddings=True)
            return [[float(value) for value in vector] for vector in vectors]
        except Exception as error:
            LOGGER.exception("Failed to embed document texts")
            raise RuntimeError("Could not embed document texts.") from error

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously embed one query string.

        Parameters:
            text: Query text.

        Returns:
            One embedding vector.

        Raises:
            ValueError: If text is empty.
            RuntimeError: If encoding fails.
        """
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously embed a list of document strings.

        Parameters:
            texts: Document texts.

        Returns:
            Embedding vectors for all documents.

        Raises:
            ValueError: If texts are invalid.
            RuntimeError: If encoding fails.
        """
        return await asyncio.to_thread(self.embed_documents, texts)


def routing_is_correct(expected_category: str, selected_category: str | None) -> bool:
    """Check whether the agent selected the expected policy category.

    Parameters:
        expected_category: Category declared by the evaluation case.
        selected_category: Category chosen by the agent.

    Returns:
        True only when both categories match exactly.

    Raises:
        None.
    """
    return bool(
        expected_category.strip()
        and selected_category
        and expected_category.strip() == selected_category.strip()
    )


def create_timestamped_run_folder(results_dir: Path) -> Path:
    """Create and return a unique timestamped report directory.

    Parameters:
        results_dir: Parent directory containing validation runs.

    Returns:
        A newly created run directory.

    Raises:
        OSError: If report directories cannot be created.
    """
    LOGGER.debug(
        "Function entry: create_timestamped_run_folder",
        extra={"results_dir": str(results_dir)},
    )
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_dir / timestamp
        counter = 1
        while run_dir.exists():
            run_dir = results_dir / f"{timestamp}_{counter}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        LOGGER.exception("Could not create evaluation results directory")
        raise

    LOGGER.info("Created evaluation run directory: %s", run_dir)
    return run_dir


def append_observation(
    observations_path: Path,
    observations: list[dict[str, object]],
    observation: dict[str, object],
) -> None:
    """Append and persist one completed evaluation observation.

    Parameters:
        observations_path: File receiving the observation JSON array.
        observations: In-memory observations accumulated for this run.
        observation: New completed observation.

    Returns:
        None.

    Raises:
        OSError: If observations cannot be written.
    """
    LOGGER.debug("Function entry: append_observation")
    observations.append(observation)
    try:
        observations_path.write_text(
            json.dumps(observations, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        LOGGER.exception("Could not persist observations to %s", observations_path)
        raise


def write_scores_csv(
    scores_path: Path,
    observations: list[dict[str, object]],
    ragas_scores: list[dict[str, float]],
) -> None:
    """Write per-case routing results and Ragas scores to CSV.

    Parameters:
        scores_path: Destination CSV file.
        observations: Completed agent observations.
        ragas_scores: Per-case score dictionaries from Ragas.

    Returns:
        None.

    Raises:
        ValueError: If observation and score counts differ.
        OSError: If the CSV cannot be written.
    """
    LOGGER.debug("Function entry: write_scores_csv")
    if len(observations) != len(ragas_scores):
        LOGGER.error(
            "Observation count %d does not match Ragas score count %d",
            len(observations),
            len(ragas_scores),
        )
        raise ValueError("Observation and Ragas score counts must match.")

    fieldnames = [
        "id",
        "category",
        "sub_category",
        "question",
        "routing_correct",
        *METRIC_NAMES,
    ]
    try:
        with scores_path.open("w", encoding="utf-8", newline="") as scores_file:
            writer = csv.DictWriter(scores_file, fieldnames=fieldnames)
            writer.writeheader()
            for observation, score in zip(observations, ragas_scores):
                writer.writerow({
                    "id": observation["id"],
                    "category": observation["category"],
                    "sub_category": observation["sub_category"],
                    "question": observation["question"],
                    "routing_correct": observation["routing_correct"],
                    **score,
                })
    except OSError:
        LOGGER.exception("Could not write scores CSV to %s", scores_path)
        raise

    LOGGER.info("Wrote evaluation scores: %s", scores_path)


def write_summary_json(
    summary_path: Path,
    settings: EvaluationSettings,
    dataset_name: str,
    observations: list[dict[str, object]],
    ragas_scores: list[dict[str, float]],
) -> None:
    """Write aggregate evaluation statistics.

    Parameters:
        summary_path: Destination JSON file.
        settings: Evaluator settings used for the run.
        dataset_name: Name identifying the evaluated dataset.
        observations: Completed observations.
        ragas_scores: Per-case Ragas scores.

    Returns:
        None.

    Raises:
        ValueError: If scores or observations are empty or mismatched.
        OSError: If summary JSON cannot be written.
    """
    LOGGER.debug("Function entry: write_summary_json")
    if not observations:
        LOGGER.error("Cannot write summary for an empty evaluation")
        raise ValueError("Observations cannot be empty.")
    if len(observations) != len(ragas_scores):
        LOGGER.error("Observation and Ragas score counts do not match")
        raise ValueError("Observation and Ragas score counts must match.")

    metric_means = {}
    for metric_name in METRIC_NAMES:
        values = [
            score[metric_name]
            for score in ragas_scores
            if isinstance(score.get(metric_name), (int, float))
        ]
        metric_means[metric_name] = round(sum(values) / len(values), 6) if values else None

    summary = {
        "dataset_name": dataset_name,
        "groq_model": settings.groq_model,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "case_count": len(observations),
        "routing_accuracy": round(
            sum(bool(row["routing_correct"]) for row in observations)
            / len(observations),
            6,
        ),
        "metric_means": metric_means,
    }
    try:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        LOGGER.exception("Could not write summary JSON to %s", summary_path)
        raise

    LOGGER.info("Wrote evaluation summary: %s", summary_path)


def run_validation(
    cases: list[EvaluationCase],
    settings: EvaluationSettings,
    agent: AgentProtocol,
) -> Path:
    """Run agent cases through Chroma, Groq, Ragas, and report writers.

    Parameters:
        cases: Validated evaluation cases.
        settings: Evaluator configuration.
        agent: Agent implementing the AgentProtocol interface.

    Returns:
        Path to the completed timestamped report directory.

    Raises:
        ValueError: If cases are empty or configuration is incomplete.
        RuntimeError: If Ragas setup, agent execution, or scoring fails.
        OSError: If report files cannot be written.
    """
    LOGGER.info("Application start: running Ragas validation")
    LOGGER.debug("Function entry: run_validation", extra={"case_count": len(cases)})
    settings.require_groq_credentials()
    if not cases:
        LOGGER.error("Ragas validation received no cases")
        raise ValueError("Ragas validation requires at least one case.")

    run_dir = create_timestamped_run_folder(settings.results_dir)
    observations_path = run_dir / "observations.json"
    scores_path = run_dir / "scores.csv"
    summary_path = run_dir / "summary.json"
    observations: list[dict[str, object]] = []
    append_observation(observations_path, observations, {
        "status": "run_started",
        "dataset_case_count": len(cases),
    })
    observations.clear()

    evaluation_rows: dict[str, list[str]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    try:
        for case in cases:
            LOGGER.info("Evaluating case: %s", case.case_id)
            agent_response = agent.answer_question(case.question)
            contexts = [chunk.text for chunk in agent_response.retrieved_chunks]
            routing_correct = routing_is_correct(
                case.category,
                agent_response.category,
            )

            observation = {
                "id": case.case_id,
                "category": case.category,
                "sub_category": case.sub_category,
                "question": case.question,
                "ground_truth": case.ground_truth,
                "selected_category": agent_response.category,
                "selected_sub_category": agent_response.subcategory,
                "routing_correct": routing_correct,
                "retrieved_chunk_ids": [
                    chunk.chunk_id for chunk in agent_response.retrieved_chunks
                ],
                "retrieved_contexts": contexts,
                "answer": agent_response.answer,
            }
            append_observation(observations_path, observations, observation)
            evaluation_rows["question"].append(case.question)
            evaluation_rows["answer"].append(agent_response.answer)
            evaluation_rows["contexts"].append(contexts)
            evaluation_rows["ground_truth"].append(case.ground_truth)
    except Exception as error:
        LOGGER.exception("Agent execution failed before validation could finish")
        raise RuntimeError("Agent execution failed during evaluation.") from error

    try:
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
                temperature=0.0,
            )
        )
        ragas_embeddings = MiniLMMRagasEmbeddings()
        metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            LLMContextPrecisionWithReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
        ]
        dataset = Dataset.from_dict(evaluation_rows)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as error:
        LOGGER.exception("Ragas setup or scoring failed")
        raise RuntimeError("Ragas scoring failed.") from error

    result_dataframe = result.to_pandas()
    ragas_scores: list[dict[str, float]] = []
    for _, row in result_dataframe.iterrows():
        score_row: dict[str, float] = {}
        for metric_name in METRIC_NAMES:
            matching_aliases = METRIC_COLUMN_ALIASES[metric_name]
            matching_column = next(
                (
                    column
                    for column in row.index
                    if str(column) in matching_aliases
                    or str(column).lower().replace(" ", "_") in matching_aliases
                ),
                None,
            )
            if matching_column is not None:
                    score_row[metric_name] = (
                        float(row[matching_column])
                        if pd.notna(row[matching_column])
                        else 0.0
                    )
            else:
                    score_row[metric_name] = 0.0
        ragas_scores.append(score_row)

    write_scores_csv(scores_path, observations, ragas_scores)
    write_summary_json(
        summary_path,
        settings,
        dataset_name="curated",
        observations=observations,
        ragas_scores=ragas_scores,
    )
    LOGGER.info("Ragas validation complete: %s", run_dir)
    return run_dir
