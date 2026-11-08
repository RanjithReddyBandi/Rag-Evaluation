"""Configuration settings for the HR Helpdesk Ragas validation project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


VALIDATION_FOLDER = Path(__file__).resolve().parent
RAG_EVALUATION_FOLDER = VALIDATION_FOLDER.parent
CHROMA_PROJECT_FOLDER = RAG_EVALUATION_FOLDER / "ChunksChroma"

DEFAULT_CHROMA_DB_DIR = CHROMA_PROJECT_FOLDER / "chroma_db"
DEFAULT_COLLECTION_NAME = "HR_Helpdesk_Policy"
DEFAULT_POLICY_CHUNKS_FILE = RAG_EVALUATION_FOLDER / "hr_policy_chunks.jsonl"
DEFAULT_RESULTS_DIR = VALIDATION_FOLDER / "results"
DEFAULT_SYNTHETIC_DATASETS_DIR = VALIDATION_FOLDER / "synthetic_datasets"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"


@dataclass(frozen=True)
class EvaluationSettings:
    """Store validated paths and external-service settings.

    Attributes:
        chroma_db_dir: Directory containing the persistent ChromaDB database.
        chroma_collection_name: ChromaDB collection used for retrieval.
        policy_chunks_file: Policy chunks used for synthetic question generation.
        results_dir: Directory where timestamped evaluation reports are written.
        synthetic_datasets_dir: Directory where reviewed synthetic datasets are saved.
        groq_api_key: Groq API key loaded from the environment.
        groq_base_url: OpenAI-compatible Groq API base URL.
        groq_model: Groq model used for generation and evaluation.
        log_level: Application logging level.
    """

    chroma_db_dir: Path
    chroma_collection_name: str
    policy_chunks_file: Path
    results_dir: Path
    synthetic_datasets_dir: Path
    groq_api_key: str | None
    groq_base_url: str
    groq_model: str
    log_level: str

    def require_groq_credentials(self) -> None:
        """Validate that Groq credentials are configured.

        Parameters:
            None.

        Returns:
            None.

        Raises:
            ValueError: If GROQ_API_KEY is empty or not configured.
        """
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Set it in the environment "
                "or in a .env file before running live evaluation."
            )


def load_environment_file(environment_file: Path | None = None) -> None:
    """Load a dotenv file without overriding shell environment variables.

    Parameters:
        environment_file: Optional dotenv file to load.

    Returns:
        None.

    Raises:
        OSError: If python-dotenv is present but the environment file cannot be read.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # The shell environment remains usable without python-dotenv.
        return

    selected_file = environment_file or (RAG_EVALUATION_FOLDER / ".env")
    if selected_file.is_file():
        load_dotenv(dotenv_path=selected_file, override=False)


def load_settings(environment_file: Path | None = None) -> EvaluationSettings:
    """Load and validate Ragas evaluator settings.

    Parameters:
        environment_file: Optional dotenv file. When omitted, the standard
            Rag-Evaluation/.env location is checked.

    Returns:
        A validated EvaluationSettings instance.

    Raises:
        ValueError: If a required string setting is empty or invalid.
        OSError: If the environment file cannot be read.
    """
    load_environment_file(environment_file)

    chroma_db_dir = Path(
        os.getenv("CHROMA_DB_DIR", str(DEFAULT_CHROMA_DB_DIR))
    ).expanduser().resolve()
    chroma_collection_name = os.getenv(
        "CHROMA_COLLECTION_NAME",
        DEFAULT_COLLECTION_NAME,
    ).strip()
    policy_chunks_file = Path(
        os.getenv("POLICY_CHUNKS_FILE", str(DEFAULT_POLICY_CHUNKS_FILE))
    ).expanduser().resolve()
    results_dir = Path(
        os.getenv("RAGAS_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))
    ).expanduser().resolve()
    synthetic_datasets_dir = Path(
        os.getenv(
            "RAGAS_SYNTHETIC_DATASETS_DIR",
            str(DEFAULT_SYNTHETIC_DATASETS_DIR),
        )
    ).expanduser().resolve()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip() or None
    groq_base_url = os.getenv(
        "GROQ_BASE_URL",
        DEFAULT_GROQ_BASE_URL,
    ).strip()
    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()

    if not chroma_collection_name:
        raise ValueError("CHROMA_COLLECTION_NAME cannot be empty.")
    if not groq_base_url:
        raise ValueError("GROQ_BASE_URL cannot be empty.")
    if not groq_model:
        raise ValueError("GROQ_MODEL cannot be empty.")

    return EvaluationSettings(
        chroma_db_dir=chroma_db_dir,
        chroma_collection_name=chroma_collection_name,
        policy_chunks_file=policy_chunks_file,
        results_dir=results_dir,
        synthetic_datasets_dir=synthetic_datasets_dir,
        groq_api_key=groq_api_key,
        groq_base_url=groq_base_url,
        groq_model=groq_model,
        log_level=log_level,
    )
