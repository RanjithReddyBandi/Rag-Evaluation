"""Command-line entry point for the curated HR Ragas evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .chroma_adapter import ChromaAdapter
    from .dataset import load_cases
    from .groq_agent import GroqHrAgent
    from .logging_config import configure_logging
    from .ragas_runner import run_validation
    from .settings import load_settings
else:
    package_parent = str(Path(__file__).resolve().parent.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    from ragas_validation.chroma_adapter import ChromaAdapter
    from ragas_validation.dataset import load_cases
    from ragas_validation.groq_agent import GroqHrAgent
    from ragas_validation.logging_config import configure_logging
    from ragas_validation.ragas_runner import run_validation
    from ragas_validation.settings import load_settings


LOGGER = configure_logging()


def main() -> int:
    """Run the curated validation and return a process exit code.

    Parameters:
        None.

    Returns:
        Zero for success and one for user-visible configuration or runtime
        failures.

    Raises:
        None.
    """
    LOGGER.info("Application start: curated Ragas CLI")
    try:
        settings = load_settings()
        settings.require_groq_credentials()
        cases = load_cases(Path(__file__).resolve().parent / "test_cases.json")
        chroma = ChromaAdapter(
            settings.chroma_db_dir,
            settings.chroma_collection_name,
        )
        agent = GroqHrAgent(settings, chroma)
        run_dir = run_validation(cases, settings, agent)
        LOGGER.info("CLI validation completed: %s", run_dir)
        print(f"Validation complete: {run_dir}")
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        LOGGER.error("CLI validation failed: %s", error)
        print(f"Validation could not run: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
