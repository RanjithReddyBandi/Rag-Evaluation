"""Loading and validation for Ragas evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .logging_config import configure_logging


LOGGER = configure_logging()

DATASET_FIELDS = {
    "id",
    "category",
    "sub_category",
    "question",
    "ground_truth",
}


@dataclass(frozen=True)
class EvaluationCase:
    """Represent one validated Ragas evaluation case.

    Attributes:
        case_id: Unique identifier for the case.
        category: Expected top-level policy category.
        sub_category: Expected policy subcategory.
        question: User question sent to the agent.
        ground_truth: Reference answer used by Ragas.
    """

    case_id: str
    category: str
    sub_category: str
    question: str
    ground_truth: str

    def to_dict(self) -> dict[str, str]:
        """Convert the case to the JSON dataset dictionary shape.

        Parameters:
            None.

        Returns:
            A dictionary containing exactly the five persisted dataset fields.

        Raises:
            None.
        """
        return {
            "id": self.case_id,
            "category": self.category,
            "sub_category": self.sub_category,
            "question": self.question,
            "ground_truth": self.ground_truth,
        }


def _require_non_empty_string(
    row: Mapping[str, Any],
    field_name: str,
    row_number: int,
) -> str:
    """Return one validated non-empty string field.

    Parameters:
        row: One raw dataset row.
        field_name: Required string field to read.
        row_number: One-based dataset position used in error messages.

    Returns:
        The normalized string value.

    Raises:
        ValueError: If the field is missing, blank, or not a string.
    """
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        LOGGER.error(
            "Dataset row %d has invalid '%s'; expected a non-empty string",
            row_number,
            field_name,
        )
        raise ValueError(
            f"Dataset row {row_number} field '{field_name}' must be a non-empty string."
        )
    return value.strip()


def validate_cases(raw_cases: Iterable[Any]) -> list[EvaluationCase]:
    """Validate raw dictionaries and return strongly typed evaluation cases.

    Parameters:
        raw_cases: Iterable containing candidate dataset rows.

    Returns:
        A validated list of EvaluationCase objects.

    Raises:
        TypeError: If raw_cases is not iterable or a row is not a dictionary.
        ValueError: If required fields are missing, duplicate IDs exist, or rows
            contain fields outside the approved dataset schema.
    """
    LOGGER.debug("Function entry: validate_cases")

    try:
        case_rows = list(raw_cases)
    except TypeError as error:
        LOGGER.exception("Dataset rows must be iterable")
        raise TypeError("Dataset rows must be iterable.") from error

    validated_cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()

    for row_number, row in enumerate(case_rows, start=1):
        if not isinstance(row, dict):
            LOGGER.error("Dataset row %d is not an object", row_number)
            raise TypeError(f"Dataset row {row_number} must be a JSON object.")

        unexpected_fields = set(row) - DATASET_FIELDS
        if unexpected_fields:
            LOGGER.error(
                "Dataset row %d has unexpected fields: %s",
                row_number,
                sorted(unexpected_fields),
            )
            raise ValueError(
                f"Dataset row {row_number} has unexpected fields: "
                f"{sorted(unexpected_fields)}."
            )

        case_id = _require_non_empty_string(row, "id", row_number)
        category = _require_non_empty_string(row, "category", row_number)
        sub_category = _require_non_empty_string(row, "sub_category", row_number)
        question = _require_non_empty_string(row, "question", row_number)
        ground_truth = _require_non_empty_string(row, "ground_truth", row_number)

        if case_id in seen_ids:
            LOGGER.error("Duplicate dataset id: %s", case_id)
            raise ValueError(f"Duplicate dataset id: {case_id}")
        seen_ids.add(case_id)

        validated_cases.append(
            EvaluationCase(
                case_id=case_id,
                category=category,
                sub_category=sub_category,
                question=question,
                ground_truth=ground_truth,
            )
        )

    if not validated_cases:
        LOGGER.error("Evaluation dataset is empty")
        raise ValueError("Evaluation dataset cannot be empty.")

    LOGGER.info("Validated %d evaluation cases", len(validated_cases))
    return validated_cases


def load_cases(dataset_path: str | Path) -> list[EvaluationCase]:
    """Load a JSON evaluation dataset and validate every case.

    Parameters:
        dataset_path: Path to a JSON array dataset.

    Returns:
        A validated list of EvaluationCase objects.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If JSON is invalid or the dataset is not an array.
        json.JSONDecodeError: If JSON syntax is invalid.
        OSError: If the dataset file cannot be read.
    """
    resolved_path = Path(dataset_path).expanduser().resolve()
    LOGGER.debug("Function entry: load_cases", extra={"path": str(resolved_path)})
    LOGGER.info("Loading evaluation dataset: %s", resolved_path)

    if not resolved_path.is_file():
        LOGGER.error("Evaluation dataset not found: %s", resolved_path)
        raise FileNotFoundError(f"Evaluation dataset not found: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as dataset_file:
            raw_data = json.load(dataset_file)
    except json.JSONDecodeError:
        LOGGER.exception("Invalid JSON in dataset: %s", resolved_path)
        raise
    except OSError:
        LOGGER.exception("Could not read dataset: %s", resolved_path)
        raise

    if not isinstance(raw_data, list):
        LOGGER.error("Dataset root must be a JSON array: %s", resolved_path)
        raise ValueError("Evaluation dataset root must be a JSON array.")

    cases = validate_cases(raw_data)
    LOGGER.info("Successfully loaded evaluation dataset: %s", resolved_path)
    return cases
