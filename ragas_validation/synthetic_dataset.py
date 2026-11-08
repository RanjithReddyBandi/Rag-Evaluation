"""Generate and save reviewed synthetic HR evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from .dataset import EvaluationCase, validate_cases
from .logging_config import configure_logging
from .settings import EvaluationSettings


LOGGER = configure_logging()
MIN_SYNTHETIC_CASES = 1
MAX_SYNTHETIC_CASES = 50


@dataclass(frozen=True)
class PolicyChunk:
    """Represent one policy chunk used for synthetic generation.

    Attributes:
        chunk_id: Stable source chunk identifier.
        text: Policy text used to generate a question.
        category: Top-level policy category metadata.
        subcategory: Policy subcategory metadata.
    """

    chunk_id: str
    text: str
    category: str
    subcategory: str


def _read_json_array_or_jsonl(file_path: Path) -> list[Any]:
    """Read a JSON array or JSON Lines policy file.

    Parameters:
        file_path: Source policy file.

    Returns:
        A list containing every parsed source row.

    Raises:
        FileNotFoundError: If the source file is missing.
        json.JSONDecodeError: If a JSON Lines or JSON file is malformed.
        ValueError: If the file root is neither an array nor valid JSON Lines.
        OSError: If the file cannot be read.
    """
    LOGGER.debug(
        "Function entry: _read_json_array_or_jsonl",
        extra={"path": str(file_path)},
    )
    if not file_path.is_file():
        LOGGER.error("Policy chunk file not found: %s", file_path)
        raise FileNotFoundError(f"Policy chunk file not found: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        LOGGER.exception("Could not read policy chunk file: %s", file_path)
        raise

    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, list):
            LOGGER.info("Loaded policy chunks as a JSON array")
            return parsed_json
    except json.JSONDecodeError:
        LOGGER.debug("Policy file is not one JSON array; trying JSON Lines")

    rows: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        normalized_line = line.strip()
        if not normalized_line:
            continue
        try:
            row = json.loads(normalized_line)
        except json.JSONDecodeError:
            LOGGER.exception(
                "Invalid JSONL row in policy file at line %d",
                line_number,
            )
            raise
        if not isinstance(row, dict):
            LOGGER.error("Policy JSONL row %d is not an object", line_number)
            raise ValueError(
                f"Policy JSONL row {line_number} must be a JSON object."
            )
        rows.append(row)

    if not rows:
        LOGGER.error("Policy chunk file contains no usable rows")
        raise ValueError("Policy chunk file contains no usable rows.")
    LOGGER.info("Loaded %d policy chunks from JSON Lines", len(rows))
    return rows


def load_policy_chunks(file_path: str | Path) -> list[PolicyChunk]:
    """Load and normalize flat or nested policy chunk records.

    Parameters:
        file_path: JSON array or JSONL policy chunk source.

    Returns:
        Validated PolicyChunk objects.

    Raises:
        FileNotFoundError: If file_path does not exist.
        json.JSONDecodeError: If source JSON is malformed.
        ValueError: If required chunk fields are missing or invalid.
        OSError: If the source cannot be read.
    """
    resolved_path = Path(file_path).expanduser().resolve()
    LOGGER.info("Loading policy chunks: %s", resolved_path)
    raw_rows = _read_json_array_or_jsonl(resolved_path)
    chunks: list[PolicyChunk] = []

    for row_number, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            LOGGER.error("Policy chunk row %d is not an object", row_number)
            raise ValueError(f"Policy chunk row {row_number} must be an object.")

        chunk_id = row.get("chunk_id")
        text = row.get("text", row.get("description"))
        category = row.get("category")
        subcategory = row.get("subcategory", row.get("sub_category"))

        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            category = category or metadata.get("category")
            subcategory = subcategory or metadata.get(
                "subcategory",
                metadata.get("sub_category"),
            )

        if not all(
            isinstance(value, str) and value.strip()
            for value in (chunk_id, text, category, subcategory)
        ):
            LOGGER.error(
                "Policy chunk row %d has missing text or metadata fields",
                row_number,
            )
            raise ValueError(
                f"Policy chunk row {row_number} requires chunk_id, text, "
                "category, and subcategory."
            )

        chunks.append(
            PolicyChunk(
                chunk_id=chunk_id.strip(),
                text=text.strip(),
                category=category.strip(),
                subcategory=subcategory.strip(),
            )
        )

    if not chunks:
        LOGGER.error("Policy source contains no chunks")
        raise ValueError("Policy source contains no chunks.")
    LOGGER.info("Normalized %d policy chunks", len(chunks))
    return chunks


def _parse_generated_case(model_text: str) -> dict[str, str]:
    """Parse one JSON object from generated model text.

    Parameters:
        model_text: Model response containing a JSON object.

    Returns:
        A dictionary containing question and ground_truth.

    Raises:
        ValueError: If no usable JSON object can be parsed.
    """
    LOGGER.debug("Function entry: _parse_generated_case")
    decoder = json.JSONDecoder()
    for index, character in enumerate(model_text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(model_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    LOGGER.error("Synthetic response did not contain valid JSON")
    raise ValueError("Synthetic response did not contain valid JSON.")


class SyntheticDatasetGenerator:
    """Generate policy-grounded synthetic draft evaluation cases.

    Purpose:
        Ask Groq to create one question/reference pair for selected policy
        chunks, then return those pairs for dashboard review before saving.

    Parameters:
        settings: Validated evaluator settings.

    Exceptions raised:
        ValueError: If credentials or generation inputs are invalid.
        RuntimeError: If Groq generation or response parsing fails.
    """

    def __init__(self, settings: EvaluationSettings) -> None:
        LOGGER.info("Application start: initializing synthetic dataset generator")
        settings.require_groq_credentials()
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )

    def generate_draft(self, size: int) -> list[EvaluationCase]:
        """Generate synthetic cases in memory without persisting them.

        Parameters:
            size: Number of cases to generate, from 1 through 50.

        Returns:
            Reviewable EvaluationCase objects mapped to source chunk metadata.

        Raises:
            ValueError: If size, policy chunks, or model JSON is invalid.
            RuntimeError: If Groq cannot generate a case.
            OSError: If policy chunks cannot be read.
        """
        LOGGER.debug("Function entry: generate_draft", extra={"size": size})
        if isinstance(size, bool) or not isinstance(size, int):
            LOGGER.error("Synthetic dataset size must be an integer")
            raise ValueError("Synthetic dataset size must be an integer.")
        if size < MIN_SYNTHETIC_CASES or size > MAX_SYNTHETIC_CASES:
            LOGGER.error("Synthetic dataset size is outside the allowed range")
            raise ValueError(
                f"Synthetic dataset size must be between "
                f"{MIN_SYNTHETIC_CASES} and {MAX_SYNTHETIC_CASES}."
            )

        chunks = load_policy_chunks(self._settings.policy_chunks_file)
        selected_chunks = chunks[:size]
        if len(selected_chunks) < size:
            selected_chunks = chunks

        draft_rows: list[dict[str, str]] = []
        for index in range(size):
            source_chunk = selected_chunks[index % len(selected_chunks)]
            user_prompt = (
                "Create one realistic employee HR helpdesk question and a concise "
                "reference answer grounded only in this policy excerpt.\n"
                f"Policy excerpt: {source_chunk.text}\n"
                "Return only valid JSON with these keys:\n"
                '{"question":"...", "ground_truth":"..."}'
            )
            try:
                response = self._client.chat.completions.create(
                    model=self._settings.groq_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Generate policy-grounded question and reference "
                                "answer pairs. Never invent facts."
                            ),
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=500,
                )
            except Exception as error:
                LOGGER.exception("Groq synthetic generation failed")
                raise RuntimeError(
                    "Groq could not generate synthetic test data."
                ) from error

            choices = response.choices or []
            if not choices or not isinstance(choices[0].message.content, str):
                LOGGER.error("Groq returned no synthetic JSON for chunk %s", source_chunk.chunk_id)
                raise RuntimeError("Groq returned no synthetic test case.")

            try:
                generated = _parse_generated_case(choices[0].message.content.strip())
            except ValueError:
                LOGGER.exception(
                    "Synthetic response could not be parsed for chunk %s",
                    source_chunk.chunk_id,
                )
                raise

            draft_rows.append(
                {
                    "id": f"synthetic-{source_chunk.chunk_id}-{index + 1}",
                    "category": source_chunk.category,
                    "sub_category": source_chunk.subcategory,
                    "question": str(generated.get("question", "")).strip(),
                    "ground_truth": str(generated.get("ground_truth", "")).strip(),
                }
            )

        try:
            validated_cases = validate_cases(draft_rows)
        except (TypeError, ValueError):
            LOGGER.exception("Generated synthetic cases failed schema validation")
            raise

        LOGGER.info("Generated %d synthetic draft cases", len(validated_cases))
        return validated_cases


def save_synthetic_dataset(
    settings: EvaluationSettings,
    name: str,
    cases: list[EvaluationCase],
) -> Path:
    """Save reviewed synthetic cases in a new timestamped dataset file.

    Parameters:
        settings: Evaluator settings containing the output directory.
        name: Optional human-readable dataset name without path separators.
        cases: Validated synthetic cases reviewed by the user.

    Returns:
        Path to the newly saved dataset.

    Raises:
        ValueError: If name contains path separators, or cases are invalid.
        FileExistsError: If a generated file path already exists.
        OSError: If the dataset cannot be written.
    """
    LOGGER.debug(
        "Function entry: save_synthetic_dataset",
        extra={"dataset_name": name, "case_count": len(cases)},
    )
    normalized_name = name.strip()
    if not normalized_name:
        normalized_name = "reviewed"
    if Path(normalized_name).name != normalized_name or normalized_name in {".", ".."}:
        LOGGER.error("Synthetic dataset name contains a path or traversal value")
        raise ValueError("Synthetic dataset name must not contain path separators.")

    if not cases:
        LOGGER.error("Cannot save an empty synthetic dataset")
        raise ValueError("Synthetic dataset cannot be empty.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in normalized_name
    ).strip("_")
    safe_name = safe_name or "reviewed"
    settings.synthetic_datasets_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.synthetic_datasets_dir / f"synthetic_{timestamp}_{safe_name}.json"
    if output_path.exists():
        LOGGER.error("Synthetic dataset output already exists: %s", output_path)
        raise FileExistsError(f"Synthetic dataset already exists: {output_path}")

    try:
        output_path.write_text(
            json.dumps(
                [case.to_dict() for case in cases],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        LOGGER.exception("Could not save synthetic dataset: %s", output_path)
        raise

    LOGGER.info("Saved synthetic dataset: %s", output_path)
    return output_path
