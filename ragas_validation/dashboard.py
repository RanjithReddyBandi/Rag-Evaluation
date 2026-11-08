"""Streamlit dashboard for HR Helpdesk Ragas reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from .chroma_adapter import ChromaAdapter
    from .dataset import load_cases
    from .groq_agent import GroqHrAgent
    from .logging_config import configure_logging
    from .ragas_runner import run_validation
    from .settings import load_settings
    from .dataset import EvaluationCase, validate_cases
    from .synthetic_dataset import (
        SyntheticDatasetGenerator,
        save_synthetic_dataset,
    )
except ImportError:
    import sys
    project_parent = str(Path(__file__).resolve().parent.parent)
    if project_parent not in sys.path:
        sys.path.insert(0, project_parent)
    from ragas_validation.chroma_adapter import ChromaAdapter
    from ragas_validation.dataset import load_cases
    from ragas_validation.groq_agent import GroqHrAgent
    from ragas_validation.logging_config import configure_logging
    from ragas_validation.ragas_runner import run_validation
    from ragas_validation.settings import load_settings
    from ragas_validation.dataset import EvaluationCase, validate_cases
    from ragas_validation.synthetic_dataset import (
        SyntheticDatasetGenerator,
        save_synthetic_dataset,
    )


LOGGER = configure_logging()


def get_run_directories(results_dir: Path) -> list[Path]:
    """Return completed report directories from newest to oldest.

    Parameters:
        results_dir: Root directory containing timestamped run folders.

    Returns:
        Sorted run directories containing a summary.json file.

    Raises:
        OSError: If the directory cannot be inspected.
    """
    LOGGER.debug("Function entry: get_run_directories")
    if not results_dir.exists():
        LOGGER.warning("Results directory does not exist: %s", results_dir)
        return []
    try:
        return sorted(
            (
                path for path in results_dir.iterdir()
                if path.is_dir() and (path / "summary.json").is_file()
            ),
            reverse=True,
        )
    except OSError:
        LOGGER.exception("Could not inspect results directory: %s", results_dir)
        raise


def read_json_file(file_path: Path) -> dict:
    """Read a JSON object and return it as a dictionary.

    Parameters:
        file_path: Path to JSON object.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If JSON syntax is invalid.
        OSError: If the file cannot be read.
        ValueError: If the root JSON value is not an object.
    """
    LOGGER.debug("Function entry: read_json_file", extra={"path": str(file_path)})
    if not file_path.is_file():
        LOGGER.error("JSON report not found: %s", file_path)
        raise FileNotFoundError(f"JSON report not found: {file_path}")
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.exception("Invalid JSON report: %s", file_path)
        raise
    except OSError:
        LOGGER.exception("Could not read JSON report: %s", file_path)
        raise
    if not isinstance(value, dict):
        LOGGER.error("JSON report root must be an object: %s", file_path)
        raise ValueError(f"JSON report root must be an object: {file_path}")
    return value


def read_scores_file(scores_path: Path) -> pd.DataFrame:
    """Read a score CSV report into a pandas DataFrame.

    Parameters:
        scores_path: Path to the score CSV.

    Returns:
        Non-empty score DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        csv.Error: If CSV parsing fails.
        pd.errors.ParserError: If pandas cannot parse the CSV.
        ValueError: If the CSV contains no rows.
        OSError: If the file cannot be read.
    """
    LOGGER.debug("Function entry: read_scores_file", extra={"path": str(scores_path)})
    if not scores_path.is_file():
        LOGGER.error("Scores CSV not found: %s", scores_path)
        raise FileNotFoundError(f"Scores CSV not found: {scores_path}")
    try:
        dataframe = pd.read_csv(scores_path)
    except (csv.Error, pd.errors.ParserError):
        LOGGER.exception("Could not parse scores CSV: %s", scores_path)
        raise
    except OSError:
        LOGGER.exception("Could not read scores CSV: %s", scores_path)
        raise
    if dataframe.empty:
        LOGGER.error("Scores CSV contains no rows: %s", scores_path)
        raise ValueError(f"Scores CSV contains no rows: {scores_path}")
    return dataframe


def format_metric_name(metric_name: str) -> str:
    """Convert a machine metric key into a readable dashboard label.

    Parameters:
        metric_name: Machine metric key.

    Returns:
        Human-readable metric label.

    Raises:
        None.
    """
    return metric_name.replace("_", " ").title()


def render_dashboard_tab(summary: dict, scores: pd.DataFrame) -> None:
    """Render aggregate metrics and the per-category score table.

    Parameters:
        summary: Parsed summary JSON for the selected run.
        scores: Parsed per-case score DataFrame.

    Returns:
        None.

    Raises:
        None.
    """
    LOGGER.debug("Function entry: render_dashboard_tab")
    metric_means = summary.get("metric_means", {})
    columns = st.columns(max(1, len(metric_means) + 1))
    columns[0].metric("Routing accuracy", f"{summary['routing_accuracy']:.1%}")

    for index, (metric_name, metric_value) in enumerate(
        metric_means.items(),
        start=1,
    ):
        value = metric_value if isinstance(metric_value, (int, float)) else 0.0
        columns[index].metric(format_metric_name(metric_name), f"{value:.3f}")

    st.subheader("Scores by policy category")
    category_metric_columns = [
        column for column in ("answer_relevancy", "faithfulness")
        if column in scores.columns
    ]
    if "category" in scores.columns and category_metric_columns:
        st.dataframe(
            scores.groupby("category", as_index=False)[category_metric_columns].mean(),
            use_container_width=True,
        )
    else:
        st.info("No category metric columns are available for this report.")

    st.subheader("Per-question scores")
    st.dataframe(scores, use_container_width=True)


def render_question_details_tab(run_dir: Path, scores: pd.DataFrame) -> None:
    """Render observations and per-question details for the selected run.

    Parameters:
        run_dir: Directory containing observations.json.
        scores: Parsed per-case score DataFrame.

    Returns:
        None.

    Raises:
        None.
    """
    LOGGER.debug("Function entry: render_question_details_tab")
    observations_path = run_dir / "observations.json"
    if not observations_path.is_file():
        st.warning("This run does not contain observations.json.")
        return

    try:
        observations = json.loads(observations_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        LOGGER.exception("Could not read question observations")
        st.error("Question observations could not be read.")
        return

    if not isinstance(observations, list):
        st.error("Question observations have an invalid format.")
        return

    case_labels = [
        f"{observation.get('id', index)}: {observation.get('question', '')}"
        for index, observation in enumerate(observations)
    ]
    selected_label = st.selectbox("Question", case_labels)
    selected_index = case_labels.index(selected_label)
    observation = observations[selected_index]

    st.markdown(f"**Question:** {observation.get('question', '')}")
    st.markdown(f"**Reference answer:** {observation.get('ground_truth', '')}")
    st.markdown(
        "**Selected route:** "
        f"{observation.get('selected_category')} / "
        f"{observation.get('selected_sub_category')}"
    )
    st.markdown(f"**Routing correct:** {observation.get('routing_correct')}")
    st.markdown("**Generated answer:**")
    st.write(observation.get("answer", ""))
    st.markdown("**Retrieved contexts:**")
    for context in observation.get("retrieved_contexts", []):
        st.write(context)


def render_run_comparison_tab(results_dir: Path) -> None:
    """Render metric trends across completed validation runs.

    Parameters:
        results_dir: Root directory containing timestamped runs.

    Returns:
        None.

    Raises:
        None.
    """
    LOGGER.debug("Function entry: render_run_comparison_tab")
    run_directories = get_run_directories(results_dir)
    rows = []
    for run_dir in run_directories:
        try:
            summary = read_json_file(run_dir / "summary.json")
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            LOGGER.exception("Could not load run summary: %s", run_dir)
            continue
        row = {"run": run_dir.name, **summary.get("metric_means", {})}
        row["routing_accuracy"] = summary.get("routing_accuracy")
        rows.append(row)

    if not rows:
        st.info("No completed runs are available for comparison.")
        return

    comparison = pd.DataFrame(rows).iloc[::-1].set_index("run")
    st.line_chart(comparison)


def run_curated_validation() -> Path | None:
    """Run the curated dataset from the dashboard.

    Parameters:
        None.

    Returns:
        Selected run directory when validation completes, otherwise None.

    Raises:
        None.
    """
    LOGGER.debug("Function entry: run_curated_validation")
    try:
        settings = load_settings()
        settings.require_groq_credentials()
        test_cases = load_cases(
            Path(__file__).resolve().parent / "test_cases.json"
        )
        chroma = ChromaAdapter(
            settings.chroma_db_dir,
            settings.chroma_collection_name,
        )
        agent = GroqHrAgent(settings, chroma)
        return run_validation(test_cases, settings, agent)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        LOGGER.exception("Curated dashboard validation failed")
        st.error(f"Validation could not start: {error}")
        return None


def render_synthetic_data_tab(settings) -> None:
    """Render synthetic draft generation, review, save, and evaluation.

    Parameters:
        settings: Validated evaluator settings.

    Returns:
        None.

    Raises:
        None.
    """
    LOGGER.debug("Function entry: render_synthetic_data_tab")
    st.subheader("Synthetic test data")
    generation_size = st.number_input(
        "Number of draft cases",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
    )

    if st.button("Generate synthetic draft"):
        try:
            generator = SyntheticDatasetGenerator(settings)
            draft_cases = generator.generate_draft(int(generation_size))
            st.session_state["synthetic_draft"] = [
                case.to_dict() for case in draft_cases
            ]
            LOGGER.info("Dashboard synthetic draft generated")
        except (ValueError, RuntimeError, OSError) as error:
            LOGGER.exception("Synthetic draft generation failed")
            st.error(f"Draft generation failed: {error}")

    if not st.session_state.get("synthetic_draft"):
        st.info("Generate a draft first, then review it before saving.")
        return

    editor_rows = [
        {**row, "evaluate": True}
        for row in st.session_state["synthetic_draft"]
    ]
    edited_rows = st.data_editor(
        pd.DataFrame(editor_rows),
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="synthetic_editor",
    )
    dataset_name = st.text_input(
        "Dataset name",
        value="reviewed_policy_cases",
    )

    if st.button("Save reviewed dataset"):
        try:
            save_rows = [
                {key: value for key, value in row.items() if key != "evaluate"}
                for row in edited_rows.to_dict("records")
            ]
            saved_cases = validate_cases(save_rows)
            output_path = save_synthetic_dataset(
                settings,
                dataset_name,
                saved_cases,
            )
            st.success(f"Synthetic dataset saved: {output_path}")
        except (ValueError, FileExistsError, OSError) as error:
            LOGGER.exception("Synthetic dataset save failed")
            st.error(f"Dataset could not be saved: {error}")

    if st.button("Evaluate selected questions"):
        try:
            selected_rows = [
                {key: value for key, value in row.items() if key != "evaluate"}
                for row in edited_rows.to_dict("records")
                if bool(row.get("evaluate", False))
            ]
            selected_cases = validate_cases(selected_rows)
            chroma = ChromaAdapter(
                settings.chroma_db_dir,
                settings.chroma_collection_name,
            )
            agent = GroqHrAgent(settings, chroma)
            selected_run = run_validation(selected_cases, settings, agent)
            st.session_state["selected_run"] = str(selected_run)
            st.success(f"Synthetic validation completed: {selected_run.name}")
        except (ValueError, RuntimeError, OSError) as error:
            LOGGER.exception("Selected synthetic evaluation failed")
            st.error(f"Synthetic evaluation failed: {error}")


def main() -> None:
    """Configure and render the Streamlit Ragas dashboard.

    Parameters:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    LOGGER.info("Application start: Ragas Streamlit dashboard")
    st.set_page_config(page_title="HR Ragas Validation", layout="wide")
    st.title("HR Helpdesk Ragas Validation")

    try:
        settings = load_settings()
    except (OSError, ValueError) as error:
        LOGGER.exception("Dashboard configuration failed")
        st.error(f"Configuration error: {error}")
        return

    if st.sidebar.button("Run curated validation"):
        with st.spinner("Running agent and Ragas evaluation..."):
            selected_run = run_curated_validation()
            if selected_run is not None:
                st.session_state["selected_run"] = str(selected_run)
                st.success(f"Validation completed: {selected_run.name}")

    run_directories = get_run_directories(settings.results_dir)
    if not run_directories:
        st.info("No completed validation runs yet. Configure GROQ_API_KEY and run a curated validation.")
        return

    run_names = [run_dir.name for run_dir in run_directories]
    default_run = st.session_state.get("selected_run")
    default_index = run_names.index(Path(default_run).name) if default_run in [
        str(run_dir) for run_dir in run_directories
    ] else 0
    selected_run_name = st.sidebar.selectbox(
        "Validation run",
        run_names,
        index=default_index,
    )
    selected_run = settings.results_dir / selected_run_name

    try:
        summary = read_json_file(selected_run / "summary.json")
        scores = read_scores_file(selected_run / "scores.csv")
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        LOGGER.exception("Could not load selected validation run")
        st.error(f"Could not load the selected run: {error}")
        return

    dashboard_tab, details_tab, comparison_tab, synthetic_tab = st.tabs(
        [
            "Dashboard",
            "Question details",
            "Run comparison",
            "Synthetic test data",
        ]
    )
    with dashboard_tab:
        render_dashboard_tab(summary, scores)
    with details_tab:
        render_question_details_tab(selected_run, scores)
    with comparison_tab:
        render_run_comparison_tab(settings.results_dir)
    with synthetic_tab:
        render_synthetic_data_tab(settings)


if __name__ == "__main__":
    main()
