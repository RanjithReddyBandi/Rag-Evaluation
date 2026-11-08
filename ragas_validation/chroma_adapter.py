"""ChromaDB integration adapter for HR policy retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import chromadb
from chromadb.api.models.Collection import Collection

from .logging_config import configure_logging


LOGGER = configure_logging()


@dataclass(frozen=True)
class RetrievedChunk:
    """Represent one document returned by ChromaDB.

    Attributes:
        chunk_id: ChromaDB document identifier.
        text: Retrieved document text.
        metadata: Filterable document metadata.
        distance: ChromaDB distance returned for the result.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None


class ChromaAdapter:
    """Provide validated access to one persistent ChromaDB collection.

    Purpose:
        Open the configured database and expose a clean retrieval interface
        without coupling evaluation code to the ChromaDB query response shape.

    Parameters:
        db_dir: Directory containing the persistent ChromaDB database.
        collection_name: Name of the collection to retrieve from.

    Exceptions raised:
        ValueError: If db_dir or collection_name is invalid.
        FileNotFoundError: If db_dir does not exist.
        RuntimeError: If ChromaDB cannot open the database or collection.
    """

    def __init__(self, db_dir: str | Path, collection_name: str) -> None:
        LOGGER.info("Application start: initializing Chroma adapter")
        LOGGER.debug(
            "Function entry: ChromaAdapter.__init__",
            extra={"db_dir": str(db_dir), "collection_name": collection_name},
        )

        resolved_db_dir = Path(db_dir).expanduser().resolve()
        normalized_collection_name = collection_name.strip()

        if not normalized_collection_name:
            LOGGER.error("Chroma collection name cannot be empty")
            raise ValueError("Chroma collection name cannot be empty.")
        if not resolved_db_dir.exists():
            LOGGER.error("Chroma database directory does not exist: %s", resolved_db_dir)
            raise FileNotFoundError(
                f"Chroma database directory not found: {resolved_db_dir}"
            )
        if not resolved_db_dir.is_dir():
            LOGGER.error("Chroma database path is not a directory: %s", resolved_db_dir)
            raise ValueError(
                f"Chroma database path is not a directory: {resolved_db_dir}"
            )

        try:
            client = chromadb.PersistentClient(path=str(resolved_db_dir))
            self._collection = client.get_collection(normalized_collection_name)
        except Exception as error:
            LOGGER.exception(
                "Failed to open Chroma collection '%s' at %s",
                normalized_collection_name,
                resolved_db_dir,
            )
            raise RuntimeError(
                f"Could not open Chroma collection '{normalized_collection_name}'."
            ) from error

        self.db_dir = resolved_db_dir
        self.collection_name = normalized_collection_name
        LOGGER.info(
            "Chroma adapter ready: collection=%s records=%d db_dir=%s",
            self.collection_name,
            self._collection.count(),
            self.db_dir,
        )

    @property
    def collection(self) -> Collection:
        """Return the underlying ChromaDB collection for advanced operations.

        Parameters:
            None.

        Returns:
            The opened ChromaDB Collection.

        Exceptions raised:
            None.
        """
        return self._collection

    def count_records(self) -> int:
        """Return the number of records in the collection.

        Parameters:
            None.

        Returns:
            Non-negative number of stored records.

        Raises:
            RuntimeError: If ChromaDB cannot report the collection size.
        """
        LOGGER.debug("Function entry: count_records")
        try:
            record_count = int(self._collection.count())
            LOGGER.debug("Collection record count: %d", record_count)
            return record_count
        except Exception as error:
            LOGGER.exception("Failed to count records in collection '%s'", self.collection_name)
            raise RuntimeError(
                f"Could not count records in collection '{self.collection_name}'."
            ) from error

    def build_metadata_filter(
        self,
        category: str | None = None,
        subcategory: str | None = None,
    ) -> Mapping[str, Any] | None:
        """Build a ChromaDB metadata filter from routing values.

        Parameters:
            category: Optional exact category metadata value.
            subcategory: Optional exact subcategory metadata value.

        Returns:
            A ChromaDB ``where`` mapping, or None when no filters are provided.

        Raises:
            TypeError: If a supplied filter is not a string.
        """
        LOGGER.debug(
            "Function entry: build_metadata_filter",
            extra={"category": category, "subcategory": subcategory},
        )

        conditions: list[dict[str, Any]] = []
        for key, raw_value in (
            ("category", category),
            ("subcategory", subcategory),
        ):
            if raw_value is None:
                continue
            if not isinstance(raw_value, str):
                LOGGER.error("Metadata filter '%s' must be a string", key)
                raise TypeError(f"{key} must be a string when provided.")
            normalized_value = raw_value.strip()
            if not normalized_value:
                LOGGER.warning("Ignoring empty metadata filter: %s", key)
                continue
            conditions.append({key: normalized_value})

        if not conditions:
            LOGGER.debug("No metadata filters were requested")
            return None
        if len(conditions) == 1:
            LOGGER.debug("Built metadata filter: %s", conditions[0])
            return conditions[0]

        filter_mapping: Mapping[str, Any] = {"$and": conditions}
        LOGGER.debug("Built metadata filter: %s", filter_mapping)
        return filter_mapping

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category: str | None = None,
        subcategory: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve normalized chunks for a text query.

        Parameters:
            query: Non-empty text to search for semantically.
            top_k: Maximum number of results. Must be between 1 and 100.
            category: Optional exact category metadata filter.
            subcategory: Optional exact subcategory metadata filter.

        Returns:
            A list of normalized RetrievedChunk objects ordered by ChromaDB.

        Raises:
            ValueError: If query or top_k is invalid.
            RuntimeError: If retrieval fails.
        """
        LOGGER.debug(
            "Function entry: retrieve",
            extra={"query_length": len(query), "top_k": top_k},
        )

        if not isinstance(query, str):
            LOGGER.error("Query must be a string")
            raise ValueError("Query must be a string.")
        normalized_query = query.strip()
        if not normalized_query:
            LOGGER.error("Query cannot be empty")
            raise ValueError("Query cannot be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            LOGGER.error("top_k must be an integer")
            raise ValueError("top_k must be an integer.")
        if top_k < 1 or top_k > 100:
            LOGGER.error("top_k must be between 1 and 100: %d", top_k)
            raise ValueError("top_k must be between 1 and 100.")

        requested_count = min(top_k, self.count_records())
        if requested_count == 0:
            LOGGER.warning("Skipping retrieval because the collection is empty")
            return []

        metadata_filter = self.build_metadata_filter(category, subcategory)
        try:
            response = self._collection.query(
                query_texts=[normalized_query],
                n_results=requested_count,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            LOGGER.exception("Chroma retrieval failed for query='%s'", normalized_query)
            raise RuntimeError("ChromaDB retrieval failed.") from error

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            document = documents[index].strip() if index < len(documents) and documents[index] else ""
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            distance = distances[index] if index < len(distances) else None

            if not document:
                LOGGER.warning("Skipping retrieved empty document: %s", chunk_id)
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=document,
                    metadata=dict(metadata),
                    distance=float(distance) if distance is not None else None,
                )
            )

        LOGGER.info(
            "Retrieved %d chunks: collection=%s query=%r",
            len(results),
            self.collection_name,
            normalized_query,
        )
        return results
