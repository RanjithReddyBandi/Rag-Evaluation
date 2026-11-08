"""Groq-backed routing and answer generation for the HR Helpdesk evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from .chroma_adapter import ChromaAdapter, RetrievedChunk
from .logging_config import configure_logging
from .settings import EvaluationSettings


LOGGER = configure_logging()
ROUTING_MAX_TOKENS = 200
ANSWER_MAX_TOKENS = 700
TEMPERATURE = 0.0


@dataclass(frozen=True)
class AgentResponse:
    """Represent the final result of one agent question.

    Attributes:
        category: Category chosen by intent routing, if any.
        subcategory: Subcategory chosen by intent routing, if any.
        retrieved_chunks: Policy chunks used for answer generation.
        answer: User-facing generated answer.
    """

    category: str | None
    subcategory: str | None
    retrieved_chunks: list[RetrievedChunk]
    answer: str


class GroqHrAgent:
    """Route questions, retrieve policy chunks, and generate grounded answers.

    Purpose:
        Reproduce the evaluator-facing part of an HR agent while using the
        existing ChromaDB collection as its retrieval source.

    Parameters:
        settings: Validated evaluator configuration containing Groq settings.
        chroma_adapter: Adapter for the existing policy collection.

    Exceptions raised:
        ValueError: If settings are incomplete or inputs are invalid.
        RuntimeError: If Groq routing or answer generation fails.
    """

    def __init__(
        self,
        settings: EvaluationSettings,
        chroma_adapter: ChromaAdapter,
    ) -> None:
        LOGGER.info("Application start: initializing Groq HR agent")
        LOGGER.debug("Function entry: GroqHrAgent.__init__")

        settings.require_groq_credentials()
        self._settings = settings
        self._chroma = chroma_adapter
        self._client = OpenAI(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )

        LOGGER.info(
            "Groq HR agent ready: model=%s collection=%s",
            settings.groq_model,
            chroma_adapter.collection_name,
        )

    def _available_routes(self) -> dict[str, list[str]]:
        """Read approved category and subcategory values from Chroma metadata.

        Parameters:
            None.

        Returns:
            A mapping of category names to sorted subcategory names.

        Raises:
            RuntimeError: If Chroma metadata cannot be read.
        """
        LOGGER.debug("Function entry: _available_routes")
        try:
            collection_data = self._chroma.collection.get(include=["metadatas"])
            routes: dict[str, set[str]] = {}
            for metadata in collection_data.get("metadatas", []) or []:
                category = str(metadata.get("category", "")).strip()
                subcategory = str(metadata.get("subcategory", "")).strip()
                if not category:
                    LOGGER.warning("Ignoring Chroma metadata with an empty category")
                    continue
                routes.setdefault(category, set())
                if subcategory:
                    routes[category].add(subcategory)
        except Exception as error:
            LOGGER.exception("Failed to read approved routing metadata")
            raise RuntimeError("Could not read policy routing metadata.") from error

        return {
            category: sorted(subcategories)
            for category, subcategories in sorted(routes.items())
        }

    def _call_chat_model(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        """Call Groq through the OpenAI-compatible chat API.

        Parameters:
            system_prompt: Instructions defining the model's task.
            user_prompt: User content sent to the model.
            max_tokens: Maximum number of output tokens.

        Returns:
            The trimmed text returned by the model.

        Raises:
            ValueError: If prompt arguments or token limits are invalid.
            RuntimeError: If Groq returns no usable response.
        """
        LOGGER.debug(
            "Function entry: _call_chat_model",
            extra={"max_tokens": max_tokens, "user_prompt_length": len(user_prompt)},
        )

        if not isinstance(system_prompt, str) or not system_prompt.strip():
            LOGGER.error("System prompt cannot be empty")
            raise ValueError("System prompt cannot be empty.")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            LOGGER.error("User prompt cannot be empty")
            raise ValueError("User prompt cannot be empty.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            LOGGER.error("max_tokens must be an integer")
            raise ValueError("max_tokens must be an integer.")
        if max_tokens < 1:
            LOGGER.error("max_tokens must be positive")
            raise ValueError("max_tokens must be positive.")

        try:
            response = self._client.chat.completions.create(
                model=self._settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
            )
        except Exception as error:
            LOGGER.exception("Groq chat completion failed")
            raise RuntimeError("The Groq model could not complete the request.") from error

        if not response.choices:
            LOGGER.error("Groq returned no choices")
            raise RuntimeError("The Groq model returned no response.")

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            LOGGER.error("Groq returned empty content")
            raise RuntimeError("The Groq model returned an empty response.")

        LOGGER.info("Groq chat completion succeeded")
        return content.strip()

    def _parse_routing_response(self, model_text: str) -> dict[str, str | None]:
        """Parse the category and subcategory returned by routing JSON.

        Parameters:
            model_text: Text containing a JSON routing object.

        Returns:
            A dictionary containing optional ``category`` and ``subcategory``
            values.

        Raises:
            ValueError: If no valid JSON object can be parsed.
        """
        LOGGER.debug("Function entry: _parse_routing_response")
        decoder = json.JSONDecoder()

        for index, character in enumerate(model_text):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(model_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return {
                    "category": parsed.get("category"),
                    "subcategory": parsed.get("subcategory"),
                }

        LOGGER.error("Could not parse routing JSON: %s", model_text)
        raise ValueError("The routing model did not return valid JSON.")

    def classify_question(self, question: str) -> tuple[str | None, str | None]:
        """Classify a question into one approved policy route.

        Parameters:
            question: Non-empty employee policy question.

        Returns:
            A tuple containing optional category and subcategory values.

        Raises:
            ValueError: If the question is empty or routing JSON is invalid.
            RuntimeError: If Groq routing fails or metadata cannot be read.
        """
        LOGGER.debug("Function entry: classify_question")
        if not isinstance(question, str) or not question.strip():
            LOGGER.error("Classification question cannot be empty")
            raise ValueError("Classification question cannot be empty.")

        available_routes = self._available_routes()
        if not available_routes:
            LOGGER.error("No approved routing metadata was found")
            raise RuntimeError("No policy routing metadata is available.")

        user_prompt = (
            "Choose exactly one approved route for the employee question below.\n"
            f"Approved routes: {json.dumps(available_routes, ensure_ascii=False)}\n"
            "Return only JSON with keys category and subcategory. Use null if "
            "no route is appropriate.\n"
            f"Question: {question.strip()}"
        )
        routing_text = self._call_chat_model(
            system_prompt=(
                "You are an HR policy intent classifier. Select only values from "
                "the approved routes. Never invent category or subcategory names."
            ),
            user_prompt=user_prompt,
            max_tokens=ROUTING_MAX_TOKENS,
        )

        parsed_routing = self._parse_routing_response(routing_text)
        category = parsed_routing["category"]
        subcategory = parsed_routing["subcategory"]

        category = category.strip() if isinstance(category, str) else None
        subcategory = subcategory.strip() if isinstance(subcategory, str) else None

        if category and category not in available_routes:
            LOGGER.warning("Routing model invented unknown category: %s", category)
            category = None
            subcategory = None
        elif category and subcategory not in available_routes.get(category, []):
            LOGGER.warning(
                "Routing model selected unknown subcategory '%s' for '%s'",
                subcategory,
                category,
            )
            subcategory = None

        LOGGER.info(
            "Question classified: category=%s subcategory=%s",
            category,
            subcategory,
        )
        return category, subcategory

    def _build_answer_prompt(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Build an answer-generation prompt containing only retrieved context.

        Parameters:
            question: Employee policy question.
            chunks: Retrieved policy chunks to ground the answer.

        Returns:
            A prompt containing the question and numbered policy excerpts.

        Raises:
            ValueError: If the question is empty.
        """
        LOGGER.debug("Function entry: _build_answer_prompt")
        if not isinstance(question, str) or not question.strip():
            LOGGER.error("Answer question cannot be empty")
            raise ValueError("Answer question cannot be empty.")

        if not chunks:
            return (
                "The employee asked: "
                f"{question.strip()}\n\n"
                "No policy context was retrieved. State that the policy system "
                "does not have enough context and suggest contacting HR."
            )

        excerpts = "\n\n".join(
            f"[{index}] ({chunk.chunk_id}) {chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        )
        return (
            f"Employee question: {question.strip()}\n\n"
            f"Retrieved policy excerpts:\n{excerpts}\n\n"
            "Answer the question concisely using only the retrieved excerpts. "
            "If the excerpts do not contain the answer, say that clearly."
        )

    def answer_question(self, question: str, top_k: int = 3) -> AgentResponse:
        """Classify, retrieve, and answer one employee policy question.

        Parameters:
            question: Non-empty employee policy question.
            top_k: Maximum number of policy chunks to retrieve.

        Returns:
            The routing decisions, retrieved chunks, and generated answer.

        Raises:
            ValueError: If question or top_k is invalid.
            RuntimeError: If classification, retrieval, or answer generation fails.
        """
        LOGGER.info("Processing agent question")
        LOGGER.debug("Function entry: answer_question", extra={"top_k": top_k})
        if not isinstance(question, str) or not question.strip():
            LOGGER.error("Agent question cannot be empty")
            raise ValueError("Agent question cannot be empty.")

        category, subcategory = self.classify_question(question)
        chunks = self._chroma.retrieve(
            query=question.strip(),
            top_k=top_k,
            category=category,
            subcategory=subcategory,
        )
        answer = self._call_chat_model(
            system_prompt=(
                "You are an HR Helpdesk assistant. Use only the supplied policy "
                "context. Do not invent policy details or legal advice."
            ),
            user_prompt=self._build_answer_prompt(question, chunks),
            max_tokens=ANSWER_MAX_TOKENS,
        )

        LOGGER.info(
            "Agent answer generated: category=%s subcategory=%s chunks=%d",
            category,
            subcategory,
            len(chunks),
        )
        return AgentResponse(
            category=category,
            subcategory=subcategory,
            retrieved_chunks=chunks,
            answer=answer,
        )
