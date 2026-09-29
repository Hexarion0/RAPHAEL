import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from raphael.config import get_settings
from raphael.logging import get_logger
from raphael.providers.base import ChatMessage, LLMResponse, LLMStreamChunk
from raphael.providers.manager import ProviderManager

logger = get_logger("providers.router")


class ComplexityLevel(str, Enum):
    """Query complexity categories."""

    SIMPLE = "simple"  # Greetings, time, short factual queries, basic arithmetic
    MEDIUM = "medium"  # Explanations, summaries, multi-step questions
    COMPLEX = "complex"  # Coding, architecture, deep reasoning, long context


@dataclass
class RoutingDecision:
    """Outcome of query analysis."""

    complexity: ComplexityLevel
    provider_name: str
    model_name: str | None
    reason: str
    override_applied: str | None = None


class ModelRouter:
    """Analyzes prompt complexity and routes queries to optimal AI providers/models."""

    # Heuristic patterns for complexity classification
    CODE_KEYWORDS = re.compile(
        r"\b(code|function|class|algorithm|python|javascript|rust|bug|error|refactor|sql|debug|regex|api)\b",
        re.IGNORECASE,
    )
    REASONING_KEYWORDS = re.compile(
        r"\b(analyze|architect|design|compare|evaluate|synthesize|plan|"
        r"step-by-step|pros and cons|optimize)\b",
        re.IGNORECASE,
    )
    SIMPLE_GREETINGS = re.compile(
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|who\s+are\s+you|what\s+time\s+is\s+it|what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+|thanks|thank\s+you)\b",
        re.IGNORECASE,
    )

    def __init__(self, manager: ProviderManager | None = None) -> None:
        self.manager = manager or ProviderManager()

    def classify_complexity(self, prompt: str) -> tuple[ComplexityLevel, str, str | None]:
        """Classify prompt into SIMPLE, MEDIUM, or COMPLEX with explanation and override info."""
        clean = prompt.strip()

        # 1. Manual Overrides
        if clean.startswith("/fast"):
            return ComplexityLevel.SIMPLE, "Manual /fast override requested", "fast"
        if clean.startswith("/strong") or clean.startswith("/deep"):
            return ComplexityLevel.COMPLEX, "Manual /strong override requested", "strong"
        if clean.startswith("/local"):
            return ComplexityLevel.SIMPLE, "Manual /local override requested", "local"

        # 2. Simple Heuristics (short greetings, time, simple math)
        word_count = len(clean.split())
        if word_count <= 8 and self.SIMPLE_GREETINGS.search(clean):
            return ComplexityLevel.SIMPLE, "Short conversational query or basic math", None

        # 3. Complex Heuristics (code requests, architecture, deep analysis)
        if (
            self.CODE_KEYWORDS.search(clean)
            or self.REASONING_KEYWORDS.search(clean)
            or word_count > 40
        ):
            return (
                ComplexityLevel.COMPLEX,
                "Detected code/technical reasoning keywords or long prompt length",
                None,
            )

        # 4. Default: Medium complexity
        return ComplexityLevel.MEDIUM, "Standard explanatory or multi-turn query", None

    def route(self, prompt: str) -> RoutingDecision:
        """Determine the optimal provider and model for a given prompt."""
        complexity, reason, override = self.classify_complexity(prompt)

        # Route by complexity and provider availability
        if override == "local":
            decision = RoutingDecision(
                complexity=complexity,
                provider_name="ollama",
                model_name="llama3.2",
                reason=reason,
                override_applied=override,
            )
        elif complexity == ComplexityLevel.SIMPLE:
            # Prefer ultra-fast inference (Groq) if available, otherwise NIM default
            if self.manager.groq.is_configured():
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="groq",
                    model_name="llama-3.1-8b-instant",
                    reason=f"{reason} → Ultra-low latency via Groq",
                    override_applied=override,
                )
            else:
                nim_model = get_settings().providers.nim_model
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="nim",
                    model_name=nim_model,
                    reason=f"{reason} → Default NIM provider ({nim_model})",
                    override_applied=override,
                )
        elif complexity == ComplexityLevel.COMPLEX:
            # Complex reasoning, coding, architecture -> Ultra model
            if self.manager.nim.is_configured():
                nim_complex_model = get_settings().providers.nim_complex_model
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="nim",
                    model_name=nim_complex_model,
                    reason=f"{reason} → Deep reasoning/coding via NIM ({nim_complex_model})",
                    override_applied=override,
                )
            elif self.manager.openrouter.is_configured():
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="openrouter",
                    model_name="meta-llama/llama-3.1-70b-instruct",
                    reason=f"{reason} → OpenRouter gateway",
                    override_applied=override,
                )
            else:
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="groq",
                    model_name="llama-3.1-8b-instant",
                    reason=f"{reason} → Groq fallback",
                    override_applied=override,
                )
        else:
            # Medium: Standard explanatory / conversation -> Default super model
            if self.manager.nim.is_configured():
                nim_model = get_settings().providers.nim_model
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="nim",
                    model_name=nim_model,
                    reason=f"{reason} → High-capacity conversational model via NIM ({nim_model})",
                    override_applied=override,
                )
            elif self.manager.openrouter.is_configured():
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="openrouter",
                    model_name="meta-llama/llama-3.1-70b-instruct",
                    reason=f"{reason} → OpenRouter gateway",
                    override_applied=override,
                )
            else:
                decision = RoutingDecision(
                    complexity=complexity,
                    provider_name="groq",
                    model_name="llama-3.1-8b-instant",
                    reason=f"{reason} → Groq fallback",
                    override_applied=override,
                )

        logger.info(
            "🔀 Routed prompt [%s] to provider '%s' (Model: %s) — %s",
            decision.complexity.value.upper(),
            decision.provider_name,
            decision.model_name or "default",
            decision.reason,
        )
        return decision

    def send(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = 512,
    ) -> LLMResponse:
        """Route and execute a chat completion with automatic fallback."""
        latest_user_text = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        decision = self.route(latest_user_text)

        # Strip override command prefixes from messages if present
        clean_messages = self._clean_override_prefixes(messages)

        # Execute with manager fallback starting from chosen provider
        return self.manager.send_with_fallback(
            clean_messages,
            preferred_provider=decision.provider_name,
            model=decision.model_name,
            temperature=temperature,
            max_tokens=max_tokens or 512,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int | None = 512,
    ) -> Iterator[LLMStreamChunk]:
        """Route and stream chat completion tokens with automatic fallback."""
        latest_user_text = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        decision = self.route(latest_user_text)
        clean_messages = self._clean_override_prefixes(messages)

        yield from self.manager.stream_with_fallback(
            clean_messages,
            preferred_provider=decision.provider_name,
            model=decision.model_name,
            temperature=temperature,
            max_tokens=max_tokens or 512,
        )

    @staticmethod
    def _clean_override_prefixes(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Strip /fast, /strong, /local command prefixes from message text."""
        cleaned: list[ChatMessage] = []
        for m in messages:
            if m.role == "user":
                content = re.sub(r"^/(fast|strong|deep|local)\s*", "", m.content)
                cleaned.append(ChatMessage(role=m.role, content=content))
            else:
                cleaned.append(m)
        return cleaned
