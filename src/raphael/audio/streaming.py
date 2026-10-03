"""Speak sentence-sized replies while provider tokens continue arriving."""

import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from raphael.logging import get_logger
from raphael.providers.base import ChatMessage, LLMResponse, LLMStreamChunk

logger = get_logger("audio.streaming")
_READERS = threading.BoundedSemaphore(2)
_TAGS = ("think", "thought", "reasoning", "reflection")
_TOKENS = tuple(f"<{tag}>" for tag in _TAGS) + tuple(f"</{tag}>" for tag in _TAGS)


class VisibleText:
    """Discard reasoning and fenced code even when delimiters cross token boundaries."""

    def __init__(self, *, omit_code: bool = True) -> None:
        self.pending = ""
        self.hidden: list[str] = []
        self.code = False
        self.omit_code = omit_code

    def feed(self, text: str, *, final: bool = False) -> str:
        """Return only newly available visible text."""
        self.pending += text
        output = []
        while self.pending:
            lowered = self.pending.lower()
            tokens = ("```",) if self.code else (*_TOKENS, "```")
            token = next((item for item in tokens if lowered.startswith(item)), None)
            if token:
                self.pending = self.pending[len(token):]
                if token == "```":
                    self.code = not self.code
                    if not self.hidden:
                        if not self.omit_code:
                            output.append(token)
                        elif not self.code:
                            output.append(" [code omitted] ")
                elif token.startswith("</"):
                    if self.hidden and self.hidden[-1] == token[2:-1]:
                        self.hidden.pop()
                else:
                    self.hidden.append(token[1:-1])
                continue
            if any(item.startswith(lowered) for item in tokens):
                if final:
                    self.pending = ""
                break
            char, self.pending = self.pending[0], self.pending[1:]
            if not self.hidden and (not self.code or not self.omit_code):
                output.append(char)
        return "".join(output)


class SentenceBuffer:
    """Keep decimals and common abbreviations together; bound unpunctuated chunks."""

    def __init__(self, max_chars: int = 240) -> None:
        self.pending = ""
        self.max_chars = max_chars

    def feed(self, text: str, *, final: bool = False) -> list[str]:
        """Emit complete sentences, or the final unfinished sentence."""
        self.pending += text
        output = []
        while self.pending:
            end = None
            for match in re.finditer(r'[.!?]["\u201d\u2019\x27]*(?=\s)|\n', self.pending):
                prefix = self.pending[:match.end()]
                if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|e\.g|i\.e)\.$", prefix, re.I):
                    continue
                if re.fullmatch(r"\s*\d+\.", prefix):
                    continue
                end = match.end()
                break
            if end is None and len(self.pending) > self.max_chars:
                end = self.pending.rfind(" ", 0, self.max_chars)
                if end <= 0:
                    end = self.max_chars
            if end is None:
                if not final:
                    break
                end = len(self.pending)
            segment, self.pending = self.pending[:end].strip(), self.pending[end:].lstrip()
            if segment:
                output.append(segment)
        return output


@dataclass
class StreamedReply:
    """Generated text and playback outcome, including a canceled partial reply."""

    response: LLMResponse
    canceled: bool = False
    spoken: bool = False
    first_token_seconds: float | None = None
    first_audio_seconds: float | None = None
    total_seconds: float = 0.0
    error: Exception | None = None


def stream_reply(
    router,
    messages: list[ChatMessage],
    tts,
    current: Callable[[], bool],
    *,
    max_tokens: int = 400,
    cancel_event: threading.Event | None = None,
    on_sentence_start: Callable[[str], None] | None = None,
) -> StreamedReply:
    """Read tokens concurrently with playback and return promptly on interruption.

    ``on_sentence_start`` receives intended sentence text after playback starts;
    it does not provide word alignment or verify synthesized pronunciation.

    Two bounded readers cap resource use if a provider cannot immediately abort
    a blocked read. Every reader owns and closes its own generator.
    """
    started = time.monotonic()
    result = StreamedReply(LLMResponse("", "", ""))
    if not current():
        result.canceled = True
        return result
    if not _READERS.acquire(blocking=False):
        raise RuntimeError("Previous provider streams are still closing; try again shortly")
    halted = threading.Event()
    events: queue.Queue = queue.Queue(maxsize=32)
    end = object()
    generation_elapsed: list[float | None] = [None]

    def offer(item) -> None:
        while not halted.is_set():
            try:
                events.put(item, timeout=0.02)
                return
            except queue.Full:
                continue

    def read() -> None:
        iterator = None
        try:
            iterator = router.stream(
                messages, temperature=0.7, max_tokens=max_tokens, cancel_event=halted,
            )
            for chunk in iterator:
                if halted.is_set():
                    break
                offer(chunk)
        except Exception as err:
            offer(err)
        finally:
            try:
                close = getattr(iterator, "close", None)
                if close:
                    close()
            finally:
                generation_elapsed[0] = time.monotonic() - started
                offer(end)
                _READERS.release()

    worker = threading.Thread(target=read, name="raphael-reply-stream", daemon=True)
    try:
        generation = tts.begin_stream()
        worker.start()
    except Exception:
        _READERS.release()
        raise
    visible, audible, sentences = VisibleText(omit_code=False), VisibleText(), SentenceBuffer()
    speech_text = ""

    def audio_started(sentence: str) -> None:
        if result.first_audio_seconds is None:
            result.first_audio_seconds = time.monotonic() - started
            logger.info("Reply audio started after %.2fs.", result.first_audio_seconds)
        if on_sentence_start is not None and current():
            on_sentence_start(sentence)

    def speak(parts: list[str]) -> None:
        for part in parts:
            if not current():
                return
            played = tts.speak(
                part, block=True, cancel_event=cancel_event or halted, generation=generation,
                on_start=lambda sentence=part: audio_started(sentence),
            )
            result.spoken |= bool(played)

    try:
        while current():
            try:
                item = events.get(timeout=0.02)
            except queue.Empty:
                continue
            if item is end or isinstance(item, Exception):
                result.error = item if isinstance(item, Exception) else None
                tail = visible.feed("", final=True)
                result.response.content += tail
                speech_tail = audible.feed(tail, final=True)
                speech_text += speech_tail
                tts.update_stream(generation, speech_text)
                speak(sentences.feed(speech_tail, final=True))
                break
            if not isinstance(item, LLMStreamChunk):
                raise TypeError("Provider emitted an invalid stream chunk")
            result.response.provider, result.response.model = item.provider, item.model
            delta = visible.feed(item.delta)
            if delta and result.first_token_seconds is None:
                result.first_token_seconds = time.monotonic() - started
                logger.info("First reply text arrived after %.2fs.", result.first_token_seconds)
            result.response.content += delta
            speech_delta = audible.feed(delta)
            speech_text += speech_delta
            tts.update_stream(generation, speech_text)
            speak(sentences.feed(speech_delta))
        result.canceled = not current()
        if result.canceled:
            # stop() also invalidates synthesis that has not reached playback yet.
            halted.set()
            tts.stop()
        result.response.content = result.response.content.strip()
        result.total_seconds = time.monotonic() - started
        result.response.latency = generation_elapsed[0] or result.total_seconds
        logger.info(
            "Reply timing: generation=%.2fs, total=%.2fs, canceled=%s.",
            result.response.latency, result.total_seconds, result.canceled,
        )
        if result.error and not result.response.content and not result.canceled:
            raise result.error
        if not result.response.content and not result.canceled:
            raise RuntimeError("Provider stream ended without a visible reply")
        return result
    finally:
        halted.set()
        tts.end_stream(generation)
        worker.join(timeout=0.1)
